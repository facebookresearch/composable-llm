import inspect
import json
import logging
import os
import shlex
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import wandb
from evals.api import Prediction, Predictor, TaskConfig
from evals.tasks import TaskRegistry, build_task
from evals.utils.common import (
    filter_by_pattern,
    flatten_dict,
    format_dict,
    get_git_info,
    get_global_rank,
    get_job_info,
    get_random_state,
    get_version,
    unroll_configs,
)
from evals.utils.io import write_to_json, write_to_jsonl

from ..generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
)

# from apps.main.transformer import LMTransformer, LMTransformerArgs
# from difformer.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
# from difformer.metrics import WandbArgs

logger = logging.getLogger(__name__)
EVAL_FOLDER_NAME = "{:010d}"


@dataclass
class EvalArgs:
    name: str = "evals"
    dataset_dir: str = ""
    mp_size: int = 1
    period: int = 0

    task_configs: dict[str, Any] = field(default_factory=dict)
    tasks: Optional[str] = None
    task_args: Optional[dict[str, Any]] = None

    ckpt_dir: str = ""
    generator: PackedCausalTransformerGeneratorArgs = field(default_factory=PackedCausalTransformerGeneratorArgs)
    seed: int = 42

    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None
    no_resume: Optional[bool] = False

    # wandb: Optional[WandbArgs] = None

    max_samples: Optional[int] = None
    show_progress: bool = False

    global_step: Optional[int] = None  # for in-training evaluation

    def __post_init__(self) -> None:
        self.metric_log_dir = self.metric_log_dir or self.dump_dir


def batched(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


# Light wrapper around generator for internal evals
class InternalEvalsTransformer(Predictor):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def from_config(config):
        pass

    @torch.inference_mode()
    def __call__(
        self,
        prompts: list[str],
        max_prompt_len: int,
        max_gen_len: int,
        temperature: float,
        top_p: float,
        top_k: int,
        echo: bool,
        return_logprobs: bool,
        show_progress: bool,
        **kwargs: Any,
    ) -> list[Prediction]:
        # Temperature, top_p and top_k already configured in
        # the generator object, we only pass in max_gen_len and max_prompt_len
        self.generator.max_gen_len = max_gen_len
        self.generator.max_prompt_len = max_prompt_len
        self.generator.show_progress = show_progress
        generations, logprobs, _ = self.generator.generate(prompts)
        if echo:
            generations = [p + g for p, g in zip(prompts, generations)]
        predictions = []
        for i, (prompt, result, ll) in enumerate(zip(prompts, generations, logprobs)):
            detoks, text_offsets = self.generator.tokenizer.get_token_offsets(prompt)
            predictions.append(
                Prediction(
                    text=result,
                    tokens=detoks,
                    text_offsets=text_offsets,
                    logprobs=ll.tolist() if return_logprobs else None,
                )
            )
        return predictions


os.environ["MPU_MODULE"] = "apps.main.internal.mpu_utils"


def task_configs(args: EvalArgs) -> dict[str, TaskConfig]:
    configs: dict[str, TaskConfig] = {}  # {"ppl": args.ppl} if args.ppl else {}
    for name in filter_by_pattern(TaskRegistry.names(), args.tasks):
        defaults = inspect.signature(TaskRegistry._REGISTRY[name]).parameters
        params = (args.task_args or {}).get(name, {})
        try:
            know_config, unknow_config = unroll_configs(
                defaults, params, prefix=name, params_to_skip_adding_in_task_name=[]
            )
        except TypeError:
            # depreciated version
            know_config, unknow_config = unroll_configs(defaults, params, prefix=name)

        for fname, kwargs in know_config.items():
            if "dataset_dir" in defaults:
                kwargs["dataset_dir"] = args.dataset_dir
            configs[fname] = TaskRegistry.get_config(name, **kwargs)
    return configs


def eval_model(cfg: EvalArgs, generator: PackedCausalTransformerGenerator) -> None:
    cfg.task_configs = task_configs(cfg)

    torch.manual_seed(cfg.seed)
    if cfg.dump_dir is not None and get_global_rank() == 0:
        metadata = {
            "timestamp": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "command": " ".join(map(shlex.quote, sys.argv)),
            "job_info": get_job_info(),
            "git_info": get_git_info(),
            "config": asdict(cfg),
            "task_configs": cfg.task_configs,
        }
        metadata_file = os.path.join(cfg.dump_dir, "metadata.jsonl")
        logger.info(f"Writing configs and metadata to {metadata_file}")
        write_to_jsonl([metadata], metadata_file, mode="a")
        # ckpt_cfg = OmegaConf.load(Path(cfg.ckpt_dir) / "params.json")
        # ckpt_cfg = OmegaConf.to_container(ckpt_cfg)
        # if cfg.wandb is not None:
        #     wandb.init(config=ckpt_cfg, **asdict(cfg.wandb))

    logger.info(f"Evals version {get_version()} ({Path(__file__).parent.parent})")
    logger.info(f"Config: {asdict(cfg)}")
    assert cfg.task_configs, f"No tasks were found given pattern '{cfg.tasks}'"
    logger.info(f"Selected tasks for execution: {list(cfg.task_configs)}")

    # torch.distributed.barrier()

    predictor = InternalEvalsTransformer(generator)

    metrics: dict[str, dict[str, float]] = {}
    for name, task_config in cfg.task_configs.items():
        start = time.monotonic()
        logger.info(f"Running evaluation on task {name}")
        if cfg.dump_dir is not None:
            result_file = os.path.join(cfg.dump_dir, "results", f"{name}.json")
            if not cfg.no_resume and os.path.exists(os.path.expandvars(result_file)):
                logger.info(f"Loading cached evaluation results from {result_file}")
                with open(os.path.expandvars(result_file)) as f:
                    metrics[name] = json.load(f)["results"]

        if name not in metrics:
            task = build_task(task_config)

            result = task.run(
                predictor=predictor,
                random_state=get_random_state(cfg.seed),
                max_samples=cfg.max_samples,
                show_progress=cfg.show_progress,
                cache_prefix=(
                    f"{cfg.dump_dir}/cache/{name}" if cfg.dump_dir is not None and not cfg.no_resume else None
                ),
            )
            # Make sure all ranks complete the task before moving on to the next one
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            metrics[name] = {k: v.value for k, v in result.metrics.items()}
            if cfg.dump_dir and result.raw_results and get_global_rank() == 0:
                raw_file = os.path.join(cfg.dump_dir, "raw_results", f"{name}.jsonl")
                logger.info(f"Writing raw results to {raw_file}")
                write_to_jsonl(result.raw_results, raw_file)

                result_content = {"results": metrics[name], "configs": task_config}
                logger.info(f"Writing metric results to {result_file}")
                write_to_json(result_content, result_file, indent=4)

                if cfg.wandb is not None and wandb.run is not None and get_global_rank() == 0:
                    wandb.log(
                        {
                            f"evals/{name.replace('/', '.')}.{metric.replace('/', '.')}": value
                            for metric, value in metrics[name].items()
                        },
                        step=cfg.global_step,
                    )

        log = format_dict(flatten_dict(metrics[name]), delimiter=" | ", decimal=6)
        logger.info(f"Evaluation results on task {name}: {log}")
        logger.info(f"Task {name} took {time.monotonic() - start:.2f} seconds")
        torch.cuda.empty_cache()

    results = flatten_dict(metrics)
    logger.info(f"All evaluation results: {format_dict(results)}")
    if cfg.metric_log_dir and get_global_rank() == 0:
        metric_log_path = os.path.join(cfg.metric_log_dir, "metrics.eval.jsonl")
        logger.info(f"Writing metric logs to {metric_log_path}")
        timestamp = {
            "created_at": datetime.utcnow().isoformat(),
        }
        if cfg.global_step is not None:
            timestamp["global_step"] = cfg.global_step
        write_to_jsonl([timestamp | results], metric_log_path, mode="a")

    del predictor
    torch.cuda.empty_cache()


def launch_eval(cfg: EvalArgs, model, tokenizer):
    generator = PackedCausalTransformerGenerator(cfg.generator, model, tokenizer)
    eval_model(cfg, generator)
