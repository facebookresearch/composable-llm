"""
Training script with online generation of batch of data.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import logging
import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from ...nanollama.data.text import (
    DataArgs,
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)
from ...nanollama.data.tokenizer import build_tokenizer
from ...nanollama.distributed import (
    ClusterConfig,
    ClusterManager,
    get_rank,
    get_world_size,
    is_master_process,
)
from ...nanollama.model import Transformer, TransformerConfig
from ...nanollama.monitor import (
    # Checkpointer,
    Logger,
    OrchestratorConfig,
    PreemptionHandler,
    Profiler,
    UtilityManager,
    WandbLogger,
)
from ...nanollama.optim import (
    OptimizerConfig,
    init_optimizer,
    init_optimizer_state,
    init_scheduler,
)
from ...nanollama.utils import TrainState, initialize_nested_object
from .internal.eval import (
    EVAL_FOLDER_NAME,
    EvalArgs,
    launch_eval,
)

# from .evaluation import EvaluationConfig, EvaluationRunConfig, run_evaluation

_logger = logging.getLogger("nanollama")


# -----------------------------------------------------------------------------
# Configuration Class
# -----------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    data: DataArgs = field(default_factory=DataArgs)
    model: TransformerConfig = field(default_factory=TransformerConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    eval: EvalArgs = field(default_factory=EvalArgs)
    orchestration: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    tokenizer: Any = field(init=False)

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # restriction for cpu run
        if self.cluster.device.type == "cpu":
            assert self.optim.fused is False, "Fused Adam is not supported on CPU"
            assert self.orchestration.profiler.active is False, "Profiler is not supported on CPU"

        # vocabulary size
        self.tokenizer = build_tokenizer(self.data.tokenizer.name, self.data.tokenizer.path)
        self.model.vocab_size = self.tokenizer.n_words
        # evaluation paths
        # self.evaluation.path = self.orchestration.logging.metric_path

        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def loss_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = preds.size(-1)
    return F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1))


def train(config: TrainingConfig) -> None:
    with ExitStack() as context_stack:
        # ---------------------------------------------------------------------
        # Handle preemption
        # ---------------------------------------------------------------------

        preemption: PreemptionHandler = context_stack.enter_context(PreemptionHandler())

        # ---------------------------------------------------------------------
        # Computing Environment
        # ---------------------------------------------------------------------

        cluster: ClusterManager = context_stack.enter_context(ClusterManager(config.cluster))

        # ---------------------------------------------------------------------
        # Monitor: logging, and utils
        # ---------------------------------------------------------------------

        logger: Logger = context_stack.enter_context(Logger(config.orchestration.logging))
        utils: UtilityManager = context_stack.enter_context(UtilityManager(config.orchestration.utils))
        wandb: WandbLogger = context_stack.enter_context(WandbLogger(config.orchestration.wandb, run_config=config))

        # ---------------------------------------------------------------------
        # Build and Parallelize model
        # ---------------------------------------------------------------------

        _logger.info("Building model")
        model = Transformer(config.model)
        model = cluster.initialize_model(model)

        # ---------------------------------------------------------------------
        # Build Optimizer
        # ---------------------------------------------------------------------

        _logger.info("Building optimizer")
        optimizer = init_optimizer(model, config.optim)
        scheduler = init_scheduler(optimizer, config.optim)
        _logger.info("Done building optimizer")

        # ---------------------------------------------------------------------
        # Recover Checkpoint
        # ---------------------------------------------------------------------

        state = TrainState(
            data=init_dataloader_state_from_args(config.data, get_rank(), get_world_size()),
            optim=init_optimizer_state(),
        )

        # checkpoint: Checkpointer = context_stack.enter_context(
        #     Checkpointer(
        #         config.orchestration.checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, state=state
        #     )
        # )

        # ---------------------------------------------------------------------
        # DataLoader
        # ---------------------------------------------------------------------

        dataloader = context_stack.enter_context(
            build_dataloader_from_args(
                config.data,
                state=state.data,
            )
        )

        # ---------------------------------------------------------------------
        # Global information
        # ---------------------------------------------------------------------

        profiler: Profiler = context_stack.enter_context(Profiler(config.orchestration.profiler, state=state))

        # TODO for flops calculation
        token_per_step = 0
        profiler.report_statistics(model, token_per_step)
        logger.report_statistics(model)

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------

        model.train()

        # aliases
        log_period = config.orchestration.logging.period
        eval_period = config.eval.period

        while state.optim.step < config.optim.steps:
            # handle preemption
            if preemption():
                _logger.warning("Preemption flag set")
                break

            # accumulation step
            state.optim.acc_step += 1
            state.optim.acc_step = state.optim.acc_step % config.optim.grad_acc_steps

            # -----------------------------------------------------------------
            # Batch of data (with reproducibility information)
            # -----------------------------------------------------------------

            profiler.start_timer()
            batch, restart_info = next(dataloader)
            batch = torch.from_numpy(batch).to(cluster.device)
            profiler.end_timer("data_cpu_time")

            profiler.start_timer()
            batch = batch.to(device=cluster.device, non_blocking=True)
            X_batch = batch[..., 0]
            y_batch = batch[..., 1]
            profiler.end_timer("data_io_time")

            # -----------------------------------------------------------------
            # Forward and backward pass
            # -----------------------------------------------------------------

            profiler.start_timer()

            # forward propagation
            preds = model(X_batch)
            loss = loss_func(preds, y_batch)

            # rescale when using gradient accumulation (backprop on mean, not sum)
            loss = loss / config.optim.grad_acc_steps

            # backward propagation
            loss.backward()

            # gradient accumulation
            if state.optim.acc_step != 0:
                continue

            # optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # state.data.report_restart_info(*restart_info)
            state.optim.step += 1

            profiler.end_timer("model_time", sync=True)

            # -----------------------------------------------------------------
            # Call monitors for garbage collection, checkpointing...
            # -----------------------------------------------------------------

            # alias
            step = state.optim.step

            profiler.start_timer()
            # checkpoint()
            profiler()
            utils()
            profiler.end_timer("monitor_time")

            # -----------------------------------------------------------------
            # Log metrics
            # -----------------------------------------------------------------

            profiler.start_timer()

            if log_period > 0 and step % log_period == 0:
                # For logging we undo gradient accumulation scaling
                loss = loss.detach() * config.optim.grad_acc_steps
                metrics = {
                    "loss": loss.item(),
                    "step": step,
                    "acc_step": state.optim.acc_step,
                    # "deterministic_test": batch[0, 0].item(),
                }
                logger(metrics)
                wandb(metrics)

                # log to console
                if is_master_process():
                    _logger.info(f"Step: {metrics['step']}, Loss: {round(metrics['loss'], 4):>7}")

            profiler.end_timer("log_time")

            # -----------------------------------------------------------------
            # Evaluation
            # -----------------------------------------------------------------

            profiler.start_timer()

            if eval_period > 0 and step % eval_period == 0:
                eval_args = config.eval
                eval_args.global_step = step
                # checkpoint.update()
                # eval_args.ckpt_dir = str(checkpoint.existing_saves[-1])
                eval_args.dump_dir = str(
                    os.path.join(
                        config.orchestration.log_dir,
                        "evals",
                        EVAL_FOLDER_NAME.format(step),
                    )
                )
                eval_args.metric_log_dir = config.orchestration.log_dir

                launch_eval(eval_args, model, config.tokenizer)
                eval_args.task_configs = {}
                model.train()

            #     # run evaluation now
            #     if not config.evaluation.asynchronous:
            #         run_evaluation(config.evaluation, model=model, step=step)

            #     # launch evaluation job on slurm
            #     elif is_master_process():
            #         # checkpoint
            #         checkpoint.update(eval=True)

            #         # alias
            #         orchestration_config = config.evaluation.orchestration
            #         orchestration_config.log_dir = str(Path(orchestration_config.log_dir) / f"{step:010d}")

            #         # launcher config
            #         launch_config = initialize_nested_object(
            #             LauncherConfig,
            #             {
            #                 "name": orchestration_config.name,
            #                 "log_dir": orchestration_config.log_dir,
            #                 "overwrite": False,
            #                 "copy_code": False,
            #                 "script": "src.apps.gssm.evaluation",
            #                 "slurm": config.evaluation.slurm.to_dict(),
            #             },
            #         )

            #         # run config
            #         orchestration_config.train_step = step
            #         eval_config: EvaluationRunConfig = {
            #             "model": asdict(config.model),
            #             "data": asdict(config.evaluation.data),
            #             "cluster": config.evaluation.cluster.to_dict(),
            #             "orchestration": orchestration_config.to_dict(),
            #         }

            #         # luanch job without device binding
            #         with clean_environment():
            #             launch_job(launch_config, eval_config)

            #         orchestration_config.log_dir = str(Path(orchestration_config.log_dir).parent)

            # profiler.end_timer("evaluation")

    _logger.info("Training done.")


def main() -> None:
    """
    Launch a training job from configuration file specified by cli argument.

    Usage:
    ```
    python -m apps.my_app.train apps/my_app/configs/my_config.yaml
    ```
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # parse file configuration path
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", type=str, help="Path to configuration file")
    path = parser.parse_args().config

    # obtain configuration from file
    with open(os.path.expandvars(path)) as f:
        file_config: dict[str, Any] = yaml.safe_load(f)
    if "run_config" in file_config:
        run_config: dict[str, Any] = file_config.pop("run_config")
    else:
        run_config = file_config
    launcher: dict[str, Any] = file_config.pop("launcher", {})

    # casting logging directory to run_config
    if "orchestration" not in run_config:
        run_config["orchestration"] = {}
    for key in ["name", "log_dir"]:
        if key in launcher and key not in run_config["orchestration"]:
            run_config["orchestration"][key] = launcher[key]

    # configuration inheritance between training and evaluation
    # eval_config = run_config.pop("evaluation", {})
    # run_config["slurm"] = launcher.pop("slurm", {})
    # config_inheritance(run_config, eval_config)
    # run_config.pop("slurm")

    # initialize configuration
    config = initialize_nested_object(TrainingConfig, run_config)

    # launch job
    train(config)


if __name__ == "__main__":
    main()
