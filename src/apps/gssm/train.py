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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ...nanollama.data.hdf5 import DataConfig, FileDataLoader, init_dataloader_state
from ...nanollama.distributed import ClusterConfig, ClusterManager, clean_environment, is_master_process
from ...nanollama.launcher import LauncherConfig, SlurmConfig, launch_job
from ...nanollama.model import Transformer, TransformerConfig
from ...nanollama.monitor import (
    Checkpointer,
    Logger,
    LoggerConfig,
    OrchestratorConfig,
    PreemptionHandler,
    Profiler,
    ProfilerConfig,
    UtilityConfig,
    UtilityManager,
    WandbConfig,
    WandbLogger,
)
from ...nanollama.optim import (
    OptimizerConfig,
    init_optimizer,
    init_optimizer_state,
    init_scheduler,
)
from ...nanollama.utils import TrainState, flatten_config, initialize_nested_object, unflatten_config
from .evaluation import EvaluationConfig, EvaluationRunConfig, run_evaluation

_logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Configuration Class
# ------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    model: TransformerConfig = field(default_factory=None)
    model_gen: callable = field(init=False, default=None)

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # restriction for cpu run
        if self.cluster.device.type == "cpu":
            assert self.optim.fused is False, "Fused Adam is not supported on CPU"
            assert self.orchestration.profiler.active is False, "Profiler is not supported on CPU"

        # evaluation paths
        self.evaluation.path = self.orchestration.logging.metric_path

        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()


@dataclass
class TransformerTrainingConfig(TrainingConfig):
    model: TransformerConfig = field(default_factory=TransformerConfig)
    model_gen: callable = field(init=False, default=Transformer)


def mamba_config_gen() -> Any:
    """
    Wrapper to load mamba packages only if needed.
    """
    from ...nanollama.model.mamba import Mamba, MambaConfig

    @dataclass
    class MambaTrainingConfig(TrainingConfig):
        model: MambaConfig = field(default_factory=MambaConfig)
        model_gen: callable = field(init=False, default=Mamba)

    return MambaTrainingConfig


def rnn_config_gen() -> Any:
    """
    Wrapper to load fastRNN packages only if needed.
    """
    os.environ["CUDA_HOME"] = "/public/apps/cuda/12.2.0"  # monkey patching for accelerated_scan to work
    from ...nanollama.model.rnn import FastRNNConfig, Hawk, MinGRU, MinLSTM

    @dataclass
    class FastRNNTrainingConfig(TrainingConfig):
        model: FastRNNConfig = field(default_factory=FastRNNConfig)
        model_gen: callable = field(init=False, default=None)

        def __post_init__(self):
            self.model_gen = dict(hawk=Hawk, mingru=MinGRU, minlstm=MinLSTM)[self.model.implementation]
            super().__post_init__()

    return FastRNNTrainingConfig


def config_inheritance(train_config: dict[str, Any], eval_config: dict[str, Any]) -> None:
    """
    Cast training configuration arguments into evaluation configuration.
    """
    if eval_config.get("period", 0) <= 0:
        train_config["evaluation"] = eval_config
        return

    # flatten configurations for easier access
    flat_config = flatten_config(train_config)
    eval_config = flatten_config(eval_config)

    # special inheritance
    # orchestration
    eval_config["orchestration.name"] = flat_config["orchestration.name"] + "_eval"
    eval_config["orchestration.parent_dir"] = flat_config["orchestration.log_dir"]
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    eval_config["orchestration.log_dir"] = str(Path(flat_config["orchestration.log_dir"]) / "evals" / task_id)
    eval_config["orchestration.task_id"] = int(task_id)

    # generic inheritance
    configs_keys = [
        (DataConfig, "data"),
    ]

    if eval_config.get("asynchronous"):
        configs_keys += [
            (SlurmConfig, "slurm"),
            (ClusterConfig, "cluster"),
            (LoggerConfig, "orchestration.logging"),
            (ProfilerConfig, "orchestration.profiler"),
            (UtilityConfig, "orchestration.utils"),
            (WandbConfig, "orchestration.wandb"),
        ]

    for config_cls, cls_key in configs_keys:
        for key, finfo in config_cls.__dataclass_fields__.items():
            if not finfo.init:
                continue
            flat_key = f"{cls_key}.{key}"
            if flat_key not in eval_config and flat_key in flat_config:
                eval_config[flat_key] = flat_config[flat_key]

    # merge configuration
    train_config["evaluation"] = unflatten_config(eval_config)


# ------------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------------


def loss_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = preds.size(-1)
    return F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1))


def train(config: TrainingConfig) -> None:
    with ExitStack() as context_stack:
        # ---------------------------------------------------------------------
        # Handle preemption, computing environment, logging, and utils
        # ---------------------------------------------------------------------

        preemption: PreemptionHandler = context_stack.enter_context(PreemptionHandler())
        cluster: ClusterManager = context_stack.enter_context(ClusterManager(config.cluster))
        logger: Logger = context_stack.enter_context(Logger(config.orchestration.logging))
        utils: UtilityManager = context_stack.enter_context(UtilityManager(config.orchestration.utils))
        wandb: WandbLogger = context_stack.enter_context(
            WandbLogger(
                config.orchestration.wandb,
                run_config=flatten_config(asdict(config), flatten_list=True),
            )
        )

        # ---------------------------------------------------------------------
        # Build and Parallelize model, optimizer, scheduler
        # ---------------------------------------------------------------------

        _logger.info("Building model")
        model: nn.Module = config.model_gen(config.model)
        model = cluster.initialize_model(model)

        _logger.info("Building optimizer")
        optimizer = init_optimizer(model, config.optim)
        scheduler = init_scheduler(optimizer, config.optim)
        _logger.info("Done building optimizer")

        # ---------------------------------------------------------------------
        # Recover Checkpoint
        # ---------------------------------------------------------------------

        state = TrainState(
            data=init_dataloader_state(config.data),
            optim=init_optimizer_state(),
        )

        checkpoint: Checkpointer = context_stack.enter_context(
            Checkpointer(
                config.orchestration.checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, state=state
            )
        )

        # ---------------------------------------------------------------------
        # DataLoader
        # ---------------------------------------------------------------------

        dataloader: FileDataLoader = context_stack.enter_context(
            FileDataLoader(
                config=config.data,
                state=state.data,
            )
        )

        # ---------------------------------------------------------------------
        # Global information
        # ---------------------------------------------------------------------

        profiler: Profiler = context_stack.enter_context(Profiler(config.orchestration.profiler, state=state))

        logger.report_statistics(model)
        seq_len = 0  # config.model.block.seq_len
        token_per_step = seq_len * config.data.batch_size * config.optim.grad_acc_steps
        profiler.report_statistics(model, token_per_step=token_per_step, seq_len=seq_len)

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------

        model.train()

        # aliases
        log_period = config.orchestration.logging.period
        eval_period = config.evaluation.period

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
            if cluster.device.type != "cpu":
                batch = batch.pin_memory()
            profiler.end_timer("data_cpu_time")

            profiler.start_timer()
            batch = batch.to(device=cluster.device, non_blocking=True)
            X_batch = batch[:, :-1]
            y_batch = batch[:, 1:]
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
            state.data.report_restart_info(*restart_info)
            state.optim.step += 1

            profiler.end_timer("model_time", sync=True)

            # -----------------------------------------------------------------
            # Call monitors for garbage collection, checkpointing...
            # -----------------------------------------------------------------

            # alias
            step = state.optim.step

            profiler.start_timer()
            checkpoint()
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
                    "deterministic_test": batch[0, 1].item(),
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
                # run evaluation now
                if not config.evaluation.asynchronous:
                    run_evaluation(config.evaluation, model=model, step=step)

                # launch evaluation job on slurm
                elif is_master_process():
                    # checkpoint
                    checkpoint.update(eval=True)

                    # alias
                    orchestration_config = config.evaluation.orchestration
                    orchestration_config.log_dir = str(Path(orchestration_config.log_dir) / f"{step:010d}")

                    # launcher config
                    launch_config = initialize_nested_object(
                        LauncherConfig,
                        {
                            "name": orchestration_config.name,
                            "log_dir": orchestration_config.log_dir,
                            "overwrite": False,
                            "copy_code": False,
                            "script": "src.apps.gssm.evaluation",
                            "slurm": config.evaluation.slurm.to_dict(),
                        },
                    )

                    # run config
                    orchestration_config.train_step = step
                    eval_config: EvaluationRunConfig = {
                        "model": asdict(config.model),
                        "data": asdict(config.evaluation.data),
                        "cluster": config.evaluation.cluster.to_dict(),
                        "orchestration": orchestration_config.to_dict(),
                    }

                    # luanch job without device binding
                    with clean_environment():
                        launch_job(launch_config, eval_config)

                    orchestration_config.log_dir = str(Path(orchestration_config.log_dir).parent)

            profiler.end_timer("evaluation")

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
    eval_config = run_config.pop("evaluation", {})
    run_config["slurm"] = launcher.pop("slurm", {})
    config_inheritance(run_config, eval_config)
    run_config.pop("slurm")

    # grid id system to handle multiple datasets
    grid_id = run_config.get("grid_id", 0)
    run_config["data"]["path"] = run_config["data"]["path"].replace("$GRIDID", str(grid_id))
    run_config["evaluation"]["data"]["path"] = run_config["evaluation"]["data"]["path"].replace("$GRIDID", str(grid_id))

    # initialize configuration
    implementation = run_config.get("model", {}).get("implementation", "").lower()
    if implementation == "mamba":
        config_gen = mamba_config_gen()
    elif implementation in ["hawk", "mingru", "minlstm"]:
        config_gen = rnn_config_gen()
    else:
        config_gen = TransformerTrainingConfig
    config = initialize_nested_object(config_gen, run_config)

    # launch job
    train(config)


if __name__ == "__main__":
    main()
