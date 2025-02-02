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

from ...nanollama.data.gssm import DataConfig, OnlineDataLoader, init_dataloader_state
from ...nanollama.distributed import ClusterConfig, ClusterManager, is_master_process
from ...nanollama.model.mamba import (
    Mamba,
    MambaConfig,
)
from ...nanollama.monitor import (
    Checkpointer,
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

_logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Configuration Class
# ------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: MambaConfig = field(default_factory=MambaConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # sequence length
        # if not self.model.seq_len:
        #     self.model.seq_len = self.data.seq_len

        # vocabulary size
        if not self.model.vocab_size:
            nodes = self.data.gssm.nodes
            for node in nodes:
                if node.name == "X":
                    break
            _logger.info(f"Setting vocab size to {node.state_dim}")
            self.model.vocab_size = node.state_dim

        # restriction for cpu run
        if self.cluster.device.type == "cpu":
            assert self.optim.fused is False, "Fused Adam is not supported on CPU"
            assert self.orchestration.profiler.active is False, "Profiler is not supported on CPU"

        # check validity of submodule
        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()


# ------------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------------


def loss_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = preds.size(-1)
    return F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1))


def train(config: TrainingConfig) -> None:
    with ExitStack() as context_stack:
        # ---------------------------------------------------------------------
        # Preemption, cluster, logging, and utility contexts
        # ---------------------------------------------------------------------

        preemption: PreemptionHandler = context_stack.enter_context(PreemptionHandler())
        cluster: ClusterManager = context_stack.enter_context(ClusterManager(config.cluster))
        logger: Logger = context_stack.enter_context(Logger(config.orchestration.logging))
        utils: UtilityManager = context_stack.enter_context(UtilityManager(config.orchestration.utils))
        wandb: WandbLogger = context_stack.enter_context(WandbLogger(config.orchestration.wandb, run_config=config))

        # ---------------------------------------------------------------------
        # Build and Parallelize model, as well as Optimizer
        # ---------------------------------------------------------------------

        _logger.info("Building model")
        model = Mamba(config.model)
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

        dataloader: OnlineDataLoader = context_stack.enter_context(
            OnlineDataLoader(
                config=config.data,
                state=state.data,
            )
        )

        # ---------------------------------------------------------------------
        # Training loop with profiling
        # ---------------------------------------------------------------------

        profiler: Profiler = context_stack.enter_context(Profiler(config.orchestration.profiler, state=state))

        logger.report_statistics(model)
        seq_len = config.data.seq_len
        token_per_step = seq_len * config.data.batch_size * config.optim.grad_acc_steps
        profiler.report_statistics(model, token_per_step=token_per_step)

        model.train()

        # aliases
        log_period = config.orchestration.logging.period

        while state.optim.step < config.optim.steps:
            # handle preemption
            if preemption():
                _logger.warning("Preemption flag set")
                break

            # accumulation step
            state.optim.acc_step += 1
            state.optim.acc_step = state.optim.acc_step % config.optim.grad_acc_steps

            # -----------------------------------------------------------------
            # Batch of data (with random state for reproducibility)
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
            state.data.report_restart_info(restart_info)
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
                # For logging we undo that scaling
                loss = loss.detach() * config.optim.grad_acc_steps
                metrics = {
                    "loss": loss.item(),
                    "step": step,
                    "acc_step": state.optim.acc_step,
                    "deterministic_test": batch[0, 0].item(),
                }
                logger(metrics)
                wandb(metrics)

                # log to console
                if is_master_process():
                    _logger.info(f"Step: {metrics['step']}, Loss: {round(metrics['loss'], 4):>7}")

            profiler.end_timer("log_time")

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
        file_config = yaml.safe_load(f)
    if "run_config" in file_config:
        run_config = file_config.pop("run_config")
    else:
        run_config = file_config
    launcher: dict[str, Any] = file_config.pop("launcher", {})

    # casting logging directory to run_config
    if "orchestration" not in run_config:
        run_config["orchestration"] = {}
    for key in ["name", "log_dir"]:
        if key in launcher and key not in run_config["orchestration"]:
            run_config["orchestration"][key] = launcher[key]

    # initialize configuration
    config = initialize_nested_object(TrainingConfig, run_config)

    # Launch training with the config
    train(config)


if __name__ == "__main__":
    main()
