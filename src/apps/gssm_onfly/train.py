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
import signal
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from signal import Signals
from types import FrameType

import torch
import torch.nn.functional as F
import yaml

from ...nanollama.data.gssm import DataConfig, OnlineDataLoaderManager, init_dataloader_state
from ...nanollama.distributed import ClusterConfig, ClusterManager, get_hostname, is_master_process
from ...nanollama.model import Transformer, TransformerConfig
from ...nanollama.monitor import Checkpointer, Logger, OrchestratorConfig, Profiler, UtilityManager, WandbLogger
from ...nanollama.optim import (
    OptimizerConfig,
    init_optimizer,
    init_optimizer_state,
    init_scheduler,
)
from ...nanollama.utils import TrainState, initialize_nested_object

_logger = logging.getLogger("nanollama")


# -------------------------------------------------------------------------------
# Configuration Class
# -------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: TransformerConfig = field(default_factory=TransformerConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # sequence length
        if not self.model.seq_len:
            self.model.seq_len = self.data.seq_len

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


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def loss_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = preds.size(-1)
    return F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1))


def train(config: TrainingConfig) -> None:
    # -------------------------------------------------------------------------
    # Handle preemption
    # -------------------------------------------------------------------------

    preemption_flag = dict(flag=False)

    def signal_handler(signum: Signals, frame: FrameType):
        _logger.warning("Signal handler called with signal " + str(signum))
        preemption_flag["flag"] = True

    def term_handler(signum: Signals, frame: FrameType):
        _logger.warning("Received termination signal " + str(signum))
        # do not requeue to avoid requeuing on `scancel`

    signal.signal(signal.SIGUSR1, signal_handler)
    signal.signal(signal.SIGTERM, term_handler)

    with ExitStack() as context_stack:
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

        dataloader: OnlineDataLoaderManager = context_stack.enter_context(
            OnlineDataLoaderManager(
                config=config.data,
                state=state.data,
            )
        )

        # ---------------------------------------------------------------------
        # Global information
        # ---------------------------------------------------------------------

        profiler: Profiler = context_stack.enter_context(Profiler(config.orchestration.profiler, state=state))

        profiler.report_statistics()
        logger.report_statistics(model)

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------

        model.train()

        while state.optim.step < config.optim.steps:
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

            profiler.start_timer()
            checkpoint()
            profiler()
            utils()
            profiler.end_timer("monitor_time")

            # -----------------------------------------------------------------
            # Log metrics
            # -----------------------------------------------------------------

            profiler.start_timer()

            if state.optim.step % config.orchestration.logging.period == 0:
                # For logging we undo that scaling
                loss = loss.detach() * config.optim.grad_acc_steps
                metrics = {
                    "loss": loss.item(),
                    "step": state.optim.step,
                    "acc_step": state.optim.acc_step,
                    "deterministic_test": batch[0, 0].item(),
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

            # if trigger_update(state, config.monitor.evaluation.period):
            #     pass

            # -----------------------------------------------------------------
            # Handle preemption
            # -----------------------------------------------------------------

            if preemption_flag["flag"]:
                break

    if preemption_flag["flag"]:
        prod_id = int(os.environ["SLURM_PROCID"])
        _logger.warning(f"Host: {get_hostname()} - Global rank: {prod_id}")
        if prod_id == 0:
            _logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
            os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
        else:
            _logger.warning("Not the master process, no need to requeue.")
        sys.exit(0)

    _logger.info("Training done.")


def main() -> None:
    """
    Launch a training job from configuration file specified by cli argument.

    Usage:
    ```
    python -m apps.my_app.train apps/my_app/configs/debug.yaml
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
    with open(path) as f:
        file_config = yaml.safe_load(f)

    # casting logging directory to run_config
    if "orchestration" not in run_config:
        run_config["orchestration"] = {}
    if "launcher" in file_config:
        for key in ["name", "log_dir"]:
            if key in file_config["launcher"]:
                run_config["orchestration"][key] = file_config["launcher"][key]

    # initialize configuration
    config = initialize_nested_object(TrainingConfig, run_config)

    # Launch training with the config
    train(config)


if __name__ == "__main__":
    main()
