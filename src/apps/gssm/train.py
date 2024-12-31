"""
Training Script.

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

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ...nanollama.cluster import ClusterConfig, ClusterManager, get_hostname, is_master_process
from ...nanollama.data.gssm import DataConfig, OnlineDataLoaderManager, init_dataloader_state
from ...nanollama.model import Transformer, TransformerConfig
from ...nanollama.monitor import MonitorConfig, Orchestrator
from ...nanollama.optim import (
    OptimizerConfig,
    init_optimizer,
    init_optimizer_state,
    init_scheduler,
)
from ...nanollama.train import TrainState
from ...nanollama.utils import trigger_update

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------
# Configuration Class
# -------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: TransformerConfig = field(default_factory=TransformerConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    def __manual_post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__manual_post_init__"):
                module.__manual_post_init__()

        # Sequence length
        if self.model.seq_len == -1 and self.data.seq_len == -1:
            raise ValueError("seq_len must be provided in either model or data")
        if self.model.seq_len == -1:
            self.model.seq_len = self.data.seq_len
        if self.data.seq_len == -1:
            self.data.seq_len = self.model.seq_len

        # TODO: vocabulary size


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def loss_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = preds.size(-1)
    return F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1))


def train(config: TrainingConfig):
    # -------------------------------------------------------------------------
    # Handle preemption
    # -------------------------------------------------------------------------

    preemption_flag = dict(flag=False)

    def signal_handler(signum, frame):
        logger.warning("Signal handler called with signal " + str(signum))
        preemption_flag["flag"] = True

    def term_handler(signum, frame):
        logger.warning("Received termination signal " + str(signum))
        # do not requeue to avoid requeuing on `scancel`

    signal.signal(signal.SIGUSR1, signal_handler)
    signal.signal(signal.SIGTERM, term_handler)

    with ExitStack() as context_stack:
        # ---------------------------------------------------------------------
        # Computing Environment
        # ---------------------------------------------------------------------

        cluster = context_stack.enter_context(ClusterManager(config.cluster))

        # ---------------------------------------------------------------------
        # Monitor: checkpointing, profiling, probing, logging
        # ---------------------------------------------------------------------

        monitor = context_stack.enter_context(Orchestrator(config.monitor))

        # ---------------------------------------------------------------------
        # Build and Parallelize model
        # ---------------------------------------------------------------------

        logger.info("Building model")
        model = Transformer(config.model)
        model.to(device=cluster.device)
        if config.cluster.compile_model:
            model = torch.compile(model)
        logger.info("Done building model")

        # Parallelize model
        model = cluster.parallelize_model(model)

        # ---------------------------------------------------------------------
        # Build Optimizer
        # ---------------------------------------------------------------------

        logger.info("Building optimizer")
        optimizer = init_optimizer(model, config.optim)
        scheduler = init_scheduler(optimizer, config.optim)
        logger.info("Done building optimizer")

        # ---------------------------------------------------------------------
        # Recover Checkpoint
        # ---------------------------------------------------------------------

        state = TrainState(
            data=init_dataloader_state(config.data),
            optim=init_optimizer_state(),
        )

        monitor.report_objects(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            config=config,
        )

        # ---------------------------------------------------------------------
        # DataLoader
        # ---------------------------------------------------------------------

        dataloader = context_stack.enter_context(
            OnlineDataLoaderManager(
                config=config.data,
                state=state.data,
            )
        )

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------

        model.train()

        # poor man's profiler
        timer = monitor.profiler

        while state.optim.step < config.optim.steps:
            # accumulation step
            state.optim.acc_step += 1
            state.optim.acc_step = state.optim.acc_step % config.optim.grad_acc_steps

            # -----------------------------------------------------------------
            # Batch of data (with random state for reproducibility)
            # -----------------------------------------------------------------

            timer.start_timer()
            batch, restart_info = next(dataloader)
            batch = batch.pin_memory()
            timer.end_timer("data_cpu_time")

            timer.start_timer()
            batch = batch.to(device=cluster.device, non_blocking=True)
            X_batch = batch[:, :-1]
            y_batch = batch[:, 1:]
            timer.end_timer("data_io_time")

            # -----------------------------------------------------------------
            # Forward and backward pass
            # -----------------------------------------------------------------

            timer.start_timer()

            # forward propagation
            preds = model(X_batch)
            loss = loss_func(preds, y_batch)

            # rescale when using gradient accumulation (backprop on mean, not sum)
            loss = loss / config.optim.grad_acc_steps

            # backward propagation
            loss.backward()

            # optimizer step
            if state.optim.acc_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                state.data.report_restart_info(restart_info)
                state.optim.step += 1

            timer.end_timer("model_time", sync=True)

            # -----------------------------------------------------------------
            # Call monitor for garbage collection, checkpointing...
            # -----------------------------------------------------------------

            timer.start_timer()
            monitor()
            timer.end_timer("monitor_time")

            # -----------------------------------------------------------------
            # Log metrics
            # -----------------------------------------------------------------

            timer.start_timer()

            if trigger_update(state, config.monitor.logging.period):
                # For logging we undo that scaling
                loss = loss.detach() * config.optim.grad_acc_steps
                metrics = {
                    "loss": loss.item(),
                    "step": state.optim.step,
                    "acc_step": state.optim.acc_step,
                }
                monitor.report_metrics(metrics)

                # log to console
                if is_master_process():
                    logger.info(f"Step: {metrics['step']}, Loss: {round(metrics['loss'], 4):>7}")

            timer.end_timer("log_time")

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
        logger.warning(f"Host: {get_hostname()} - Global rank: {prod_id}")
        if prod_id == 0:
            logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
            os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
        else:
            logger.warning("Not the master process, no need to requeue.")
        sys.exit(0)

    logger.info("Training done.")


def main():
    """
    Command line interface using OmegaConf

    Read argument from a config file specified by the `config` cli argument. E.g.,
    ```bash
    python -m apps.gssm.train config=apps/gssm/configs/debug.yaml
    ```

    Non-specified arguments will be filled with the default values of the Config classes.
    """
    # Load config from path specified by the `config` cli argument
    cli_args = OmegaConf.from_cli()
    file_config = OmegaConf.load(cli_args.pop("config", None))

    # Default to default arguments for unspecified values
    default_config = OmegaConf.structured(TrainingConfig())
    config = OmegaConf.merge(default_config, file_config, cli_args)
    config = OmegaConf.to_object(config)
    config.__manual_post_init__()

    # Launch training with the config
    train(config)


if __name__ == "__main__":
    main()
