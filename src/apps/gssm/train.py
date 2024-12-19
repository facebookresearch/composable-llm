"""
Vanilla Transformer Training Script.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

# -------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------

import logging
import os
import signal
import socket
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from ...composition.checkpoint import CheckpointConfig, CheckpointManager
from ...composition.cluster import ClusterConfig
from ...composition.data.gssm import (
    DataConfig,
    DataLoaderManager,
    init_dataloader_state,
)
from ...composition.model.transfomer import Transformer, TransformerConfig
from ...composition.monitor import MonitorConfig, MonitorsManager
from ...composition.optim import (
    OptimizerConfig,
    init_optimizer,
    init_optimizer_state,
    init_scheduler,
)
from ...composition.train import TrainState
from ...composition.utils import trigger_update

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------
# Configuration Class
# -------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: TransformerConfig = field(default_factory=TransformerConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
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

        # vocabulary size

        # checkpoint directory
        if self.checkpoint.path == "":
            dir = self.monitor.dir
            self.checkpoint.path = str(Path(dir) / "checkpoints")


# -------------------------------------------------------------------------------
# Preemption Handling
# -------------------------------------------------------------------------------


def loss_func(preds, targets):
    vocab_size = preds.size(-1)
    return F.cross_entropy(preds.view(-1, vocab_size), targets.view(-1))


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


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
    logger.info("Signal installed.")

    with ExitStack() as context_stack:

        # ---------------------------------------------------------------------
        # Monitor: profiling, probing, logging
        # ---------------------------------------------------------------------

        monitor = context_stack.enter_context(MonitorsManager(config.monitor))

        # ---------------------------------------------------------------------
        # Build and Parallelize model
        # ---------------------------------------------------------------------

        backend = "nccl"
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        logger.info(f"Running on ddp rank: {ddp_rank} / {world_size}")

        logger.info("Building model")
        model = Transformer(config.model)
        model.to(device=config.cluster.device)
        logger.info("Done building model")

        if world_size > 1:
            model = DDP(model, device_ids=[ddp_local_rank])

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
            data=init_dataloader_state(config.data, rank=ddp_rank),
            optim=init_optimizer_state(),
        )

        checkpointer = context_stack.enter_context(
            CheckpointManager(
                config=config.checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
            )
        )

        monitor.report_objects(model=model, optimizer=optimizer, scheduler=scheduler, state=state)

        # ---------------------------------------------------------------------
        # DataLoader
        # ---------------------------------------------------------------------

        dataloader = context_stack.enter_context(
            DataLoaderManager(
                config=config.data,
                state=state.data,
            )
        )

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------

        model.train()

        while state.optim.step < config.optim.steps:
            # accumulation step
            state.optim.acc_step += 1
            state.optim.acc_step = state.optim.acc_step % config.optim.grad_acc_steps

            # -----------------------------------------------------------------
            # Batch of data
            # -----------------------------------------------------------------

            dataloader_time = timer()
            batch = next(dataloader)
            X_batch = torch.tensor(
                batch[:, :-1],
                dtype=torch.long,
            ).to(device=config.cluster.device)

            y_batch = torch.tensor(
                batch[:, 1:],
                dtype=torch.long,
            ).to(device=config.cluster.device)
            dataloader_time = round(timer() - dataloader_time, 4)

            # -----------------------------------------------------------------
            # Forward and backward pass
            # -----------------------------------------------------------------

            model_time = torch.cuda.Event(enable_timing=True)
            model_endtime = torch.cuda.Event(enable_timing=True)
            model_time.record()

            # forward propagation
            preds = model(X_batch)
            loss = loss_func(preds, y_batch)

            # rescale when using gradient accumulation
            loss = loss / config.optim.grad_acc_steps

            # backward propagation
            loss.backward()

            # optimizer step
            if state.optim.acc_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                state.optim.step += 1

            model_endtime.record()
            torch.cuda.synchronize()
            model_time = round(model_time.elapsed_time(model_endtime) * 1e-3, 4)

            # -----------------------------------------------------------------
            # Call managers for garbage collection, checkpointing...
            # -----------------------------------------------------------------

            checkpointer()
            monitor()

            # -----------------------------------------------------------------
            # Log metrics
            # -----------------------------------------------------------------

            if trigger_update(state, config.monitor.log_period):
                # For logging we undo that scaling
                loss = loss.detach() * config.optim.grad_acc_steps
                metrics = {
                    "loss": loss.item(),
                    "step": state.optim.step,
                    "acc_step": state.optim.acc_step,
                    "data_time": dataloader_time,
                    "model_time": model_time,
                }
                monitor.report_metrics(metrics)

                # log to console
                logger.info(f"Step: {metrics['step']}, Loss: {round(metrics['loss'], 4):>7}")

            # -----------------------------------------------------------------
            # Evaluation
            # -----------------------------------------------------------------

            # -----------------------------------------------------------------
            # Handle preemption
            # -----------------------------------------------------------------

            if preemption_flag["flag"]:
                break

    if preemption_flag["flag"]:
        prod_id = int(os.environ["SLURM_PROCID"])
        logger.warning(f"Host: {socket.gethostname()} - Global rank: {prod_id}")
        if prod_id == 0:
            logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
            os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
        else:
            logger.warning("Not the master process, no need to requeue.")
        sys.exit(0)

    logger.info("Training finished")

    if world_size > 1:
        destroy_process_group()


def main():
    """
    Command line interface using OmegaConf

    Read argument from a config file specified by the `config` cli argument. E.g.,
    ```bash
    python -m apps.train config=apps/debug.yaml
    ```

    Non-specified arguments will be filled with the default values of the Config classes.
    """
    # Load config from path specified by the `config` cli argument
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    # Default to default arguments for unspecified values
    default_cfg = OmegaConf.structured(TrainingConfig())
    config = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    config = OmegaConf.to_object(config)
    config.__manual_post_init__()

    # Launch training with the config
    train(config)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    main()
