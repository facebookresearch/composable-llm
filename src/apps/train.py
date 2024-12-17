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
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ..composition.checkpoint import CheckpointConfig, CheckpointManager
from ..composition.computing import ComputeConfig
from ..composition.data import DataConfig, dataloader_manager, init_dataloader_state
from ..composition.model import Transformer, TransformerConfig
from ..composition.monitor import MonitorConfig, MonitorsManager
from ..composition.optim import (
    OptimizerConfig,
    init_optimizer,
    init_optimizer_state,
    init_scheduler,
)
from ..composition.train import TrainState

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
    compute: ComputeConfig = field(default_factory=ComputeConfig)
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
# Training State and Preemption Handling
# -------------------------------------------------------------------------------


def loss_func(preds, targets):
    vocab_size = preds.size(-1)
    return F.cross_entropy(preds.view(-1, vocab_size), targets.view(-1))


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train(config: TrainingConfig):

    with ExitStack() as context_stack:

        # ---------------------------------------------------------------------
        # Monitor
        # ---------------------------------------------------------------------

        monitor = context_stack.enter_context(MonitorsManager(config.monitor))

        # ---------------------------------------------------------------------
        # Build and Parallelize model
        # ---------------------------------------------------------------------

        logger.info("Building model")
        model = Transformer(config.model)
        model.to(device=config.compute.device)
        logger.info("Done building model")

        monitor.report_model(model)

        # Build Optimizer
        logger.info("Building optimizer")
        optimizer = init_optimizer(model, config.optim)
        scheduler = init_scheduler(optimizer, config.optim)
        logger.info("Done building optimizer")

        # ---------------------------------------------------------------------
        # Recover Checkpoint
        # ---------------------------------------------------------------------

        state = TrainState(
            data=init_dataloader_state(config.data.seed),
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

        # ---------------------------------------------------------------------
        # DataLoader
        # ---------------------------------------------------------------------

        dataloader = context_stack.enter_context(
            dataloader_manager(
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

            batch, state.data = next(dataloader)
            X_batch = torch.tensor(
                batch[:, :-1],
                dtype=torch.long,
            ).to(device=config.compute.device)

            y_batch = torch.tensor(
                batch[:, 1:],
                dtype=torch.long,
            ).to(device=config.compute.device)

            # -----------------------------------------------------------------
            # Forward and backward pass
            # -----------------------------------------------------------------

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

            logger.info(f"Step: {state.optim.step}, Loss: {loss.item()}")

            # -----------------------------------------------------------------
            # Manager calls
            # -----------------------------------------------------------------

            # checkpointing
            checkpointer()


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
