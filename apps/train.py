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

import torch
from omegaconf import OmegaConf
from torch.distributed.checkpoint.stateful import Stateful

from composition.computing import ComputeConfig
from composition.data import (
    DataConfig,
    DataLoaderState,
    data_loader_context,
    init_dataloader_state,
)
from composition.model import Transformer, TransformerConfig
from composition.monitor import MonitorConfig
from composition.optim import OptimizerConfig, build_optimizer

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------
# Configuration Class
# -------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: TransformerConfig = field(default_factory=TransformerConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)

    def __manual_post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # Sequence length
        if self.model.seq_len == -1 and self.data.seq_len == -1:
            raise ValueError("seq_len must be provided in either model or data")
        if self.model.seq_len == -1:
            self.model.seq_len = self.data.seq_len
        if self.data.seq_len == -1:
            self.data.seq_len = self.model.seq_len


# -------------------------------------------------------------------------------
# Training State and Preemption Handling
# -------------------------------------------------------------------------------


@dataclass
class TrainState(Stateful):
    step: int  # nb of steps taken by the optimizer
    acc_step: int  # nb of accumulation steps done since last optimizer step
    scheduler: None
    data_loader_state: DataLoaderState


def loss_func(preds, targets):
    return torch.nn.functional.cross_entropy(preds.view(-1, 4), targets.view(-1))


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train(config: TrainingConfig):

    with ExitStack() as context_stack:

        # ---------------------------------------------------------------------
        # Monitor
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # Build and Parallelize model
        # ---------------------------------------------------------------------

        logger.info("Building model")
        model = Transformer(config.model)
        model.to(device="cuda")
        logger.info("Done building model")

        # Build Optimizer
        logger.info("Building optimizer")
        optimizer, scheduler = build_optimizer(model, config.optim)
        logger.info("Done building optimizer")

        # ---------------------------------------------------------------------
        # Recover Checkpoint
        # ---------------------------------------------------------------------

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=None,
            scheduler=None,
        )
        train_state.data_loader_state = init_dataloader_state(config.data.seed)

        # ---------------------------------------------------------------------
        # DataLoader
        # ---------------------------------------------------------------------

        data_loader = context_stack.enter_context(
            data_loader_context(
                config.data,
                state=train_state.data_loader_state,
            )
        )

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------

        model.train()

        while train_state.step < config.optim.steps:
            # accumulation step
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % config.optim.grad_acc_steps

            # -----------------------------------------------------------------
            # Batch of data
            # -----------------------------------------------------------------

            batch, train_state.data_loader_state = next(data_loader)
            X_batch = torch.tensor(
                batch[:, :-1],
                dtype=torch.long,
            ).cuda()

            y_batch = torch.tensor(
                batch[:, 1:],
                dtype=torch.long,
            ).cuda()

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
            if train_state.acc_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            print(f"Step: {train_state.step}, Loss: {loss.item()}")


def main():
    """
    The command line interface here uses OmegaConf

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

    # Load structured config
    default_cfg = OmegaConf.structured(TrainingConfig())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    cfg.__manual_post_init__()

    # Launch training with the config
    train(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    main()
