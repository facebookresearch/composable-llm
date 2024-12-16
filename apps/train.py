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
from torch.distributed.checkpoint.stateful import Stateful

from composition.data import (
    DataConfig,
    DataLoaderState,
    data_loader_context,
    init_dataloader_state,
)
from composition.distributed import ComputeConfig
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

    def __post_init__(self):
        # TO BE REMOVED temporary patch
        self.model.vocab_size = 4
        self.model.seq_len = 32
        self.model.emb_dim = 64
        self.model.nb_heads = 2
        self.model.ffn_dim = 256
        self.model.nb_layers = 2

        self.data.seq_len = self.model.seq_len
        self.data.batch_size = 32


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
        # Build and Parallelize model
        # ---------------------------------------------------------------------

        logger.info("Building model")
        model = Transformer(config.model)
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
        # Build Data and Monitoring Context
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
            )

            y_batch = torch.tensor(
                batch[:, 1:],
                dtype=torch.long,
            )

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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    train(TrainingConfig())
