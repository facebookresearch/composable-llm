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
    https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments

    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        mode: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    # Load config from path specified by the `config` cli argument
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # remove 'config' attribute as the underlying config class does not have it
    del cli_args.config

    # Load structured config
    default_cfg = OmegaConf.structured(TrainingConfig())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    # Launch training with the config
    train(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    main()
