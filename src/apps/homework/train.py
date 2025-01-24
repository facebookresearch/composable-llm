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
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ...nanollama.distributed import ClusterConfig, ClusterManager
from ...nanollama.model import Transformer, TransformerConfig
from ...nanollama.monitor import (
    OrchestratorConfig,
    PreemptionHandler,
    Profiler,
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
class DataConfig:
    batch_size: int = 2048
    seq_len: int = 128


@dataclass
class TrainingConfig:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

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
        # logger: Logger = context_stack.enter_context(Logger(config.orchestration.logging))
        # utils: UtilityManager = context_stack.enter_context(UtilityManager(config.orchestration.utils))
        # wandb: WandbLogger = context_stack.enter_context(WandbLogger(config.orchestration.wandb, run_config=config))

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
            data=(config.data),
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

        # dataloader = FileDataLoader = context_stack.enter_context(
        #     FileDataLoader(
        #         config=config.data,
        #         state=state.data,
        #     )
        # )

        # ---------------------------------------------------------------------
        # Global information
        # ---------------------------------------------------------------------

        profiler: Profiler = context_stack.enter_context(Profiler(config.orchestration.profiler, state=state))

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------

        model.train()

        # aliases
        # eval_period = config.evaluation.period

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

            # profiler.start_timer()
            batch = torch.ones((config.data.batch_size, config.data.seq_len), dtype=torch.long)
            restart_info = None

            # batch, restart_info = next(dataloader)
            if cluster.device.type != "cpu":
                batch = batch.pin_memory()

            batch = batch.to(device=cluster.device, non_blocking=True)
            X_batch = batch[:, :-1]
            y_batch = batch[:, 1:]

            # -----------------------------------------------------------------
            # Forward and backward pass
            # -----------------------------------------------------------------

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

            profiler()

            print("step", preds.std().item())

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
    run_config["slurm"] = launcher.pop("slurm", {})
    run_config.pop("slurm")

    # # grid id system to handle multiple datasets
    # grid_id = run_config.get("grid_id", 0)
    # run_config["data"]["path"] = run_config["data"]["path"].replace("$GRIDID", str(grid_id))
    # run_config["evaluation"]["data"]["path"] = run_config["evaluation"]["data"]["path"].replace("$GRIDID", str(grid_id))

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
