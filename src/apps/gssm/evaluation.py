"""
Evaluation Script

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import json
import logging
import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from types import TracebackType

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ...nanollama.data.hdf5 import DataConfig, FileEvaluator
from ...nanollama.distributed import ClusterConfig, ClusterManager, get_local_rank, get_rank, is_master_process
from ...nanollama.launcher import SlurmConfig
from ...nanollama.model import Transformer, TransformerConfig
from ...nanollama.monitor import (
    EvalCheckpointer,
    EvalOrchestratorConfig,
    Logger,
    PreemptionHandler,
    WandbLogger,
)
from ...nanollama.utils import initialize_nested_object

logger = getLogger("nanollama")

# ------------------------------------------------------------------------------
# Online Evaluation
# ------------------------------------------------------------------------------


@dataclass
class EvaluationConfig:
    # evaluation period in training run
    period: int = 0

    path: str = field(init=False, default="")
    data: DataConfig = field(default_factory=DataConfig)

    # for asynchronous evaluation
    asynchronous: bool = False
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: EvalOrchestratorConfig = field(default_factory=EvalOrchestratorConfig)

    def __check_init__(self) -> None:
        """Check validity of arguments"""
        assert self.path, "path was not set"

        # manual post initialization of all modules
        if self.asynchronous:
            modules = self.__dict__.values()
        else:
            modules = [self.data]
        for module in modules:
            if hasattr(module, "__check_init__"):
                module.__check_init__()


def loss_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = preds.size(-1)
    return F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1))


class EvalComputer:
    """
    Evaluation manager

    Running evaluation into chunks in order to handle job preemption.

    Usage:
    ```python
    with EvalComputer(*args) as computer:
        while next(computer):
            pass
    """

    def __init__(self, config: EvaluationConfig, model: nn.Module, train_step: int) -> None:
        self.train_step = train_step
        self.model = model
        self.data_config = config.data

        self.path = Path(config.path) / f"eval_{get_rank()}.jsonl"
        self.tmp_file = Path(config.path) / f".{get_rank()}_{train_step}.tmp"

        self.step = 0
        self.loss = 0
        self.scaling = 0

    def __enter__(self) -> "EvalComputer":
        logger.info("Evaluating model.")
        self.model.eval()

        # retrieve previous computations
        if self.tmp_file.exists():
            with open(os.path.expandvars(self.tmp_file)) as f:
                data = json.loads(f)
                self.loss: float = data["loss"]
                self.scaling: float = data["scaling"]
                self.step: int = data["step"]
            logger.info(f"Found previous evaluation at step {self.step}")
        else:
            self.tmp_file.touch()

        # skip batches that were already evaluated
        self.loader = FileEvaluator(self.data_config).__enter__()
        for _ in range(self.step):
            next(self.loader)

        return self

    @torch.no_grad()
    def __next__(self) -> None:
        try:
            batch, _ = next(self.loader)
            batch = batch.to(get_local_rank())
            X_batch = batch[:, :-1]
            y_batch = batch[:, 1:]

            # evaluate
            preds = self.model(X_batch)
            scaling = batch.size(0) / self.data_config.batch_size
            self.loss += scaling * loss_func(preds, y_batch).item()
            self.scaling += scaling
            self.step += 1

            return True

        except StopIteration:
            # rescale loss and save it
            self.loss /= self.scaling
            with open(os.path.expandvars(self.path), "a") as f:
                print(json.dumps({"loss": self.loss, "step": self.train_step}), file=f, flush=True)

            # remove temporary file
            self.tmp_file.unlink()
            return False

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        # if the evaluation was interrupted, save the current state
        if self.tmp_file.exists():
            with open(os.path.expandvars(self.tmp_file), "w") as f:
                print(json.dumps({"loss": self.loss, "scaling": self.scaling, "step": self.step}), file=f, flush=True)

        self.model.train()
        self.loader.__exit__(exc, value, tb)


@torch.no_grad()
def run_evaluation(config: EvaluationConfig, model: nn.Module, step: int) -> None:
    with EvalComputer(config, model, step) as computer:
        while next(computer):
            logger.debug(f"Evaluation. step: {computer.step} - loss: {round(computer.loss, 4):>7}")

    if is_master_process():
        logger.info(f"Test loss: {round(computer.loss, 4):>7}")


# ------------------------------------------------------------------------------
# Evaluation Run
# ------------------------------------------------------------------------------


@dataclass
class EvaluationRunConfig:
    path: str = field(init=False, default="")

    data: DataConfig = field(default_factory=DataConfig)
    model: TransformerConfig = field(default_factory=TransformerConfig)

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: EvalOrchestratorConfig = field(default_factory=EvalOrchestratorConfig)

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # path to stored results
        self.path = self.orchestration.logging.metric_path

        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()


@torch.no_grad()
def eval(config: EvaluationRunConfig) -> None:
    with ExitStack() as context_stack:
        # ---------------------------------------------------------------------
        # Handle preemption
        # ---------------------------------------------------------------------

        preemption: PreemptionHandler = context_stack.enter_context(PreemptionHandler())

        # ---------------------------------------------------------------------
        # Computing Environment
        # ---------------------------------------------------------------------

        cluster: ClusterManager = context_stack.enter_context(ClusterManager(config.cluster))

        # ---------------------------------------------------------------------
        # Instanciate logging
        # ---------------------------------------------------------------------

        context_stack.enter_context(Logger(config.orchestration.logging))

        # ---------------------------------------------------------------------
        # Build and Parallelize model
        # ---------------------------------------------------------------------

        logger.info("Building model")
        model = Transformer(config.model)
        model = cluster.initialize_model(model)

        # ---------------------------------------------------------------------
        # Recover Checkpoint
        # ---------------------------------------------------------------------

        # alias
        train_step = config.orchestration.train_step
        context_stack.enter_context(EvalCheckpointer(model, config.orchestration.checkpoint_path, train_step))

        # ---------------------------------------------------------------------
        # Run evaluation into chunks
        # ---------------------------------------------------------------------

        computer: EvalComputer = context_stack.enter_context(EvalComputer(config, model, train_step))

        while next(computer):
            # -----------------------------------------------------------------
            # Handle preemption
            # -----------------------------------------------------------------

            if preemption():
                logger.warning("Preemption flag set")
                break

            logger.debug(f"Evaluation. step: {computer.step} - loss: {round(computer.loss, 4):>7}")

        # wandb logging
        wandb: WandbLogger = context_stack.enter_context(WandbLogger(config.orchestration.wandb, config))
        wandb({"test_loss": computer.loss, "step": train_step})

    if is_master_process():
        logger.info(f"Test loss: {round(computer.loss, 4):>7}")

    logger.info("Evaluation done.")


def main() -> None:
    """
    Launch a evaluation job from configuration file specified by cli argument.

    Usage:
    ```
    python -m apps.my_app.eval apps/my_app/configs/my_config.yaml
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

    # casting logging directory to run_config
    if "orchestration" not in run_config:
        run_config["orchestration"] = {}
    if "launcher" in file_config:
        for key in ["name", "log_dir"]:
            if key in file_config["launcher"] and key not in run_config["orchestration"]:
                run_config["orchestration"][key] = file_config["launcher"][key]

    # initialize configuration
    config = initialize_nested_object(EvaluationRunConfig, run_config)

    # launch job
    eval(config)


if __name__ == "__main__":
    main()
