"""
Evaluation Script

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import json
from contextlib import ExitStack
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from types import TracebackType

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nanollama.data.hdf5 import DataConfig, FileEvaluator
from ...nanollama.distributed import ClusterConfig, ClusterManager, get_local_rank, get_rank, is_master_process
from ...nanollama.model import Transformer, TransformerConfig
from ...nanollama.monitor import Logger, OrchestratorConfig, PreemptionHandler

logger = getLogger("nanollama")

# ------------------------------------------------------------------------------
# Online Evaluation
# ------------------------------------------------------------------------------


@dataclass
class EvaluationConfig:
    period: int = 0
    asynchronous: bool = False
    path: str = field(init=False, default="")
    data: DataConfig = field(default_factory=DataConfig)

    def __check_init__(self) -> None:
        """Check validity of arguments"""
        assert self.path, "path was not set"

        # manual post initialization of all modules
        for module in self.__dict__.values():
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

    def __init__(self, config: EvaluationConfig, model: nn.Module, step: int) -> None:
        self.model = model
        self.train_step = step
        self.data_config = config.data

        self.path = Path(config.path) / f"eval_{get_rank()}.jsonl"
        self.tmp_file = Path(config.path) / f".{get_rank()}_{step}.tmp"

        self.step = 0
        self.loss = 0
        self.scaling = 0

    def __enter__(self) -> "EvalComputer":
        logger.info(f"Entering evaluation at step {self.train_step}")
        self.model.eval()

        # retrieve previous computations
        if self.tmp_file.exists():
            with open(self.tmp_file) as f:
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
            with open(self.path, "a") as f:
                print(json.dumps({"loss": self.loss, "step": self.train_step}), file=f, flush=True)

            # remove temporary file
            self.tmp_file.unlink()
            return False

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        # if the evaluation was interrupted, save the current state
        if self.tmp_file.exists():
            with open(self.tmp_file, "w") as f:
                print(json.dumps({"loss": self.loss, "scaling": self.scaling, "step": self.step}), file=f, flush=True)

        self.model.train()
        self.loader.__exit__(exc, value, tb)


@torch.no_grad()
def run_evaluation(config: EvaluationConfig, model: nn.Module, step: int) -> None:
    with EvalComputer(config, model, step) as computer:
        while next(computer):
            logger.debug(f"Evaluation. step: {computer.step} - loss: {round(computer.loss,4):>7}")

    if is_master_process():
        logger.info(f"Test loss: {round(computer.loss, 4):>7}")


# ------------------------------------------------------------------------------
# Evaluation Run
# ------------------------------------------------------------------------------


@dataclass
class EvaluationRunConfig:
    train_step: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: TransformerConfig = field(default_factory=TransformerConfig)

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: OrchestratorConfig = field(default_factory=OrchestratorConfig)


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

        # TODO: load model from checkpoint

        # ---------------------------------------------------------------------
        # Run evaluation into chunks
        # ---------------------------------------------------------------------

        computer = context_stack.enter_context(EvalComputer(config.data, model, config.train_step))

        while next(computer):
            # -----------------------------------------------------------------
            # Handle preemption
            # -----------------------------------------------------------------

            if preemption():
                logger.warning("Preemption flag set")
                break

    logger.info("Evaluation done.")

    # TODO: add wandb logging?


def main() -> None:
    # parse config
    # launch eval
    pass
