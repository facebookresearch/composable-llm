"""
Checkpoint manager

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import json
import re
import shutil
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path, PosixPath
from types import TracebackType

import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from ..distributed import get_rank, is_master_process
from ..utils import TrainState
from .monitor import Monitor

logger = getLogger("nanollama")


@dataclass
class CheckpointConfig:
    period: int = 0
    keep_only: int = 0
    sync_step: bool = True  # whether profiler step should be sync with optimizer step
    path: str = field(init=False, default="")

    def __check_init__(self):
        """Check validity of arguments."""
        assert self.path, "path was not set"


class Checkpointer(Monitor):
    """
    Checkpoint manager

    Attributes
    ----------
    period:
        Number of updates between each checkpoint
    keep_only:
        Number of checkpoints to keep
    path:
        Path to the checkpoint directory
    model:
        Model to checkpoint
    optimizer:
        Optimizer to checkpoint
    state:
        Training state to checkpoint
    saved:
        Whether the latest model has been saved
    """

    state_name = "train_state_{:05d}.json"
    folder_name = "{:010d}"
    re_folder = r"\d{10}"
    re_digits = re.compile(r"\d+")

    def __init__(
        self,
        config: CheckpointConfig,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler.LambdaLR,
        state: TrainState,
    ):
        super().__init__(config)

        self.keep_only = config.keep_only
        self.path = Path(config.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.sync = config.sync_step

        # Create alias for the objects to monitor.
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = state

        self.device_rank = get_rank()
        self.saved_step = 0

    @torch.no_grad()
    def __enter__(self) -> "Checkpointer":
        """
        Loading checkpoint if any
        """
        path = self.get_last_checkpoint_path(self.path)
        if not path:
            return self

        logger.info("Reloading train state")
        file_path = path / self.state_name.format(self.device_rank)
        with open(file_path) as f:
            train_state_dict = json.load(f)
        self.state.load_state_dict(train_state_dict)
        logger.info("Train state reloaded")

        logger.info(f"Loading from: {str(path)}")
        state_dict = torch.load(path / "checkpoint.pth", weights_only=True)
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optim"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        logger.info("Model, optimizer and scheduler reloaded")

        if self.sync:
            self.step = self.state.optim.step
        self.saved_step = self.step

        return self

    def update(self, eval: bool = False) -> None:
        """
        Checkpoint model, optimizer, scheduler and training state

        Parameters
        ----------
        eval:
            Whether to save the checkpoint for evaluation
        """
        # do not checkpoint twice
        if self.saved_step == self.step:
            return

        save_dir = self.path / self.folder_name.format(self.state.optim.step)
        save_dir.mkdir(parents=False, exist_ok=True)

        # add evaluation flag, if needed
        if eval:
            eval_flag = save_dir / "eval"
            eval_flag.touch()

        logger.info(f"Saving checkpoint at step {self.state.optim.step} to {str(save_dir)}")

        filename = self.state_name.format(self.device_rank)
        with open(save_dir / filename, "w") as f:
            json.dump(self.state.state_dict(), f, indent=2)

        if is_master_process():
            logger.info("Saving model, optimizer and scheduler")
            state_dict = {
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }
            torch.save(state_dict, save_dir / "checkpoint.pth")
            self._cleaning()

        self.saved_step = self.step

    @classmethod
    def get_last_checkpoint_path(cls, path: str) -> str:
        """
        Get last existing checkpoint
        """
        folders = cls._list_checkpoints(path)
        if folders:
            return max(folders, key=lambda p: cls._get_key_step(p.name))
        return ""

    def _cleaning(self) -> None:
        """
        Clean up old checkpoints
        """
        if self.keep_only <= 0:
            return
        all_checkpoints = self._list_checkpoints(self.path)
        all_checkpoints.sort(key=lambda p: self._get_key_step(p.name))
        for prefix in all_checkpoints[: -self.keep_only]:
            if not (prefix / "eval").exists():
                logger.info(f"Removing: {str(prefix)}")
                shutil.rmtree(prefix)

    @classmethod
    def _list_checkpoints(cls, path: str) -> list[PosixPath]:
        """
        List all existing checkpoints
        """
        return [p for p in path.iterdir() if p.is_dir() and re.match(cls.re_folder, p.name)]

    @classmethod
    def _get_key_step(cls, name: str) -> int:
        return int(re.findall(cls.re_digits, name)[-1])

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """
        Exit checkpoint context by saving checkpoint if needed
        """
        self.update()


# ------------------------------------------------------------------------------
# Evaluation logic
# ------------------------------------------------------------------------------


class EvalCheckpointer:
    def __init__(self, model: nn.Module, path: str, train_step: int = None) -> None:
        self.model = model
        if train_step is None:
            path = Path(path)
            self.save_dir = Path(Checkpointer.get_last_checkpoint_path(path))
        else:
            self.save_dir = Path(path) / Checkpointer.folder_name.format(train_step)

    def __enter__(self):
        logger.info(f"Loading model from: {str(self.save_dir)}")
        state_dict = torch.load(self.save_dir / "checkpoint.pth", weights_only=True)
        self.model.load_state_dict(state_dict["model"])

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """
        Exit checkpoint context by remove `eval` flag

        See Also
        --------
        Checkpointer.update(eval=True)
        """
        eval_flag = self.save_dir / "eval"
        if eval_flag.exists():
            eval_flag.unlink()
