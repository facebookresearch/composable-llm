"""
Checkpoint manager

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PosixPath
from types import TracebackType

import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from ..cluster import get_rank, is_master_process
from ..train import TrainState
from ..utils import trigger_update

logger = logging.getLogger(__file__)


@dataclass
class CheckpointConfig:
    period: int = -1
    keep_only: int = -1
    path: str = ""


class Checkpointer:
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
    ):
        self.period = config.period
        self.keep_only = config.keep_only
        self.path = Path(config.path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.state = None

        self.device_rank = get_rank()
        self.up_to_date = True

    def __enter__(self):
        """
        Enter checkpoint context by loading checkpoint if it exists
        """
        return self

    def report_objects(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler.LambdaLR,
        state: TrainState,
    ) -> None:
        """
        Report object and load checkpoint if it exists
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = state

        path = self.get_last_checkpoint_path()
        if path:
            self.load(path)

    def __call__(self) -> None:
        """
        Save checkpoint if it matching the period
        """
        if trigger_update(self.state, self.period):
            self.save()
            self.up_to_date = True
        else:
            self.up_to_date = False

    def __exit__(
        self,
        exc: type[BaseException],
        value: BaseException,
        tb: TracebackType,
    ):
        """
        Exit checkpoint context by saving checkpoint if needed
        """
        if not self.up_to_date:
            self.save()

    def save(self) -> None:
        """
        Checkpoint model, optimizer, scheduler and training state
        """
        save_dir = self.path / self.folder_name.format(self.state.optim.step)
        save_dir.mkdir(parents=False, exist_ok=True)
        logger.info(f"Saving checkpoint at step {self.state.optim.step} to {str(save_dir)}")

        filename = self.state_name.format(self.device_rank)
        with open(save_dir / filename, "w") as f:
            json.dump(self.state.state_dict(), f, indent=2)

        if is_master_process():
            logging.info("Saving model, optimizer and scheduler")
            state_dict = {
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }
            torch.save(state_dict, save_dir / "checkpoint.pth")
            self.cleaning()

    @torch.no_grad()
    def load(self, path: Path) -> None:
        """
        Load from checkpoint

        Parameters
        ----------
        path:
            Path to the checkpoint directory
        """

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

    def get_last_checkpoint_path(self) -> str:
        """
        Get last existing checkpoint
        """
        folders = self.list_checkpoints()
        if folders:
            return max(folders, key=lambda p: self._get_key_step(p.name))
        return ""

    def cleaning(self) -> None:
        """
        Clean up old checkpoints
        """
        if self.keep_only == -1:
            return
        all_checkpoints = self.list_checkpoints()
        all_checkpoints.sort(key=lambda p: self._get_key_step(p.name))
        for prefix in all_checkpoints[: -self.keep_only]:
            logger.info(f"Removing: {str(prefix)}")
            shutil.rmtree(prefix)

    def list_checkpoints(self) -> list[PosixPath]:
        """
        List all existing checkpoints
        """
        return [p for p in self.path.iterdir() if p.is_dir() and re.match(self.re_folder, p.name)]

    @classmethod
    def _get_key_step(cls, name: str) -> int:
        return int(re.findall(cls.re_digits, name)[-1])
