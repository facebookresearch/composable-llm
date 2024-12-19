"""
Checkpoint manager

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from .train import TrainState
from .utils import trigger_update

logger = logging.getLogger(__file__)


@dataclass
class CheckpointConfig:
    period: int = -1
    keep_only: int = -1
    path: str = ""


class CheckpointManager:
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
        self.period = config.period
        self.keep_only = config.keep_only
        self.path = Path(config.path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = state

        self.device_rank = int(os.environ["RANK"])
        self.up_to_date = True

    def __enter__(self):
        """
        Enter checkpoint context by loading checkpoint if it exists
        """
        path = self.get_last_checkpoint_path()
        if path:
            self.load(path)
        return self

    def __call__(self):
        """
        Save checkpoint if it matching the period
        """
        if trigger_update(self.state, self.period):
            self.save()
            self.up_to_date = True
        else:
            self.up_to_date = False

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit checkpoint context by saving checkpoint if needed
        """
        if not self.up_to_date:
            self.save()

    def save(self) -> bool:
        """
        Checkpoint model, optimizer, scheduler and training state
        """
        save_dir = self.path / self.folder_name.format(self.state.optim.step)
        save_dir.mkdir(parents=False, exist_ok=True)
        logger.info(f"Saving checkpoint at step {self.state.optim.step} to {str(save_dir)}")

        state_dict = {
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state_dict, save_dir / "checkpoint.pth")

        filename = self.state_name.format(self.device_rank)
        with open(save_dir / filename, "w") as f:
            json.dump(self.state.state_dict(), f)

        self.cleaning()

    @torch.no_grad()
    def load(self, path: Path):
        """
        Load from checkpoint

        Parameters
        ----------
        path:
            Path to the checkpoint directory
        """

        logger.info("Reloading train state")
        file_path = path / self.state_name.format(self.device_rank)
        with open(file_path, "r") as f:
            train_state_dict = json.load(f)
        self.state.load_state_dict(train_state_dict)
        logger.info("Train state reloaded")

        logger.info(f"Loading from: {str(path)}")
        state_dict = torch.load(path / "checkpoint.pth", weights_only=True)
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optim"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        logger.info("Model, optimizer and scheduler reloaded")

    def get_last_checkpoint_path(self):
        """
        Get last existing checkpoint
        """
        path = None
        filename = self.state_name.format(self.device_rank)
        all_checkpoints = self.list_checkpoints()
        for dir_name in reversed(all_checkpoints):
            if (dir_name / filename).is_file():
                path = dir_name
                break
        return path

    def list_checkpoints(self) -> list[Path]:
        """
        List all existing checkpoints
        """
        folders = [p for p in self.path.iterdir() if p.is_dir() and re.match(self.re_folder, p.name)]
        folders.sort(key=lambda p: self._get_key_step(p.name))
        return folders

    @classmethod
    def _get_key_step(cls, name: str):
        return int(re.findall(cls.re_digits, name)[-1])

    def cleaning(self):
        """
        Clean up old checkpoints
        """
        if self.keep_only == -1:
            return
        all_checkpoints = self.list_checkpoints()
        for prefix in all_checkpoints[: -self.keep_only]:
            logger.info(f"Removing: {str(prefix)}")
            shutil.rmtree(prefix)
