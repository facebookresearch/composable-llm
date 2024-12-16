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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .optim import OptimizerConfig
from .path import CHECKPOINT_DIR
from .train import TrainState

logger = logging.getLogger(__file__)


@dataclass
class CheckpointConfig:
    freq: int = -1
    keep_only: int = 1
    path: Optional[str] = None

    def __post_init__(self):
        if self.path is None:
            self.path = str(CHECKPOINT_DIR)
        if self.keep_only == 1:
            self.overwrite: bool = True


class CheckpointManager:
    """
    Checkpoint manager

    Attributes
    ----------
    path:
        Path to the checkpoint directory
    freq:
        Frequency at which to save checkpoints
    keep_only:
        Number of checkpoints to keep
    model:
        Model to checkpoint
    optimizer:
        Optimizer to checkpoint
    state:
        Training state to checkpoint
    dp_rank:
        Device rank
    saved:
        Whether the latest model has been saved
    """

    FOLDER_NAME = "{:010d}"
    RE_FOLDER = r"\d{10}"
    CONFIG_NAME = "params.json"
    TRAIN_STATE_NAME = "train_state_rank_{:05d}.json"
    RE_DIGITS = re.compile(r"\d+")

    def __init__(self, config: CheckpointConfig, model: torch.nn.Module, optimizer: OptimizerConfig, state: TrainState):
        self.path = config.path
        self.freq = config.freq
        self.keep_only = config.keep_only

        self.model = model
        self.optimizer = optimizer
        self.state = state
        os.makedirs(self.path, exist_ok=True)

        # device rank (WE DO NOT HANDLE PARALLELISM YET)
        self.dp_rank = 0
        self.saved = False

    def __enter__(self):
        # load checkpoint if it exists
        path = self.get_last_checkpoint_path()
        if path is None:
            self.saved = False
        else:
            self.load()
            self.saved = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # TODO: Handle exceptions
        self.save()

    def __call__(self):
        # check whether it is time to save checkpoint
        pass

    def list_checkpoints(self) -> list[Path]:
        """
        List all existing checkpoints
        """
        folders = [p for p in Path(self.path).iterdir() if p.is_dir() and re.match(self.RE_FOLDER, p.name)]
        folders.sort(key=lambda p: self._get_key_step(p.name))
        return folders

    def get_last_checkpoint_path(self):
        """
        Get last existing checkpoint
        """
        path = None
        all_checkpoints = self.list_checkpoints()
        for p in reversed(all_checkpoints):
            if (p / self.TRAIN_STATE_NAME.format(self.dp_rank)).is_file():
                path = p
                break
        return path

    def get_path(self):
        pass

    @torch.no_grad()
    def get_state_dict(self, model, optimizer):
        return {"model": model.state_dict(), "optim": optimizer.state_dict()}

    def save(self, model, optimizer, train_state, config) -> bool:
        """
        Checkpoint model, optimizer and training state
        """
        path = Path(self.path)
        curr_save_dir = path / self.FOLDER_NAME.format(train_state.step)
        curr_save_dir.mkdir(parents=False, exist_ok=True)

        logger.info(f"Saving to: {str(curr_save_dir)}")
        state_dict = self.get_state_dict(model, optimizer)
        torch.save(state_dict, curr_save_dir / "checkpoint.pth")

        with open(curr_save_dir / self.CONFIG_NAME, "w") as f:
            json.dump(config, f)

        train_state_name = self.TRAIN_STATE_NAME.format(0)
        with open(curr_save_dir / train_state_name, "w") as f:
            json.dump(train_state.state_dict(), f)

        self.existing_saves.append(curr_save_dir)
        self.clean_up()

    @torch.no_grad()
    def load(self, model, optimizer, train_state) -> bool:
        """
        Load from checkpoint

        Parameters
        ----------
        model: nn.Module
            The model to load the checkpoint into.
        optimizer: OptimizerConfig
            The optimizer to load the checkpoint into.
        train_state: TrainingConfig
            The training state to load the checkpoint into.

        Returns
        -------
        bool
            Whether a checkpoint was successfully loaded.
        """
        return False

        # logger.info("Reloading train state")
        # train_state_name = self.TRAIN_STATE_NAME.format(self.dp_rank)
        # with open(path / train_state_name, "r") as f:
        #     train_state_dict = json.load(f)
        # train_state.load_state_dict(train_state_dict)
        # logger.info("Train state reloaded")

        # logger.info(f"Loading from: {str(path)}")
        # state_dict = torch.load(path / "checkpoint.pth")
        # model.load_state_dict(state_dict["model"])
        # optimizer.load_state_dict(state_dict["optim"])
        # logger.info("Model and optimizer reloaded")

        # return True

    @classmethod
    def _get_key_step(cls, name: str):
        return int(re.findall(cls.RE_DIGITS, name)[-1])
