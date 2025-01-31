"""
Dataloader from hdf5 file

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import os
from collections.abc import Generator
from dataclasses import dataclass, field
from logging import getLogger
from types import TracebackType
from typing import Any

import h5py
import numpy as np
from numpy.random import default_rng

from ..distributed import get_rank, get_world_size
from .loader import DataLoader

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# DataLoader for training
# ------------------------------------------------------------------------------


@dataclass
class DataConfig:
    path: str = ""
    n_data: int = 0
    batch_size: int = 0
    seed: int = 0
    asynchronous: bool = True  # asynchronous data loading
    buffer_size: int = 4  # number of batches to bufferize asynchronously for data loading

    def __post_init__(self):
        assert self.n_data, "Number of data must be specified."
        assert self.batch_size, "Batch size must be specified."
        self.path = os.path.expandvars(self.path)


@dataclass
class DataLoaderState:
    rng_state: dict[str, Any]
    epoch: int = 0
    step: int = 0  # batch step
    residual_idx: list[int] = field(default_factory=list)  # residual data from the previous epoch

    def __post_init__(self):
        if not isinstance(self.residual_idx, np.ndarray):
            self.residual_idx = np.array(self.residual_idx, dtype=int)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng_state,
            "epoch": self.epoch,
            "step": self.step,
            "residual_idx": self.residual_idx.tolist(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rng_state = state_dict["rng_state"]
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.residual_idx = np.array(state_dict["residual_idx"], dtype=int)

    def report_restart_info(
        self, rng_state: dict[str, Any], epoch: int, step: int, residual_idx: np.ndarray[int]
    ) -> None:
        self.rng_state = rng_state
        self.epoch = epoch
        self.step = step
        self.residual_idx = residual_idx


def init_dataloader_state(config: DataConfig) -> DataLoaderState:
    """
    Initialize the state of random number generators.
    """
    # recover state from seed
    rng_state = default_rng(config.seed).bit_generator.state
    return DataLoaderState(rng_state=rng_state)


class FileDataLoader(DataLoader):
    """
    Context manager for the data loader from file.

    Parameters
    ----------
    config:
        The configuration of the data loader.
    state:
        The state of the data loader.
    """

    def __init__(self, config: DataConfig, state: DataLoaderState):
        super().__init__(config)

        # data loader configuration
        self.n_data = config.n_data
        self.batch_size = config.batch_size
        self.path = config.path

        # track randomness
        self.epoch = state.epoch
        self.step = state.step
        self.residual_idx = state.residual_idx
        self.rng_state = state.rng_state
        logger.debug(f"RNG: {state}")

        # get the start and end indices of the data to be processed by the current device.
        world_size = get_world_size()
        n_local_data = self.n_data // world_size
        residual = self.n_data % world_size
        rank = get_rank()
        if rank < residual:
            self.start = (n_local_data + 1) * rank
            self.end = self.start + (n_local_data + 1)
        else:
            self.start = (n_local_data + 1) * residual + n_local_data * (rank - residual)
            self.end = self.start + n_local_data
        logger.debug(f"Data range: {self.start} - {self.end} ({rank + 1}/{world_size})")

    def batch_iterator(self) -> Generator[np.ndarray, None, None]:
        """
        Generate batches of sentences.
        """
        # ensure consistency of randomness over restart
        rng = default_rng()
        rng.bit_generator.state = self.rng_state

        # iterate over epochs
        while True:
            # restart information
            self.rng_state = rng.bit_generator.state

            # schedule data processing order
            epoch_idx = rng.permutation(self.n_data)

            # restrict data to the device's range
            device_idx = epoch_idx[self.start : self.end]

            # add residual data from the previous epoch
            device_idx = np.append(self.residual_idx, device_idx)

            # iterate over batches
            while True:
                begin, end = self.step * self.batch_size, (self.step + 1) * self.batch_size
                batch_idx = device_idx[begin:end]

                # do not process last batch alone if it is smaller than batch_size
                if batch_idx.shape[0] < self.batch_size:
                    self.step = 0
                    self.epoch += 1
                    self.residual_idx = batch_idx
                    break

                # sort batch_idx to optimize I/O
                batch_idx.sort()

                # remove potential duplicate when inserting residual batch
                if begin == 0:
                    batch_idx, duplicate = np.unique(batch_idx, return_counts=True)
                else:
                    duplicate = None

                # read from hdf5 data file
                with h5py.File(self.path, "r") as f:
                    batch = f["data"][batch_idx]

                # handle duplicate
                if duplicate is not None and (duplicate != 1).any():
                    ind = np.repeat(np.arange(len(duplicate)), duplicate)
                    batch = batch[ind]

                self.step += 1
                yield batch
                logger.debug(f"Epoch {self.epoch}, Step {self.step}")

    def get_restart_info(self) -> tuple[dict[str, Any], int, int, np.ndarray[int]]:
        """
        Get restart information.
        """
        return self.rng_state, self.epoch, self.step, self.residual_idx


# ------------------------------------------------------------------------------
# DataLoader for testing
# ------------------------------------------------------------------------------


class ChunkEvaluator(DataLoader):
    """
    Context manager for the evaluation data loader from file.

    Parameters
    ----------
    config:
        The configuration of the data loader.
    """

    TYPE = "test"

    def __init__(self, config: DataConfig, start_ind: int, end_ind: int):
        super().__init__(config)

        # data loader configuration
        self.batch_size = config.batch_size
        self.path = config.path

        # chunk info
        self.start_ind = start_ind
        self.end_ind = end_ind

    def batch_iterator(self) -> Generator[np.ndarray, None, None]:
        """
        Generate batches of sentences.
        """
        # iterate over batches
        end = self.start_ind
        while end < self.end_ind:
            begin, end = end, end + self.batch_size
            end = min(end, self.end_ind)

            # read from hdf5 data file
            with h5py.File(self.path, "r") as f:
                batch = f["data"][begin:end]
            yield batch

    def get_restart_info(self) -> None:
        return


class FileEvaluator:
    """
    DDP Context manager for the evaluation data loader from file.

    Parameters
    ----------
    config:
        The configuration of the data loader.
    """

    TYPE = "test"

    def __init__(self, config: DataConfig):
        # create chunks
        rank = get_rank()
        world_size = get_world_size()
        start_ind = rank * config.n_data // world_size
        end_ind = (rank + 1) * config.n_data // world_size

        logger.info(f"Intializing Evaluator on device {rank + 1}/{world_size}")
        self.worker = ChunkEvaluator(config, start_ind, end_ind)

    def __enter__(self):
        return self.worker.__enter__()

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        return self.worker.__exit__(exc, value, tb)
