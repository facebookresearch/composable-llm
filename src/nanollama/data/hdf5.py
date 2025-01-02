"""
Dataloader from hdf5 file

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import logging
import os
from collections.abc import Generator
from dataclasses import dataclass
from multiprocessing import Process, Queue
from queue import Empty, Full
from types import TracebackType
from typing import Any

import h5py
import numpy as np
import torch
from numpy.random import SeedSequence, default_rng

from ..distributed import get_rank

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    path: str = ""
    n_data: int = -1
    seq_len: int = -1
    batch_size: int = -1
    seed: int = 0
    asynchronous: bool = True  # asynchronous data loading
    buffer_size: int = 4  # number of batches to bufferize asynchronously for data loading

    def __post_init__(self):
        self.path = os.path.expandvars(self.path)


@dataclass
class DataLoaderState:
    rng_state: dict[str, Any]
    epoch: int = 0
    step: int = 0  # batch step
    residual_idx: np.ndarray[int] = None  # residual data from the previous epoch

    def __post_init__(self):
        if self.residual_idx is None:
            self.residual_idx = np.array([], dtype=int)

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
    # generate independent seeds
    ss = SeedSequence(config.seed)
    rank = get_rank()
    seed = ss.spawn(rank + 1)[-1]

    # recover state from seeds
    rng_state = default_rng(seed).bit_generator.state
    return DataLoaderState(rng_state=rng_state)


class FileDataLoaderManager:
    """
    Context manager for the data loader from file.

    Parameters
    ----------
    config:
        The configuration of the data loader.
    state:
        The state of the data loader.

    Attributes
    ----------
    rng:
        Random number generator.

    Yields
    ------
    tuple[np.ndarray, dict[str, Any]]
        The generated batch of sentences and the state of the random number generator.
    """

    def __init__(self, config: DataConfig, state: DataLoaderState):
        self.n_data = config.n_data
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.path = config.path
        self.asynchronous = config.asynchronous

        # track randomness
        self.rng = default_rng()

        # ensure consistency of randomness over restart
        self.rng.bit_generator.state = state.rng_state
        logger.debug(f"RNG: {state}")

        # asynchronous data loader: a worker writes batches in a buffer, that a reader consumes
        if self.asynchronous:
            self.process = Process(target=self.async_create_batch)
            self.buffer = Queue(maxsize=config.buffer_size)

        # track data processing state
        self.epoch = state.epoch
        self.step = state.step
        self.residual_idx = state.residual_idx

    def __enter__(self):
        logger.info("Entering dataloader.")
        if self.asynchronous:
            self.process.start()
        return self

    def get_batch(self) -> tuple[np.ndarray, tuple]:
        """
        Generate a batch of sentences.
        """
        return next(self.batch_generator())

    def batch_generator(self) -> Generator[tuple[np.ndarray, tuple], None, None]:
        """
        Generate batches of sentences.
        """
        # restart information
        rng_state = self.rng.bit_generator.state
        epoch = self.epoch
        step = self.step
        residual_idx = self.residual_idx

        # iterate over epochs
        while True:
            # schedule data processing order
            epoch_idx = self.rng.permutation(self.n_data)

            # add residual data from the previous epoch
            epoch_idx = np.append(residual_idx, epoch_idx)

            # iterate over batches
            while True:
                begin, end = step * self.batch_size, (step + 1) * self.batch_size
                batch_idx = epoch_idx[begin:end]

                # do not process last batch alone if it is smaller than batch_size
                if batch_idx.shape[0] < self.batch_size:
                    rng_state = self.rng.bit_generator.state
                    epoch += 1
                    step = 0
                    residual_idx = batch_idx
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

                step += 1
                restart_info = (rng_state, epoch, step, batch_idx)
                yield batch, restart_info

    def async_create_batch(self) -> None:
        """
        Asynchronous batch generation, writting batches to the buffer.
        """
        # loop on batch creation
        while True:
            batch, restart_info = self.get_batch()
            # convert to torch tensor
            batch = torch.from_numpy(batch).long()

            # put it in the buffer
            while True:
                try:
                    self.buffer.put((batch, restart_info), timeout=0.1)
                    break
                # if the buffer is full, wait until there is space
                except Full:
                    logger.debug("Buffer is full. Waiting for data comsumption.")
            logger.debug("New batch put in the buffer.")

    def async_get_batch(self) -> tuple[np.ndarray, tuple]:
        """
        Asynchronous batch acquisition, reading batches from the buffer.
        """
        # read batch from the buffer
        while True:
            try:
                return self.buffer.get(timeout=0.1)
            # if the buffer is full, wait until it is filled
            except Empty:
                logger.debug("Buffer is empty. Waiting for data.")

    def __next__(self) -> tuple[np.ndarray, tuple]:
        if self.asynchronous:
            return self.async_get_batch()
        else:
            batch, restart_info = self.get_batch()
            batch = torch.from_numpy(batch).long()
            return (batch, restart_info)

    def __exit__(
        self,
        exc: type[BaseException],
        value: BaseException,
        tb: TracebackType,
    ):
        logger.info("Exiting dataloader.")
        if self.asynchronous:
            self.process.kill()
            self.buffer.close()
