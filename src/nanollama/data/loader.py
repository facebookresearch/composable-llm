"""
Dataloader

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from collections.abc import Generator
from dataclasses import dataclass
from logging import getLogger
from multiprocessing import Process, Queue
from queue import Empty, Full
from types import TracebackType
from typing import Any

import numpy as np
import torch

logger = getLogger("nanollama")


@dataclass
class DataConfig:
    asynchronous: bool = True
    buffer_size: int = 4


class DataLoader:
    """
    Dataloader

    Usage:
    ```python
    with DataLoader(*args) as data_loader:
        for batch, _ in data_loader:
            pass
    ```

    Parameters
    ----------
    config:
        The configuration of the data loader.
    state:
        The state of the data loader.
    """

    TYPE = "train"

    def __init__(self, config: DataConfig):
        # data loader configuration
        self.asynchronous = config.asynchronous

        # asynchronous data loader: a worker writes batches in a buffer, that a reader consumes
        if self.asynchronous:
            self.process = Process(target=self.async_batch_creator)
            self.buffer = Queue(maxsize=config.buffer_size)

        # initialize the batch generator
        self.generator = self.batch_iterator()

    def __enter__(self):
        if self.TYPE == "train":
            logger.info("Entering dataloader.")
        if self.asynchronous:
            self.process.start()
        return self

    def batch_iterator(self) -> Generator[np.ndarray, None, None]:
        """
        Generate batches of sentences.
        """
        raise NotImplementedError

    def get_restart_info(self) -> Any:
        """
        Get restart information.
        """
        raise NotImplementedError

    def async_batch_creator(self) -> None:
        """
        Asynchronous batch generation, writting batches to the buffer.
        """
        # loop on batch creation
        while True:
            try:
                batch = next(self.generator)
                restart_info = self.get_restart_info()
                batch = torch.from_numpy(batch).long()
            # handle end of data asynchrounously
            except StopIteration:
                batch = restart_info = None

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

    def __iter__(self) -> "DataLoader":
        """
        Return an iterator over batches
        """
        return self

    def __next__(self) -> tuple[torch.Tensor, Any]:
        """
        Get the next batch of sentences.
        """
        if self.asynchronous:
            batch, restart_info = self.async_get_batch()
            # handle end of data asynchrounously
            if batch is None:
                raise StopIteration
        else:
            batch = next(self.generator)
            restart_info = self.get_restart_info()
            batch = torch.from_numpy(batch).long()
        return batch, restart_info

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        if self.TYPE == "train":
            logger.info("Exiting dataloader.")
        if self.asynchronous:
            self.process.kill()
            self.buffer.close()
