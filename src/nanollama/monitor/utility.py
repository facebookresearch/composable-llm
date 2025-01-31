"""
Utility Manager:
- seed setting
- garbage collection

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import gc
from dataclasses import dataclass
from logging import getLogger
from types import TracebackType

import torch

from .monitor import Monitor

logger = getLogger("nanollama")


@dataclass
class UtilityConfig:
    seed: int = 42  # reproducibility
    period: int = 1000  # garbage collection frequency


class UtilityManager(Monitor):
    def __init__(self, config: UtilityConfig):
        super().__init__(config)
        self.seed = config.seed

    def __enter__(self) -> "UtilityManager":
        # set seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        # disable garbage collection
        gc.disable()
        return self

    def update(self) -> None:
        """
        Running utility functions: garbage collection.
        """
        logger.info("garbage collection")
        gc.collect()

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        # enable garbage collection
        gc.enable()
        return
