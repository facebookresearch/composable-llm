"""
Generic Orchestrator managing:
- garbage collection
- logging to file
- logging to wandb

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import gc
from dataclasses import dataclass
from logging import getLogger

import torch

from .monitor import Monitor

logger = getLogger(__name__)


@dataclass
class UtilityConfig:
    seed: int = 42  # reproducibility
    period: int = 1000  # garbage collection frequency


class UtilityManager(Monitor):
    def __init__(self, config: UtilityConfig):
        super().__init__(config)
        self.seed = config.seed

    def __enter__(self):
        # set seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # disable garbage collection
        gc.disable()
        return self

    def update(self) -> None:
        """
        Running utility functions: garbage collection.
        """
        logger.info("garbage collection")
        gc.collect()
