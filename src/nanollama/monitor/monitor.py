"""
Generic Orchestrator managing:
- garbage collection
- logging to file
- logging to wandb

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import gc
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from ..train import TrainState
from ..utils import trigger_update
from .checkpoint import CheckpointConfig, Checkpointer
from .logger import Logger, LoggerConfig
from .profiler import Profiler, ProfilerConfig

logger = getLogger(__name__)


@dataclass
class MonitorConfig:
    dir: str = ""
    name: str = "composition_default"
    overwrite: bool = False  # whether to overwrite logging directory

    # reproducibility
    seed: int = 42

    # garbage collection frequency
    gc_period: int = 1000

    # submanagers
    logging: LoggerConfig = field(default_factory=LoggerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    # evaluation
    async_eval_gpus: Optional[int] = None

    # probing

    def __manual_post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__manual_post_init__"):
                module.__manual_post_init__()

        # directory
        if not self.dir:
            self.dir = str(Path.home() / "logs" / self.name)
            print(f"No logging directory set. Setting it to {self.dir}")

        # logging directory
        if not self.logging.path:
            path = Path(self.dir) / "logs"
            self.logging.path = str(path)

        # checkpoint directory
        if self.checkpoint.path == "":
            self.checkpoint.path = str(Path(self.dir) / "checkpoints")

        # wandb name
        if self.logging.wandb.name == "":
            self.logging.wandb.name = self.name

        # profile directory
        if self.profiler.path == "":
            self.profiler.path = str(Path(self.dir) / "profiler")


class Orchestrator:
    def __init__(self, config: MonitorConfig):
        self.seed = config.seed
        self.gc_period = config.gc_period

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.state = None

        # submanagers
        self.logger = Logger(config.logging)
        self.checkpointer = Checkpointer(config.checkpoint)
        self.profiler = Profiler(config.profiler)

    def __enter__(self):
        # set seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # disable garbage collection
        gc.disable()

        # open managers
        self.logger.__enter__()
        self.checkpointer.__enter__()
        self.profiler.__enter__()
        return self

    def report_objects(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler.LambdaLR,
        state: TrainState,
        config: Any,
    ):
        """
        Report the objects to monitor.

        This function is useful since we initialize the Monitor before the model is built.
        """
        # self.model = model
        # self.optimizer = optimizer
        # self.scheduler = scheduler
        self.state = state

        # load checkpoint if it exists
        self.checkpointer.report_objects(model, optimizer, scheduler, state)
        if self.logger.wandb:
            self.logger.wandb.report_objects(config)

        self.nb_params = sum([p.numel() for p in model.parameters()])
        logger.info(f"Model built with {self.nb_params:,} parameters")

        # report objects to submanagers
        self.profiler.report_objects(state)

    def __call__(self):
        # manual garbage collection
        if trigger_update(self.state, self.gc_period):
            logger.info("garbage collection")
            gc.collect()

        # call managers
        self.checkpointer()
        self.profiler()

    def report_metrics(self, metrics: dict):
        """
        Report the metrics to monitor.
        """
        self.logger(metrics)

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        gc.collect()

        # close managers
        self.logger.__exit__(exc_type, exc_value, traceback)
        self.checkpointer.__exit__(exc_type, exc_value, traceback)
        self.profiler.__exit__(exc_type, exc_value, traceback)
