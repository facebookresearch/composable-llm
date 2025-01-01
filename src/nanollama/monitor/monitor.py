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
import os
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from ..cluster import is_master_process
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
        # wandb name
        if self.logging.wandb.name == "":
            self.logging.wandb.name = self.name

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

        # profile directory
        if self.profiler.path == "":
            self.profiler.path = str(Path(self.dir) / "profiler")

        # add information about the slurm job id
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            path = Path(self.logging.path)
            self.logging.path = str(path / job_id)

        # add discriminative information if array job
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id:
            suffix = f"task_{task_id}"
            self.logging.wandb.name += suffix
            self.checkpoint.path = str(Path(self.checkpoint.path) / suffix)
            self.profiler.path = str(Path(self.profiler.path) / suffix)
            self.logging.metric_path = str(path / "metrics" / f"{suffix}.json")

            # keep a mapping of job_id to task_id
            if is_master_process():
                path.mkdir(parents=True, exist_ok=True)
                with open(path / "id_mapping", "a") as f:
                    f.write(f"task {task_id}: {job_id}\n")

        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__manual_post_init__"):
                module.__manual_post_init__()


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
    ) -> None:
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

    def __call__(self) -> None:
        # manual garbage collection
        if trigger_update(self.state, self.gc_period):
            logger.info("garbage collection")
            gc.collect()

        # call managers
        self.checkpointer()
        self.profiler()

    def report_metrics(self, metrics: dict) -> None:
        """
        Report the metrics to monitor.
        """
        self.logger(metrics)

    def __exit__(
        self,
        exc: type[BaseException],
        value: BaseException,
        tb: TracebackType,
    ):
        gc.collect()

        # close managers
        self.logger.__exit__(exc, value, tb)
        self.checkpointer.__exit__(exc, value, tb)
        self.profiler.__exit__(exc, value, tb)

        if exc is not None:
            logger.error(f"Exception: {value}")
            import traceback

            logger.info("".join(traceback.format_exception(exc, value, tb)))
