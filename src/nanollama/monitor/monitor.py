"""
Monitor class managing:
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
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PosixPath
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from ..cluster.utils import get_rank
from ..train import TrainState
from ..utils import trigger_update
from .wandb import WandbConfig, WandbManager

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    # logging
    name: str = "composition_default"
    dir: str = ""
    overwrite: bool = False  # whether to overwrite logging directory
    log_period: int = 100
    log_level: str = "INFO"
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # reproducibility
    seed: int = 42

    # garbage collection frequency
    gc_period: int = 1000

    # evaluation
    async_eval_gpus: Optional[int] = None

    # probing
    # profiling

    def __post_init__(self):
        assert self.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def __manual_post_init__(self):
        if not self.dir:
            self.dir = str(Path.home() / "logs" / self.name)


class MonitorsManager:
    def __init__(self, config: MonitorConfig):
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.gc_period = config.gc_period

        self.model = None
        self.optimizer = None
        self.scheduler = None

        # logging
        self.logger = LoggerManager(config)

    def __enter__(self):

        # disable garbage collection
        gc.disable()

        # open logger
        self.logger.__enter__()
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
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = state

        if self.logger.wandb:
            self.logger.wandb.report_run_config(config)

        self.nb_params = sum([p.numel() for p in model.parameters()])
        logger.info(f"Model built with {self.nb_params:,} parameters")

    def __call__(self):
        # manual garbage collection
        if trigger_update(self.state, self.gc_period):
            logger.info("garbage collection")
            gc.collect()

    def report_metrics(self, metrics: dict):
        """
        Report the metrics to monitor.
        """
        self.logger(metrics)

    def __exit__(self, exc_type, exc_value, traceback):
        gc.collect()

        # close logger
        self.logger.__exit__(exc_type, exc_value, traceback)


class LoggerManager:
    def __init__(self, config: MonitorConfig):
        self.log_dir = self.get_log_dir(config.dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metric = self.log_dir.parent / "metrics.jsonl"
        device_rank = get_rank()
        log_file = self.log_dir / f"device_{device_rank}.log"

        self.wandb = None
        # the master node gets to log more information
        if device_rank == 0:
            # handlers = [logging.StreamHandler(), logging.FileHandler(log_file, "a")]
            handlers = [logging.StreamHandler()]
            if config.wandb.active:
                self.wandb = WandbManager(config.wandb, log_dir=self.log_dir)
        else:
            handlers = [logging.FileHandler(log_file, "a")]

        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            handlers=handlers,
        )

        logger.info(f"Logging to {self.log_dir}")

    def __enter__(self):
        """
        Open logging files (and wandb api).
        """
        self.metric = open(self.metric, "a")
        if self.wandb is not None:
            self.wandb.__enter__()

    def __call__(self, metrics: dict):
        """
        Report the metrics to monitor.
        """
        metrics.update({"created_at": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(metrics), file=self.metric, flush=True)

        if self.wandb:
            self.wandb.report_metrics(metrics, step=metrics["step"])

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Close logging files (and wandb api).
        """
        self.metric.close()
        if self.wandb is not None:
            self.wandb.__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def get_log_dir(prefix: str) -> PosixPath:
        path = Path(prefix) / "logs"
        if os.environ.get("SLURM_JOB_ID"):
            return path / os.environ["SLURM_JOB_ID"]
        return path
