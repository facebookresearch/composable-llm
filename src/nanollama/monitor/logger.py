"""
Logging Managor

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import json
import logging
import time
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from types import TracebackType

from ..cluster import get_hostname, get_rank, is_master_process
from .wandb import WandbConfig, WandbManager

logger = getLogger(__name__)


@dataclass
class LoggerConfig:
    path: str = ""
    metric_path: str = ""
    period: int = 100
    level: str = "INFO"
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        self.level = self.level.upper()
        assert self.level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def __manual_post_init__(self):
        if self.metric_path == "" and self.path != "":
            self.metric_path = str(Path(self.path) / "metrics.json")


# -------------------------------------------------------------------------------
# Logging Manager
# -------------------------------------------------------------------------------


class Logger:
    def __init__(self, config: LoggerConfig):
        self.path = Path(config.path)
        self.path.mkdir(parents=True, exist_ok=True)
        device_rank = get_rank()
        log_file = self.path / f"device_{device_rank}.log"

        self.metric = Path(config.metric_path)
        self.metric.parent.mkdir(parents=True, exist_ok=True)
        self.wandb = None

        # Initialize logging stream
        if is_master_process():
            # the master node gets to log more information
            handlers = [logging.StreamHandler()]
            if config.wandb.active:
                self.wandb = WandbManager(config.wandb, log_dir=self.path)
        else:
            handlers = [logging.FileHandler(log_file, "a")]

        logging.basicConfig(
            level=getattr(logging, config.level),
            format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            handlers=handlers,
        )

        logger.info(f"Running on machine {get_hostname()}")
        logger.info(f"Logging to {self.path}")

    def __enter__(self):
        """
        Open logging files (and wandb api).
        """
        self.metric = open(self.metric, "a")
        if self.wandb is not None:
            self.wandb.__enter__()

    def __call__(self, metrics: dict) -> None:
        """
        Report the metrics to monitor.
        """
        metrics |= {"ts": time.time()}
        print(json.dumps(metrics), file=self.metric, flush=True)
        if self.wandb:
            self.wandb(metrics, step=metrics["step"])

    def __exit__(
        self,
        exc: type[BaseException],
        value: BaseException,
        trace: TracebackType,
    ):
        """
        Close logging files (and wandb api).
        """
        self.metric.close()
        if self.wandb is not None:
            self.wandb.__exit__(exc, value, trace)
