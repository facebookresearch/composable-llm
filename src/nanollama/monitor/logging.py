"""
Logging Managor

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import getLogger
from pathlib import Path

from ..cluster import get_rank
from .wandb import WandbConfig, WandbManager

logger = getLogger(__name__)


@dataclass
class LoggingConfig:
    dir: str = ""
    period: int = 100
    level: str = "INFO"
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        assert self.level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


# -------------------------------------------------------------------------------
# Logging Manager
# -------------------------------------------------------------------------------


class LoggingManager:
    def __init__(self, config: LoggingConfig):
        self.dir = Path(config.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metric = self.dir.parent / "metrics.jsonl"
        device_rank = get_rank()
        log_file = self.dir / f"device_{device_rank}.log"

        self.wandb = None
        # the master node gets to log more information
        if device_rank == 0:
            # handlers = [logging.StreamHandler(), logging.FileHandler(log_file, "a")]
            handlers = [logging.StreamHandler()]
            if config.wandb.active:
                self.wandb = WandbManager(config.wandb, log_dir=self.dir)
        else:
            handlers = [logging.FileHandler(log_file, "a")]

        logging.basicConfig(
            level=getattr(logging, config.level),
            format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            handlers=handlers,
        )

        logger.info(f"Logging to {self.dir}")

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
