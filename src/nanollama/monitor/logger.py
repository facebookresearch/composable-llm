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
from traceback import format_exception
from types import TracebackType
from typing import Any

from ..distributed import get_hostname, get_rank, is_master_process
from .monitor import Monitor

logger = getLogger(__name__)


@dataclass
class LoggerConfig:
    period: int = 100
    level: str = "INFO"
    stdout_path: str = field(init=False, default="")
    metric_path: str = field(init=False, default="")

    def __post_init__(self):
        self.level = self.level.upper()
        assert self.level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def __check_init__(self):
        """Check validity of arguments."""
        assert self.stdout_path, "stdout_path was not set"
        assert self.metric_path, "metric_path was not set"


# -------------------------------------------------------------------------------
# Logging Manager
# -------------------------------------------------------------------------------


class Logger(Monitor):
    def __init__(self, config: LoggerConfig):
        self.stdout_path = Path(config.stdout_path)
        self.stdout_path.mkdir(parents=True, exist_ok=True)
        device_rank = get_rank()
        stdout_file = self.stdout_path / f"device_{device_rank}.log"

        self.metric = Path(config.metric_path)
        self.metric.parent.mkdir(parents=True, exist_ok=True)

        # Initialize logging stream
        if is_master_process():
            handlers = [logging.StreamHandler()]
        else:
            handlers = [logging.FileHandler(stdout_file, "a")]

        logging.basicConfig(
            level=getattr(logging, config.level),
            format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            handlers=handlers,
        )

        logger.info(f"Running on machine {get_hostname()}")
        logger.info(f"Logging to {self.stdout_path}")

    def __enter__(self):
        """
        Open logging files.
        """
        self.metric = open(self.metric, "a")

    def __call__(self):
        """Unused function, call should be made throught the report_metrics method."""
        pass

    def report_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Report metrics to file.
        """
        metrics |= {"ts": time.time()}
        print(json.dumps(metrics), file=self.metric, flush=True)

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """
        Close logging files. Log exceptions if any.
        """
        self.metric.close()
        if exc is not None:
            logger.error(f"Exception: {value}")
            logger.info("".join(format_exception(exc, value, tb)))
