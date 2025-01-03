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
import os
import time
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from traceback import format_exception
from types import TracebackType
from typing import Any

from ..distributed import get_hostname, get_rank, is_master_process

# this file control the global logger
logger = getLogger()


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


class Logger:
    def __init__(self, config: LoggerConfig):
        self.stdout_path = Path(config.stdout_path)
        self.stdout_path.mkdir(parents=True, exist_ok=True)
        rank = get_rank()
        stdout_file = self.stdout_path / f"device_{rank}.log"

        self.metric = Path(config.metric_path + f"_{rank}.json")
        self.metric.parent.mkdir(parents=True, exist_ok=True)

        # remove existing handler
        logger.handlers.clear()

        # Initialize logging stream
        log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s")
        log_level = getattr(logging, config.level)

        handler = logging.FileHandler(stdout_file, "a")
        handler.setLevel(log_level)
        handler.setFormatter(log_format)
        logger.addHandler(handler)

        # log to console
        if is_master_process() and "SLURM_JOB_ID" not in os.environ:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            handler.setFormatter(log_format)
            logger.addHandler(handler)

        logger.info(f"Running on machine {get_hostname()}")
        logger.info(f"Logging to {self.stdout_path}")

    def __enter__(self) -> "Logger":
        """
        Open logging files.
        """
        self.metric = open(self.metric, "a")
        return self

    def __call__(self, metrics: dict[str, Any]) -> None:
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
