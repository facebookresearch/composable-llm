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
from dataclasses import asdict, dataclass, field
from logging import getLogger
from pathlib import Path
from traceback import format_exception
from types import TracebackType
from typing import Any

import torch

from ..distributed import get_hostname, get_rank, is_master_process

logger = getLogger("nanollama")


@dataclass
class LoggerConfig:
    period: int = 100
    level: str = "INFO"
    stdout_path: str = field(init=False, default="")
    metric_path: str = field(init=False, default="")

    def __post_init__(self):
        self.stdout_path = self.stdout_path
        self.metric_path = self.metric_path
        self.level = self.level.upper()
        assert self.level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def __check_init__(self):
        """Check validity of arguments."""
        assert self.stdout_path, "stdout_path was not set"
        assert self.metric_path, "metric_path was not set"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionnary to reinitialize it.
        """
        output = asdict(self)
        output.pop("stdout_path")
        output.pop("metric_path")
        return output


# ------------------------------------------------------------------------------
# Logging Manager
# ------------------------------------------------------------------------------


class Logger:
    def __init__(self, config: LoggerConfig):
        rank = get_rank()

        self.path = Path(config.metric_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.metric = str(self.path / f"raw_{rank}.jsonl")

        path = Path(config.stdout_path)
        path.mkdir(parents=True, exist_ok=True)
        stdout_file = path / f"device_{rank}.log"

        # remove existing handler
        getLogger().handlers.clear()

        # Initialize logging stream
        log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s")
        log_level = getattr(logging, config.level)
        logger.setLevel(log_level)

        handler = logging.FileHandler(stdout_file, "a")
        handler.setFormatter(log_format)
        logger.addHandler(handler)

        # log to console
        if is_master_process() and "SLURM_JOB_ID" not in os.environ:
            handler = logging.StreamHandler()
            handler.setFormatter(log_format)
            logger.addHandler(handler)
            logger.info(f"Logging to {path}")

        logger.info(f"Running on machine {get_hostname()}")

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

    def report_statistics(self, model: torch.nn.Module) -> None:
        """
        Report gobal statistics about the model.
        """
        if is_master_process():
            numel = sum([p.numel() for _, p in model.named_parameters()])
            with open(self.path / "info_model.jsonl", "a") as f:
                print(json.dumps({"model_params": numel}), file=f, flush=True)
            logger.info(f"Model has {numel} parameters.")

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """
        Close logging files. Log exceptions if any.
        """
        self.metric.close()
        if exc is not None:
            logger.error(f"Exception: {value}")
            logger.info("".join(format_exception(exc, value, tb)))
