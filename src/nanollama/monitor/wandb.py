"""
Wandb Logger

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from types import TracebackType
from typing import Optional

import wandb
from omegaconf import OmegaConf

from .monitor import Monitor

logger = logging.getLogger(__name__)


@dataclass
class WandbConfig:
    active: Optional[bool] = False

    # Wandb user and project name
    entity: str = ""
    project: str = "composition"
    name: str = field(init=False)
    id_file: str = field(init=False)

    def __post_init__(self):
        self.name = ""
        self.id_file = ""

    def __manual_post_init__(self):
        """Check validity of arguments and fill in missing values."""
        assert self.name, "name was not set"
        assert self.id_file, "name was not set"


class WandbManager(Monitor):
    def __init__(self, config: WandbConfig):
        self.entity = config.entity
        self.project = config.project
        self.name = config.name
        self.id_file = config.id_file

    def __enter__(self):
        """
        Open wandb api.
        """
        # Read run id from id file if it exists
        if os.path.exists(self.id_file):
            resuming = True
            with open(self.id_file) as file:
                run_id = file.read().strip()
        else:
            resuming = False

        if resuming:
            # Check whether run is still alive
            api = wandb.Api()
            run_state = api.run(f"{self.entity}/{self.project}/{run_id}").state
            if run_state == "running":
                logger.warning(f"Run with ID: {run_id} is currently active and running.")
                sys.exit(1)

            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                id=run_id,
                resume="must",
            )
            logger.info(f"Resuming run with ID: {run_id}")

        else:
            # Starting a new run
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
            )
            logger.info(f"Starting new run with ID: {self.run.id}")

            # Save run id to id file
            with open(self.id_file, "w") as file:
                file.write(self.run.id)

    def report_objects(self, run_config: dict) -> None:
        config_dict = OmegaConf.to_container(OmegaConf.structured(run_config))
        self.run.config.update(config_dict, allow_val_change=True)
        logger.info("Run configuration has been logged to wandb.")

    def report_metrics(self, metrics: dict) -> None:
        """
        Report metrics to wanbd.
        """
        wandb.log(metrics, step=metrics["step"])

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """
        Close wandb api.
        """
        # Handle exception
        try:
            if exc is not None:
                # Log exception in wandb
                wandb.finish(exit_code=1)
            else:
                wandb.finish()
        except Exception as e:
            logger.warning(e)

    def check_run_state(self, run_id: str) -> str:
        api = wandb.Api()
        try:
            run = api.run(f"{self.entity}/{self.project}/{run_id}")
            return run.state
        except wandb.errors.CommError:
            logger.error("Wandb run not found or API error.")
        return "no_state"
