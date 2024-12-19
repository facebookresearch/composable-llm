"""
Wandb Logger

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import wandb
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class WandbConfig:
    active: Optional[bool] = False

    # Wandb user and project name
    entity: str = ""
    project: Optional[str] = "llm"
    name: Optional[str] = "run"


class WandbManager:
    def __init__(self, config: WandbConfig, log_dir: str):
        self.entity = config.entity
        self.project = config.project
        self.name = config.name
        self.id_file = os.path.join(log_dir, "wandb.id")

    def __enter__(self):
        # Read run id from id file if it exists
        if os.path.exists(self.id_file):
            resuming = True
            with open(self.id_file, "r") as file:
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

            run = wandb.init(
                project=self.project,
                entity=self.entity,
                id=run_id,
                resume="must",
            )
            logger.info(f"Resuming run with ID: {run_id}")

        else:
            # Starting a new run
            run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
            )
            logger.info(f"Starting new run with ID: {run.id}")

            # Save run id to id file
            with open(self.id_file, "w") as file:
                file.write(run.id)

    def report_run_config(self, run_config):
        if self.run is not None:
            config_dict = OmegaConf.to_container(OmegaConf.structured(run_config))
            self.run.config.update(config_dict, allow_val_change=True)
            logger.info("Run configuration has been logged to wandb.")

    def report_metrics(self, metrics: dict, step: int):
        wandb.log(metrics, step=step)

    def __exit__(self, exc_type, exc_value, traceback):
        # Handle exception
        try:
            if exc_type is not None:
                # Log exception in wandb
                wandb.finish(exit_code=1)
            else:
                wandb.finish()
        except Exception as e:
            logger.warning(e)

    def check_run_state(self, run_id):
        api = wandb.Api()
        try:
            run = api.run(f"{self.entity}/{self.project}/{run_id}")
            return run.state
        except wandb.errors.CommError:
            logger.error("Wandb run not found or API error.")
        return "no_state"
