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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

import wandb

from ..distributed import is_master_process

logger = logging.getLogger(__name__)


@dataclass
class WandbConfig:
    active: bool = False

    # Wandb user and project name
    entity: str = ""
    project: str = "composition"
    name: str = field(init=False, default="")
    path: str = field(init=False, default="")

    def __check_init__(self):
        """Check validity of arguments and fill in missing values."""
        assert self.name, "name was not set"
        assert self.path, "path was not set"


class WandbManager:
    def __init__(self, config: WandbConfig, run_config: Any = None):
        self.active = config.active and is_master_process()
        if not self.active:
            return

        # open wandb api
        os.environ["WANDB_DIR"] = config.path
        id_file = Path(config.path) / "wandb.id"
        id_file.parent.mkdir(parents=True, exist_ok=True)
        self.open(config.entity, config.project, id_file, config.name)

        # log run configuration to wandb
        if run_config:
            config_dict = asdict(run_config)
            self.run.config.update(config_dict, allow_val_change=True)
            logger.info("Run configuration has been logged to wandb.")

    def open(self, entity: str, project: str, id_file: str, name: str) -> None:
        # Read run id from id file if it exists
        if os.path.exists(id_file):
            resuming = True
            with open(id_file) as file:
                run_id = file.read().strip()
        else:
            resuming = False

        if resuming:
            # Check whether run is still alive
            api = wandb.Api()
            run_state = api.run(f"{entity}/{project}/{run_id}").state
            if run_state == "running":
                logger.warning(f"Run with ID: {run_id} is currently active and running.")
                sys.exit(1)

            self.run = wandb.init(
                project=project,
                entity=entity,
                id=run_id,
                resume="must",
            )
            logger.info(f"Resuming run with ID: {run_id}")

        else:
            # Starting a new run
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
            )
            logger.info(f"Starting new run with ID: {self.run.id}")

            # Save run id to id file
            with open(id_file, "w") as file:
                file.write(self.run.id)

    def __enter__(self) -> "WandbManager":
        return self

    def __call__(self, metrics: dict) -> None:
        """Report metrics to wanbd."""
        if not self.active:
            return
        assert "step" in metrics, f"metrics should contain a step key.\n{metrics=}"
        wandb.log(metrics, step=metrics["step"])

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """Close wandb api."""
        if not self.active:
            return
        # Handle exception
        try:
            if exc is not None:
                # Log exception in wandb
                wandb.finish(exit_code=1)
            else:
                wandb.finish()
        except Exception as e:
            logger.warning(e)
