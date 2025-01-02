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

import os
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

from ..distributed import is_master_process
from .checkpoint import CheckpointConfig
from .logger import LoggerConfig
from .profiler import ProfilerConfig
from .utility import UtilityConfig
from .wandb import WandbConfig

logger = getLogger(__name__)


@dataclass
class OrchestratorConfig:
    log_dir: str = ""
    name: str = "composition_default"

    # submanagers
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggerConfig = field(default_factory=LoggerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    utils: UtilityConfig = field(default_factory=UtilityConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """

        # logging directory
        if not self.log_dir:
            log_dir = Path.home() / "logs" / self.name
            self.log_dir = str(log_dir)
            print(f"No logging directory set. Setting it to {self.log_dir}")
        else:
            self.log_dir = os.path.expandvars(self.log_dir)
            log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # add discriminative information if array job
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id:
            task_id = str(task_id)
        else:
            task_id = ""

        # checkpoint directory
        self.checkpoint.path = str(log_dir / "checkpoints" / task_id)

        # profile directory
        self.profiler.path = str(log_dir / "metrics" / task_id)

        # logging related
        self.logging.metric_path = str(log_dir / "metrics" / task_id / "train_eval.json")
        self.wandb.path = str(log_dir / "wandb" / task_id)
        self.wandb.name = self.name
        stdout_dir = log_dir / "logs"

        # handling grid job
        if task_id:
            job_id = os.environ.get("SLURM_JOB_ID")

            # keep a mapping of job_id to task_id
            if is_master_process():
                stdout_dir.mkdir(parents=True, exist_ok=True)
                with open(stdout_dir / "id_mapping", "a") as f:
                    f.write(f"task {task_id}: {job_id}\n")

            stdout_dir = stdout_dir / job_id
            self.wandb.name += f"_task_{task_id}"

        self.logging.stdout_path = str(stdout_dir)

        # check validity of submodule
        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()
