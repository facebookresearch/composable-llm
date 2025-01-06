"""
Generic Orchestrator Configuration

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import os
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path, PosixPath

from ..distributed import is_master_process
from .checkpoint import CheckpointConfig
from .logger import LoggerConfig
from .profiler import ProfilerConfig
from .utility import UtilityConfig
from .wandb import WandbConfig

logger = getLogger("nanollama")


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
            logger.info(f"No logging directory set. Setting it to {self.log_dir}")
        else:
            self.log_dir = os.path.expandvars(self.log_dir)
            log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # add discriminative information if array job
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

        # checkpoint directory
        self.checkpoint.path = str(log_dir / "checkpoints" / task_id)

        # profile directory
        self.profiler.path = str(log_dir / "metrics" / task_id)

        # logging related
        self.logging.stdout_path = str(log_dir / "logs" / task_id)
        self.logging.metric_path = str(log_dir / "metrics" / task_id)
        self.wandb.path = str(log_dir / "wandb" / task_id)
        self.wandb.name = f"{self.name}_{task_id}"

        # keep a mapping of job_id to task_id
        if task_id:
            job_id = os.environ.get("SLURM_JOB_ID")
            path = log_dir / "stdout"
            if is_master_process():
                path.mkdir(parents=True, exist_ok=True)
                with open(path / "id_mapping", "a") as f:
                    f.write(f"task {task_id}: {job_id}\n")

        # check validity of submodule
        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()


# -----------------------------------------------------------------------------
# Evaluation Orchestrator
# -----------------------------------------------------------------------------


@dataclass
class EvalOrchestratorConfig:
    parent_dir: str = ""  # log dir of parent training run
    log_dir: str = ""
    name: str = "composition_default"

    # train step information
    train_step: int = 0  # train step at which evaluation is performed
    task_id: int = 0  # task id of the training job

    # submanagers
    checkpoint_path: str = field(init=False, default="")
    logging: LoggerConfig = field(default_factory=LoggerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    utils: UtilityConfig = field(default_factory=UtilityConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """

        # parent directory (same logic as OrchestratorConfig)
        if not self.parent_dir:
            parent_dir = Path.home() / "logs" / self.name
            self.parent_dir = str(parent_dir)
            logger.info(f"No logging directory set. Setting it to {self.parent_dir}")
        else:
            self.parent_dir = os.path.expandvars(self.parent_dir)
            parent_dir = Path(self.parent_dir)
        parent_dir.mkdir(parents=True, exist_ok=True)
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        self.checkpoint_path = str(parent_dir / "checkpoints" / str(self.task_id))

        # logging directory
        if not self.log_dir:
            log_dir: PosixPath = parent_dir / "evals" / str(self.task_id) / f"{self.train_step:010d}"
            self.log_dir = str(log_dir)
            logger.info(f"No logging directory set. Setting it to {self.log_dir}")
        else:
            self.log_dir = os.path.expandvars(self.log_dir)
            log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # wandb directory (single dir for any steps)
        self.wandb.path = str(log_dir.parent / "wandb" / task_id)
        self.wandb.name = f"{self.name}_{self.task_id}_eval_{task_id}"

        # same logic as OrchestratorConfig
        self.profiler.path = str(log_dir / "metrics" / task_id)
        self.logging.stdout_path = str(log_dir / "logs" / task_id)
        self.logging.metric_path = str(log_dir / "metrics" / task_id)
        if task_id:
            job_id = os.environ.get("SLURM_JOB_ID")
            path = log_dir / "stdout"
            if is_master_process():
                path.mkdir(parents=True, exist_ok=True)
                with open(path / "id_mapping", "a") as f:
                    f.write(f"task {task_id}: {job_id}\n")
        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()
