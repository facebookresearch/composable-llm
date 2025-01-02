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
from types import TracebackType

from ..distributed import is_master_process
from .checkpoint import CheckpointConfig, Checkpointer
from .logger import Logger, LoggerConfig
from .monitor import Monitor
from .profiler import Profiler, ProfilerConfig
from .utility import UtilityConfig, UtilityManager
from .wandb import WandbConfig, WandbManager

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
            log_dir = str(Path.home() / "logs" / self.name)
            self.log_dir = log_dir
            print(f"No logging directory set. Setting it to {self.log_dir}")
        else:
            log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # add discriminative information if array job
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id:
            # keep a mapping of job_id to task_id
            if is_master_process():
                job_id = os.environ.get("SLURM_JOB_ID")
                with open(log_dir / "id_mapping", "a") as f:
                    f.write(f"task {task_id}: {job_id}\n")
            log_dir = log_dir / task_id

        # checkpoint directory
        self.checkpoint.path = str(log_dir / "checkpoints")

        # profile directory
        self.profiler.path = str(log_dir / "metrics")

        # logging directory and paths
        log_dir = log_dir / "logs"
        self.logging.stdout_path = str(log_dir)
        self.logging.metric_path = str(log_dir / "metrics" / "train_eval.json")
        self.wandb.id_file = str(log_dir / "wandb.id")

        # wandb name
        self.wandb.name = self.name

        # add discriminative information if array job
        if task_id:
            self.wandb.name += f"_task_{task_id}"

        # check validity of submodule
        for module in self.__dict__.values():
            if hasattr(module, "__manual_post_init__"):
                module.__manual_post_init__()


class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.state = None

        # submanagers
        self.submanagers: list[Monitor] = [
            UtilityManager(config.utils),
            Logger(config.logging),
            Checkpointer(config.checkpoint),
            Profiler(config.profiler),
        ]

        if config.wandb.active and is_master_process():
            os.environ["WANDB_DIR"] = config.logging.stdout_path
            self.submanagers.append(WandbManager(config.wandb))

    def __enter__(self):
        for manager in self.submanagers:
            manager.__enter__()
        return self

    def report_objects(self, **kwargs) -> None:
        """
        Report the objects to monitor.

        This function is useful since we initialize the Monitor before the model is built.
        """
        for manager in self.submanagers:
            manager.report_objects(**kwargs)

    def __call__(self) -> None:
        for manager in self.submanagers:
            manager()

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        for manager in self.submanagers:
            manager.__exit__(exc, value, tb)
