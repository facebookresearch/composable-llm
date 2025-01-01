"""
Distributed Computing Manager

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import logging
from dataclasses import dataclass, field
from types import TracebackType

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .os_environment import OsEnvironment, set_os_environment
from .slurm import SlurmConfig
from .utils import get_local_rank, get_rank, get_world_size, is_distributed_job

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    os_environment: OsEnvironment = field(default_factory=OsEnvironment)

    device: str = "cuda"
    compile_model: bool = True
    backend: str = "nccl"

    def __manual_post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__manual_post_init__"):
                module.__manual_post_init__()


class ClusterManager:
    def __init__(self, config: ClusterConfig):
        self.backend = config.backend
        self.device = config.device
        self.compile = config.compile_model
        set_os_environment(config.os_environment)

    def __enter__(self):
        """
        Initialize distributed environment
        """
        if is_distributed_job():
            rank = get_rank()
            local_rank = get_local_rank()
            world_size = get_world_size()
            dist.init_process_group(backend=self.backend, rank=rank, world_size=world_size)
            print(f"Setting up device ranked {rank + 1} / {world_size}")

            self.device = f"cuda:{local_rank}"
        else:
            self.device = torch.device(self.device)
            print(f"Running on {self.device}")
        return self

    def initialize_model(self, model: nn.Module) -> nn.Module:
        """
        Initialize the model by casting it to the device, compiling and parallelizing it according to configuration.
        """
        model = model.to(device=self.device)
        if self.compile:
            model = torch.compile(model)
        logger.info("Done building model")
        local_rank = get_local_rank()
        world_size = get_world_size()
        if world_size > 1:
            logger.info("Parallelizing model")
            model = DDP(model, device_ids=[local_rank])
        return model

    def __exit__(
        self,
        exc: type[BaseException],
        value: BaseException,
        tb: TracebackType,
    ):
        """
        Exit distributed environment
        """
        rank = get_rank()
        world_size = get_world_size()
        logger.info(f"Exiting distributed environment {rank + 1} / {world_size}")
        if is_distributed_job():
            dist.destroy_process_group()
