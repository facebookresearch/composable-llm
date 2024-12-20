"""
Computing Manager

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import logging
import os
from dataclasses import dataclass, field

import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .distributed import DistributedConfig
from .os_environment import OsEnvironment, set_os_environment
from .slurm import SlurmConfig

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    os_environment: OsEnvironment = field(default_factory=OsEnvironment)

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
        self.backend = config.distributed.backend
        set_os_environment(config.os_environment)

    def __enter__(self):
        # setup_torch_distributed(self.distributed)
        # world_mesh = get_device_mesh(self.distributed)

        # # need dp world size and rank
        # dp_mesh = world_mesh["dp_replicate"]
        # dp_degree = dp_mesh.size()
        # dp_rank = dp_mesh.get_local_rank()
        # if self.distributed.dp_shard > 1:
        #     dp_rank = dp_rank * dp_degree + world_mesh["dp_shard"].get_local_rank()
        #     dp_degree *= world_mesh["dp_shard"].size()

        dist.init_process_group(backend=self.backend)

        self.device_rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(f"Setting up device ranked {self.device_rank + 1} / {self.world_size}")

        self.device = f"cuda:{self.local_rank}"
        # torch.cuda.set_device(self.device)

        return self

    def parallelize_model(self, model: nn.Module):
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.local_rank])
        return model

    def __exit__(self, exc_type, exc_value, traceback):
        if self.world_size > 1:
            dist.destroy_process_group()
