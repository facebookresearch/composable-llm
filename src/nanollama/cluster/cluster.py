import logging
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from .slurm import SlurmConfig

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    # slurm configuration
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # distributed config
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = True

    # GPU communication backend
    backend: str = "nccl"

    def __manual_post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__manual_post_init__"):
                module.__manual_post_init__()

        # handling type not recognized by OmegaConf
        self.device = torch.device(self.device)


class ClusterManager:
    def __init__(self, config: ClusterConfig):
        self.backend = config.backend

    def __enter__(self):
        init_process_group(backend=self.backend)

        self.device_rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        logger.info(f"Running on ddp rank: {self.device_rank} / {self.world_size}")

        device = f"cuda:{self.local_rank}"
        torch.cuda.set_device(device)

        return self

    def parallelize_model(self, model: nn.Module):
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.local_rank])
        return model

    def __exit__(self, exc_type, exc_value, traceback):
        if self.world_size > 1:
            destroy_process_group()
