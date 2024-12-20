import logging
from dataclasses import dataclass, field

import torch

from .slurm import SlurmConfig

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    # slurm configuration
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # distributed config
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass
