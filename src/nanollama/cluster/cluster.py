import json
import logging
import subprocess
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
        # handling type not recognized by OmegaConf
        self.device = torch.device(self.device)

    @staticmethod
    def extract_sinfo() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        # retrieve partition max times (slow but run only once)

        sinfo = json.loads(subprocess.check_output("sinfo --json", shell=True))["sinfo"]
        priorities: dict[str, int] = {}
        max_times: dict[str, int] = {}
        memories: dict[str, int] = {}

        for info in sinfo:
            partition = info["partition"]["name"]
            if partition in priorities:
                continue

            priorities[partition] = info["partition"]["priority"]
            memories[partition] = info["memory"]["maximum"]  # in MB

            if info["partition"]["maximums"]["time"]["infinite"]:
                max_times[partition] = 14 * 24 * 60  # 14 days
            else:
                max_times[partition] = info["partition"]["maximums"]["time"]["number"]  # in minutes

        return priorities, max_times, memories


class ClusterManager:
    def __init__(self, config: ClusterConfig):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass
