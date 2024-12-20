import json
import logging
import subprocess
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    # slurm configuration
    partition: str = ""
    nodes: int = 1  # number of nodes to run the job on.
    nb_gpus: int = 1  # number of GPUs required per node.
    nb_cpus: int = 16  # number of CPUs allocated per GPU.
    mem: str = ""  # amount of memory to allocate per node.
    time: int = -1  # time limit of the job (in minutes).

    signal_time: int = 120  # time between US2 signal and job terminaion (in seconds)

    # slurm extra configuration
    slurm_extra: str = ""  # placeholder
    constraint: str = ""  # constraint on the nodes.
    exclude: str = ""  # nodes to exclude.
    account: str = ""
    qos: str = ""

    # cluster environment
    script_extra: str = ""

    # distributed config
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        assert self.slurm_extra == "", "slurm_extra is a placeholder and should not be set"
        for name in ["exclude", "qos", "account", "constraint"]:
            if getattr(self, name):
                self.slurm_extra += f"#SBATCH --{name}={getattr(self, name)}\n"

    def __manual_post_init__(self):
        # handling type not recognized by OmegaConf
        self.device = torch.device(self.device)

        # if partition, time or memory was not set
        priorities, max_times, memories = {}, {}, {}
        if self.partition == "" or self.time == -1 or self.mem == "":
            priorities, max_times, memories = self.extract_sinfo()
        if self.partition == "":
            self.partition = min(priorities.keys(), key=lambda k: priorities[k]["job_factor"])
            logger.info(f"No partition specified default to {self.partition}")
        if self.time == -1:
            self.time = max_times[self.partition]
            logger.info(f"No time specified, default to {self.time} minutes")
        if self.mem == "":
            self.mem = memories[self.partition]
            logger.info(f"No memory specified, default to {self.mem}MB")

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
