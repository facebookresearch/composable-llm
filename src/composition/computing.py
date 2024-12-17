from dataclasses import dataclass

import torch


@dataclass
class ComputeConfig:
    # slurm configuration
    partition: str = "scavenge"
    nodes: int = 1  # number of nodes to run the job on.
    nb_gpus: int = 1  # number of GPUs required per node.
    nb_cpus: int = 16  # number of CPUs allocated per GPU.
    mem: str = "256G"  # amount of memory to allocate per node.
    time: int = 60  # time limit of the job (in minutes).
    # time: int = -1  # time limit of the job (in minutes).

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
