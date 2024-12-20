"""
Distributed Computing Manager

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import logging
import os
import random
import subprocess
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .slurm import SlurmConfig

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
# Utilities functions
# -------------------------------------------------------------------------------


@lru_cache()
def is_torchrun_job() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def is_slurm_job() -> bool:
    # torch_run preempts slurm jobs
    return "SLURM_JOB_ID" in os.environ and not is_torchrun_job()


@lru_cache()
def is_distributed_job() -> bool:
    return is_torchrun_job() or is_slurm_job()


@lru_cache()
def get_rank() -> int:
    if is_torchrun_job():
        return int(os.environ["RANK"])
    elif is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


@lru_cache()
def get_local_rank() -> int:
    if is_torchrun_job():
        return int(os.environ["LOCAL_RANK"])
    elif is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0


@lru_cache()
def get_world_size() -> int:
    if is_torchrun_job():
        return int(os.environ["WORLD_SIZE"])
    elif is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    else:
        return 1


@lru_cache()
def is_master_process() -> bool:
    return get_rank() == 0


# -------------------------------------------------------------------------------
# Configuration Class - OS Environment
# -------------------------------------------------------------------------------


@dataclass
class OsEnvironment:
    OMP_NUM_THREADS: str = "1"


def set_os_environment(config: OsEnvironment):
    """
    TODO
    """
    env_vars = asdict(config)
    print(env_vars)

    # # When using Triton, it attempts to locate prebuilt kernels in a cache
    # # located at ~/.triton/cache, but when that's backed by NFS this can fail
    # # with a "OSError: [Errno 116] Stale file handle" error. If we were to set
    # # it to a local directory it would belong to the first user who created it
    # # and it would fail for the job of any other successive user assigned to
    # # that machine. To avoid all this mess we use a temporary per-process cache.
    # triton_cache_dir = tempfile.mkdtemp()
    # atexit.register(shutil.rmtree, triton_cache_dir, ignore_errors=True)
    # env_vars["TRITON_CACHE_DIR"] = triton_cache_dir

    # # We change the tmp dir to /scratch in case it's slurm job
    # # This avoids filling up the host's usually limited tmpfs
    # # A full tmpfs leads to very slow creation of processes and weird bugs
    # if is_slurm_job():
    #     new_tmp = f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}"
    #     if os.path.exists(new_tmp):
    #         env_vars["TMP_DIR"] = new_tmp

    for name, value in env_vars.items():
        if os.environ.get(name) != str(value):
            os.environ[name] = str(value)
            print(f"WARNING: Setting {name} to {value}")

    # set up slurm environment
    if is_slurm_job():
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
        master_addr = hostnames.split()[0].decode("utf-8")

        MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
        job_id = int(os.environ["SLURM_JOB_ID"])
        rng = random.Random(job_id)
        master_port = rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)


# -------------------------------------------------------------------------------
# Configuration Class - Distributed Configuration
# -------------------------------------------------------------------------------


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

        # handling type not recognized by OmegaConf
        self.device = torch.device(self.device)


class ClusterManager:
    def __init__(self, config: ClusterConfig):
        self.backend = config.backend
        self.device = config.device
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

        return self

    def parallelize_model(self, model: nn.Module):
        local_rank = get_local_rank()
        world_size = get_world_size()
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])
        return model

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit distributed environment
        """
        world_size = get_world_size()
        if world_size > 1:
            dist.destroy_process_group()
