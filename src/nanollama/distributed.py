"""
Distributed Computing Manager

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import os
import random
import socket
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from logging import getLogger
from types import TracebackType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------


@lru_cache
def is_torchrun_job() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache
def is_slurm_job() -> bool:
    # torch_run preempts slurm jobs
    return "SLURM_JOB_ID" in os.environ and not is_torchrun_job()


@lru_cache
def is_distributed_job() -> bool:
    return is_torchrun_job() or is_slurm_job()


@lru_cache
def get_rank() -> int:
    if is_torchrun_job():
        return int(os.environ["RANK"])
    elif is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


@lru_cache
def get_local_rank() -> int:
    if is_torchrun_job():
        return int(os.environ["LOCAL_RANK"])
    elif is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0


@lru_cache
def get_world_size() -> int:
    if is_torchrun_job():
        return int(os.environ["WORLD_SIZE"])
    elif is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    else:
        return 1


@lru_cache
def is_master_process() -> bool:
    return get_rank() == 0


@lru_cache
def get_hostname() -> str:
    return socket.gethostname()


# ------------------------------------------------------------------------------
# OS Environment
# ------------------------------------------------------------------------------


@dataclass
class OsEnvironment:
    OMP_NUM_THREADS: str = "1"


def set_os_environment(config: OsEnvironment) -> None:
    """
    Set OS environment variables based on configuration.
    """
    env_vars = asdict(config)

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
            logger.info(f"OS: Setting {name} to {value}")

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


@contextmanager
def clean_environment() -> Generator[None, None, None]:
    distrib_names = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
        "DORA_FORCE_DISTRIB",
    )
    os_environment = {
        x: os.environ.pop(x)
        for x in os.environ
        if x.startswith(("SLURM_", "SLURMD_", "SRUN_", "SBATCH_", "SUBMITIT_", "WANDB_")) or x in distrib_names
    }
    try:
        yield
    finally:
        os.environ.update(os_environment)


# ------------------------------------------------------------------------------
# Cluster Configuration and Manager
# ------------------------------------------------------------------------------


@dataclass
class ClusterConfig:
    device: torch.device = None
    compile_model: bool = True
    backend: str = "nccl"

    # submanager
    os_environment: OsEnvironment = field(default_factory=OsEnvironment)

    def __post_init__(self):
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["device"] = str(self.device)
        return output


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
            logger.info(f"Setting up device ranked {rank + 1} / {world_size}")
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            logger.info(f"Running on {self.device}")
        return self

    def initialize_model(self, model: nn.Module) -> nn.Module:
        """
        Initialize the model by casting it to the device, compiling and parallelizing it according to configuration.
        """
        model = model.to(device=self.device)
        if self.compile:
            logger.info("Compiling model")
            model = torch.compile(model)
        logger.info("Done building model")
        local_rank = get_local_rank()
        world_size = get_world_size()
        if world_size > 1:
            logger.info("Parallelizing model")
            model = DDP(model, device_ids=[local_rank])
        return model

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """
        Exit distributed environment
        """
        rank = get_rank()
        world_size = get_world_size()
        logger.info(f"Exiting distributed environment {rank + 1} / {world_size}")
        if is_distributed_job():
            dist.destroy_process_group()
