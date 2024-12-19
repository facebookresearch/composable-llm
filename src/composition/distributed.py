"""
Distributed Computing Manager

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import os
from functools import lru_cache


@lru_cache()
def is_torchrun_job() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def is_slurm_job() -> bool:
    # torch_run preempts slurm jobs
    return "SLURM_JOB_ID" in os.environ and not is_torchrun_job()


@lru_cache()
def get_global_rank() -> int:
    if is_torchrun_job():
        return int(os.environ["RANK"])
    elif is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0
