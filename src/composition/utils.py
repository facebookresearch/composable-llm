"""
Utils functions

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import os
import sys
from json import JSONEncoder
from pathlib import PosixPath

import torch

from .train import TrainState

# -----------------------------------------------------------------------------
# Frequency checker
# -----------------------------------------------------------------------------


def trigger_update(state: TrainState, period: int) -> bool:
    """
    Return a boolean indicating whether the current step is a multiple of `period`.

    Parameters
    ----------
    state : TrainState
        Current training state containing optimization and accumulation steps.
    period : int
        Number of gradient updates between each check. If -1, always return False.
    """

    if period != -1:
        return False
    return (state.optim.step % period == 0) and (state.optim.acc_step == 0)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_torch_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Json Serializer
# -----------------------------------------------------------------------------


class JsonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, PosixPath):
            return str(obj)
        return super().default(obj)


# -----------------------------------------------------------------------------
# Slurm signal handling
# -----------------------------------------------------------------------------


def handle_sig(signum, frame):
    print(f"Requeuing after {signum}...", flush=True)
    os.system(f'scontrol requeue {os.environ["SLURM_ARRAY_JOB_ID"]}_{os.environ["SLURM_ARRAY_TASK_ID"]}')
    sys.exit(-1)


def handle_term(signum, frame):
    print("Received TERM.", flush=True)
