"""
Training State Manager

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

from dataclasses import dataclass
from typing import Any

from torch.distributed.checkpoint.stateful import Stateful

from .data import DataLoaderState
from .optim import OptimizerState


@dataclass
class TrainState(Stateful):
    data: DataLoaderState
    optim: OptimizerState

    def state_dict(self) -> dict[str, Any]:
        return {
            "data.rng_state": self.data.rng_state,
            "optim": {"step": self.optim.step, "acc_step": self.optim.acc_step},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.data.rng_state = state_dict["data.rng_state"]
        optim_state = state_dict["optim"]
        self.optim.step = optim_state["step"]
        self.optim.acc_step = optim_state["acc_step"]
