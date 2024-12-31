"""
Vanilla Training State Manager

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
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
        return {"data": self.data.state_dict(), "optim": self.optim.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.data.load_state_dict(state_dict["data"])
        self.optim.load_state_dict(state_dict["optim"])
