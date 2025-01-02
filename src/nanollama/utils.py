"""
Utils functions

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, TypeVar

from torch.distributed.checkpoint.stateful import Stateful

from .data import DataLoaderState
from .optim import OptimizerState

# -------------------------------------------------------------------------------
# Training state
# -------------------------------------------------------------------------------


@dataclass
class TrainState(Stateful):
    data: DataLoaderState
    optim: OptimizerState

    def state_dict(self) -> dict[str, Any]:
        return {"data": self.data.state_dict(), "optim": self.optim.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.data.load_state_dict(state_dict["data"])
        self.optim.load_state_dict(state_dict["optim"])


# -------------------------------------------------------------------------------
# Initialization of nested configuration classes
# -------------------------------------------------------------------------------


T = TypeVar("T")


def initialize_nested_dataclass(dataclass_type: type[T], data: dict[str, Any]) -> T:
    """
    Recursively initializes a dataclass from a nested dictionary.
    """
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} is not a dataclass")
    field_values = {}
    for data_field in fields(dataclass_type):
        fname, ftype = data_field.name, data_field.type
        if fname in data:
            value = data[fname]
            if is_dataclass(ftype):
                field_values[fname] = initialize_nested_dataclass(ftype, value)
            else:
                field_values[fname] = value
    return dataclass_type(**field_values)
