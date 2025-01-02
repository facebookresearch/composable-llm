"""
Utils functions

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import copy
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, TypeVar, get_args, get_origin

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
    data = copy.deepcopy(data)
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} is not a dataclass")
    field_values = {}
    for data_field in fields(dataclass_type):
        fname, ftype = data_field.name, data_field.type
        if fname in data:
            value = data.pop(fname)
            # Check if the field is a list
            if get_origin(ftype) is list:
                item_type = get_args(ftype)[0]
                if is_dataclass(item_type):
                    # Initialize each item in the list
                    field_values[fname] = [initialize_nested_dataclass(item_type, item) for item in value]
                else:
                    # Directly assign the list if items are not dataclasses
                    field_values[fname] = value
            elif is_dataclass(ftype):
                # Recursively initialize nested dataclass
                field_values[fname] = initialize_nested_dataclass(ftype, value)
            else:
                try:
                    # Directly assign the value
                    field_values[fname] = ftype(value)
                except TypeError:
                    # print(f"Initializing {fname}:{value} without type checking ({ftype}).")
                    field_values[fname] = value
    if data:
        for fname in data:
            print(f"Field '{fname}' ignored when initializing {dataclass_type}.")
    return dataclass_type(**field_values)
