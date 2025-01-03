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
from logging import getLogger
from typing import Any, TypeVar, Union, get_args, get_origin

from torch.distributed.checkpoint.stateful import Stateful

from .data import DataLoaderState
from .optim import OptimizerState

logger = getLogger("nanollama")

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


def initialize_nested_object(object_type: type[T], data: dict[str, Any], inplace: bool = True) -> T:
    """
    Recursively initializes a typed object from a nested dictionary.
    """
    if not inplace:
        data = copy.deepcopy(data)

    # trivial cases
    if data is None or object_type is Any:
        return data
    args = get_args(object_type)

    # dataclasses
    if is_dataclass(object_type):
        field_values = {}
        for data_field in fields(object_type):
            fname, ftype = data_field.name, data_field.type
            if fname in data:
                value = data.pop(fname)
                field_values[fname] = initialize_nested_object(ftype, value)
            else:
                logger.debug(f"Field '{fname}' not found in {object_type}.")
        for fname in data:
            logger.warning(f"Field '{fname}' ignored when initializing {object_type}.")
        return object_type(**field_values)

    # list
    elif get_origin(object_type) is list and len(args) == 1:
        return [initialize_nested_object(args[0], item) for item in data]

    # dict
    elif get_origin(object_type) is dict and len(args) == 2:
        return {initialize_nested_object(args[0], k): initialize_nested_object(args[1], v) for k, v in data.items()}

    # union
    elif get_origin(object_type) is Union:
        for arg in args:
            try:
                return initialize_nested_object(arg, data)
            except (TypeError, ValueError):
                continue

    # primitive types
    try:
        return object_type(data)
    except (TypeError, ValueError):
        logger.warning(f"Initializing {object_type}:{data} without type checking.")
        return data
