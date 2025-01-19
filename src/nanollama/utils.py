"""
Utils functions

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import copy
from collections.abc import MutableMapping
from dataclasses import dataclass, fields, is_dataclass
from logging import getLogger
from typing import Any, TypeVar, Union, get_args, get_origin

from torch.distributed.checkpoint.stateful import Stateful

from .data import DataLoaderState
from .optim import OptimizerState

logger = getLogger("nanollama")

# ------------------------------------------------------------------------------
# Training state
# ------------------------------------------------------------------------------


@dataclass
class TrainState(Stateful):
    data: DataLoaderState
    optim: OptimizerState

    def state_dict(self) -> dict[str, Any]:
        return {"data": self.data.state_dict(), "optim": self.optim.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.data.load_state_dict(state_dict["data"])
        self.optim.load_state_dict(state_dict["optim"])


# ------------------------------------------------------------------------------
# Configuration utilities
# ------------------------------------------------------------------------------


def flatten_config(
    config: dict[str, Any], separator: str = ".", flatten_list: bool = False, _parent_key: str = ""
) -> dict[str, Any]:
    """
    Flatten a nested dictionary into a dot-separated format.

    Parameters
    ----------
    config:
        The dictionary to flatten
    separator:
        The string used to separate flattened keys
    flatten_list:
        Whether to flatten lists
    _parent_key:
        The string to prepend to dictionary's keys

    Return
    ------
    A flattened dictionary
    """
    args = (separator, flatten_list)
    items = []
    for k, v in config.items():
        new_key = f"{_parent_key}{separator}{k}" if _parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, *args, new_key).items())
        elif flatten_list and isinstance(v, list):
            for _k, _v in enumerate(v):
                items.extend(flatten_config({str(_k): _v}, *args, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat configuration into a nested configuration."""
    nested = {}
    for key, value in config.items():
        keys = key.split(".")
        d = nested
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return nested


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
