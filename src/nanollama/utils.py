"""
Utils functions

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

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
