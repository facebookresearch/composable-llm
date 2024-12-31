"""
Initialization of the data module

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DataLoaderState:
    rng_state: dict[str, Any]

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError("This is an abstract method and should be implemented by the child class.")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raise NotImplementedError("This is an abstract method and should be implemented by the child class.")

    def report_restart_info(self, restart_info: Any) -> None:
        raise NotImplementedError("This is an abstract method and should be implemented by the child class.")
