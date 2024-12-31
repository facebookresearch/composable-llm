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
        return {
            "rng_state": self.rng_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rng_state = state_dict["rng_state"]
