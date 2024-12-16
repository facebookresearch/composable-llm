"""
Training State Manager

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

from dataclasses import dataclass

from .data import DataLoaderState
from .optim import OptimizerState


@dataclass
class TrainState:
    data: DataLoaderState
    optim: OptimizerState
