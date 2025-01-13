"""
Initialization of the model module

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from abc import ABC, abstractmethod

import torch

from .transfomer import Transformer, TransformerConfig


class Model(ABC):
    """
    Abstract class for models
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_nb_flop(self, seq_len: int = None, mode: str = "both") -> None:
        """
        Number of flop to process a new token

        Parameters
        ----------
        seq_len:
            Sequence length.
        mode:
            Whether to consider the forward, backward pass or both
        """
        pass
