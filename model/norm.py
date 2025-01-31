"""
Normalization layers

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import torch
import torch.nn as nn

# ------------------------------------------------------------------------------
# Normalization Layer
# ------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    RMS normalization layer.

    Parameters
    ----------
        dim:
            dimension of the input tensor
        eps:
            numerical stability parameter
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)
        return (output * self.weight).type_as(x)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
