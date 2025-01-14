"""
Feed-forward network.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Feed-forward Layer
# -----------------------------------------------------------------------------


class FeedForward(nn.Module):
    """
    Feed-forward network in transformer architecture.

    Parameters
    ----------
    emb_dim:
        embedding dimension of the inputs
    ffn_dim:
        hidden dimension of the MLP
    """

    def __init__(
        self,
        emb_dim: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.ffn_dim = ffn_dim
        self.fc1 = nn.Linear(emb_dim, 2 * ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, out2 = self.fc1(x).chunk(2, dim=-1)
        out = F.silu(out1) * out2
        out = self.fc2(out)
        return out

    def reset_parameters(self, init_std: float = None, factor: float = 1.0) -> None:
        """
        Weight initialization
        """
        in_init_std = init_std or (self.emb_dim ** (-0.5))
        out_init_std = init_std or (self.ffn_dim ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor
        nn.init.trunc_normal_(
            self.fc1.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )
        nn.init.trunc_normal_(
            self.fc2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )
