"""
Mamba model

https://arxiv.org/abs/2405.21060

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .blocklm import BlockLanguageModel, BlockLanguageModelConfig
from .norm import RMSNorm
from .ssm.utils_mamba import mamba_chunk_scan_combined
from .ssm.wrapper_causal_conv1d import causal_conv1d_fn

# ------------------------------------------------------------------------------
# Mamba Layer
# ------------------------------------------------------------------------------


class SSM(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        state_dim: int,
        nb_heads: int,
        nb_groups: int,
        conv_size: Optional[int],
        # Fused kernel and sharding options
        chunk_size: int = 256,
    ):
        super().__init__()

        # dimension
        self.emb_dim = emb_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.nb_heads = nb_heads
        self.nb_groups = nb_groups
        self.head_dim = hidden_dim // nb_heads
        assert hidden_dim % nb_heads == 0, f"Hidden dim must be divisible by nb_heads: {hidden_dim} % {nb_heads} != 0"

        # others
        self.chunk_size = chunk_size

        # Order: [z, x, B, C, dt]
        hidden_size = 2 * self.hidden_dim + 2 * self.nb_groups * self.state_dim + self.nb_heads
        self.W_in = nn.Linear(self.emb_dim, hidden_size, bias=False)
        self.W_out = nn.Linear(self.hidden_dim, self.emb_dim, bias=False)

        # convolution
        self.conv_size = conv_size
        if conv_size is not None:
            self.conv_dim = self.hidden_dim + 2 * self.nb_groups * self.state_dim
            assert (self.conv_dim % 8 == 0) and (conv_size in [2, 3, 4]), (
                "Causal conv1d only supports conv_size in [2, 3, 4] and hidden_dim/head_dim % 8 == 0, "
                f"got {self.conv_dim} and {conv_size}"
            )
            self.conv_weight = nn.Parameter(torch.empty((self.conv_dim, conv_size)))

        self.A_log = nn.Parameter(torch.empty(nb_heads))
        self.D = nn.Parameter(torch.ones(nb_heads))

        self.norm = RMSNorm(self.hidden_dim, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        zxbcdt = self.W_in(x)
        out1, xBC, dt = torch.split(zxbcdt, [self.hidden_dim, self.conv_dim, self.nb_heads], dim=-1)

        # packed convolution
        if self.conv_size is not None:
            xBC = causal_conv1d_fn(x=xBC.transpose(1, 2), weight=self.conv_weight, activation="silu").transpose(1, 2)

        # unpack
        BC_size = self.nb_groups * self.state_dim
        x, B, C = torch.split(xBC, [self.hidden_dim, BC_size, BC_size], dim=-1)

        x = x.view(bsz, seq_len, self.nb_heads, self.head_dim)  # (bsz, seq_len, n_heads, head_dim)

        A_log = self.A_log
        A = -torch.exp(A_log.float())
        B = B.view(bsz, seq_len, self.nb_groups, self.state_dim)  # (bsz, seq_len, ngroups, state_dim)
        C = C.view(bsz, seq_len, self.nb_groups, self.state_dim)  # (bsz, seq_len, ngroups, state_dim)

        out2 = mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=self.chunk_size,
            D=self.D,
        )  # (bsz, seq_len, n_heads, head_dim)

        out2 = out2.view(bsz, seq_len, self.hidden_dim)

        out = F.silu(out1) * out2
        out = self.W_out(self.norm(out))
        return out

    def reset_parameters(self, init_std: float, factor: float) -> None:
        """Weight initialization"""
        # input
        in_std = init_std or (self.emb_dim ** (-0.5))
        # in_std = in_std / factor
        nn.init.trunc_normal_(self.W_in.weight, std=in_std, a=-3 * in_std, b=3 * in_std)

        # output
        out_std = init_std or (self.hidden_dim ** (-0.5))
        out_std = out_std / factor
        nn.init.trunc_normal_(self.W_out.weight, std=out_std, a=-3 * in_std, b=3 * in_std)

        # convolution
        if self.conv_size is not None:
            conv_std = init_std or (self.conv_size ** (-0.5))
            nn.init.trunc_normal_(self.conv_weight, std=conv_std, a=-3 * conv_std, b=3 * conv_std)

        # Initialize A and D
        self.A_log.data.uniform_(1, 16)
        self.A_log.data.log_()
        self.D.data.fill_(1.0)


# ------------------------------------------------------------------------------
# Configuration Class
# ------------------------------------------------------------------------------


@dataclass
class MambaBlockConfig:
    emb_dim: int = 0

    # mamba block parameters
    nb_heads: int = 0
    hidden_dim: int = 0
    state_dim: int = 128
    nb_groups: int = 1
    conv_size: Optional[int] = None
    ssm_chunk_size: int = 256
    norm_eps: float = 1e-5


# ------------------------------------------------------------------------------
# Mamba Block
# ------------------------------------------------------------------------------


class MambaBlock(nn.Module):
    def __init__(self, config: MambaBlockConfig):
        super().__init__()

        self.ssm_norm = RMSNorm(config.emb_dim, config.norm_eps)
        self.ssm = SSM(
            emb_dim=config.emb_dim,
            hidden_dim=config.hidden_dim,
            state_dim=config.state_dim,
            nb_heads=config.nb_heads,
            nb_groups=config.nb_groups,
            conv_size=config.conv_size,
            chunk_size=config.ssm_chunk_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.ssm_norm(x))
        return x

    def reset_parameters(self, init_std: float = None, factor: float = 1.0) -> None:
        """Weight initialization"""
        self.ssm_norm.reset_parameters()
        self.ssm.reset_parameters(init_std, factor)


# ------------------------------------------------------------------------------
# Mamba Architecture
# ------------------------------------------------------------------------------


@dataclass
class MambaConfig(BlockLanguageModelConfig):
    block: MambaBlockConfig = field(default_factory=MambaBlockConfig)

    def __post_init__(self):
        super().__post_init__()

        # Inherit parameters from the block model configuration.
        for attr in ["emb_dim", "norm_eps", "init_std"]:
            setattr(self.block, attr, getattr(self, attr))

        # default scaling of ffn dimension
        if not self.block.hidden_dim:
            self.block.hidden_dim = 3 * self.emb_dim


class Mamba(BlockLanguageModel):
    """Mamba Language Model"""

    def __init__(self, config: MambaConfig):
        super().__init__(config, block=MambaBlock)
