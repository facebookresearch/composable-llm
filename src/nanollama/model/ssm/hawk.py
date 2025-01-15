"""
Hawk model

https://arxiv.org/abs/2402.19427

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import torch
from torch import nn
from torch.autograd.function import FunctionCtx
from torch.nn import functional as F

from ..feedforward import FeedForward
from ..norm import RMSNorm
from .utils_rnn import RNNBlockConfig, conv1d, scan

# ------------------------------------------------------------------------------
# Square root with gradient clipping
# ------------------------------------------------------------------------------


_MAX_SQRT_GRADIENT: float = 1000.0


class SqrtBoundDerivative(torch.autograd.Function):
    """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """The forward pass, which is a normal `sqrt`."""
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        """The backward pass, which clips the `sqrt` gradient."""
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))
        return grad_output / torch.sqrt(clipped_x_times_4)


def sqrt_bounded_derivative(x: torch.Tensor) -> torch.Tensor:
    return SqrtBoundDerivative.apply(x)


# ------------------------------------------------------------------------------
# RGLRU
# ------------------------------------------------------------------------------


class RGLRU(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        nb_heads: int,
        head_dim: int,
        conv_size: int = None,
    ):
        super().__init__()

        # dimension
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.nb_heads = nb_heads
        assert emb_dim % nb_heads == 0, f"emb_dim {emb_dim} must be divisible by n_heads {nb_heads}"

        assert head_dim * nb_heads == emb_dim, (
            f"dim {emb_dim} must be equal to n_heads {nb_heads} * head_dim {head_dim}"
        )

        # matrices
        self.W_in = nn.Linear(emb_dim, 2 * emb_dim, bias=False)

        # convolution
        self.conv_size = conv_size
        if conv_size is not None:
            assert (emb_dim % 8 == 0) and (conv_size in [2, 3, 4]), (
                "Causal conv1d only supports conv_size in [2, 3, 4] and hidden_dim/head_dim % 8 == 0, "
                f"got {emb_dim} and {conv_size}"
            )
            self.conv_weight = nn.Parameter(torch.empty((self.emb_dim, conv_size)))

        self.c = 8.0
        self.register_parameter("a", nn.Parameter(torch.empty(head_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        if self.conv_size is not None:
            # conv1d_w = log_stats(self.conv_weight, "conv1d.w")
            conv1d_w = self.conv_weight
            x = conv1d(
                x=x.transpose(1, 2),
                conv_weight=conv1d_w,
                impl="parallel",
            ).transpose(1, 2)

        gate_x, gate_a = self.W_in(x).chunk(2, dim=-1)
        gate_x = F.sigmoid(gate_x)
        gate_a = F.sigmoid(gate_a)

        gate_x = gate_x.transpose(1, 2).reshape(bsz * self.nb_heads, self.head_dim, seqlen)
        gate_a = gate_a.transpose(1, 2).reshape(bsz * self.nb_heads, self.head_dim, seqlen)

        a = F.softplus(self.a).unsqueeze(0).unsqueeze(-1).expand(bsz * self.nb_heads, self.head_dim, seqlen)

        log_a = -self.c * gate_a * a
        a = log_a.exp()
        multiplier = sqrt_bounded_derivative(1.0 - (2.0 * log_a).exp())

        x = x.transpose(1, 2).reshape(bsz * self.nb_heads, self.head_dim, seqlen)

        out = scan(
            a=a.contiguous(),
            b=(multiplier * gate_x * x).contiguous(),
            impl="parallel",
        )

        out = out.view(bsz, self.emb_dim, seqlen).transpose(1, 2)
        # h = log_stats(h, "hidden_state")

        return out

    def reset_parameters(self, init_std: float, factor: float) -> None:
        """
        Weight initialization
        """
        # input
        in_std = init_std or (self.emb_dim ** (-0.5))
        in_std = in_std / factor
        nn.init.trunc_normal_(self.W_in.weight, std=in_std, a=-3 * in_std, b=3 * in_std)

        # output gain
        min_rad, max_rad = 0.9, 0.999
        self.a.data.uniform_(min_rad**2 + 1e-8, max_rad**2 + 1e-8)
        self.a.data.log_().mul_(0.5)

        # convolution
        if self.conv_size is not None:
            conv_std = init_std or (self.conv_size ** (-0.5))
            nn.init.trunc_normal_(self.conv_weight, std=conv_std, a=-3 * conv_std, b=3 * conv_std)


# ------------------------------------------------------------------------------
# Hawk Blocks
# ------------------------------------------------------------------------------


class RGLRUBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        nb_heads: int,
        conv_size: int = None,
    ):
        super().__init__()

        # dimension
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        assert hidden_dim % nb_heads == 0, f"Hidden dim must be divisible by nb_heads: {hidden_dim} % {nb_heads} != 0"

        # matrices
        self.W_in = nn.Linear(emb_dim, 2 * hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, emb_dim, bias=False)

        # rglru
        self.rglru = RGLRU(
            emb_dim=hidden_dim,
            nb_heads=nb_heads,
            head_dim=hidden_dim // nb_heads,
            conv_size=conv_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, out2 = self.W_in(x).chunk(2, dim=-1)
        out2 = self.rglru(out2)
        out = F.silu(out1) * out2
        out = self.W_out(out)
        return out

    def init_weights(self, init_std: float, factor: float) -> None:
        # input
        in_std = init_std or (self.emb_dim ** (-0.5))
        in_std = in_std / factor
        nn.init.trunc_normal_(self.W_in.weight, std=in_std, a=-3 * in_std, b=3 * in_std)

        # output
        out_std = init_std or (self.hidden_dim ** (-0.5))
        out_std = out_std / factor
        nn.init.trunc_normal_(self.W_out.weight, std=out_std, a=-3 * in_std, b=3 * in_std)

        # convolution
        self.rglru.reset_parameters(init_std, factor)


class HawkBlock(nn.Module):
    def __init__(self, config: RNNBlockConfig):
        super().__init__()

        self.rlgru_block = RGLRUBlock(
            emb_dim=config.emb_dim,
            hidden_dim=int(config.hidden_dim),
            nb_heads=config.nb_heads,
            conv_size=config.conv_size,
        )
        self.ffn = FeedForward(emb_dim=config.emb_dim, hidden_dim=config.ffn_dim)
        self.rlgru_norm = RMSNorm(config.emb_dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.emb_dim, eps=config.norm_eps)

    def reset_parameters(self, init_std: float, factor: float) -> None:
        """Weight initialization"""
        self.rlgru_block.init_weights(init_std, factor)
        self.rlgru_norm.reset_parameters()
        self.ffn.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.rlgru_block(self.rlgru_norm(x))
        out = out + self.ffn(self.ffn_norm(out))
        return out

    def get_nb_flop(self, mode: str = "both", **kwargs) -> int:
        """
        TODO
        Number of flop to process a new token

        Parameters
        ----------
        mode:
            Whether to consider the forward, backward pass or both
        """
        mode_multiplier = dict(fwd=1, bwd=2.5, both=3.5)[mode]
        return 0 * mode_multiplier
