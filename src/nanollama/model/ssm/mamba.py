"""
Mamba model

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from dataclasses import dataclass, field
from typing import List, Optional  # noqa: UP035 (torch compile does not parse `list` keyword yet)

import torch
from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_bwd, _mamba_chunk_scan_combined_fwd
from torch import nn
from torch.autograd.function import FunctionCtx
from torch.nn import functional as F

from ..norm import RMSNorm
from .wrapper_causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)

# ------------------------------------------------------------------------------
# Compiled functions
# ------------------------------------------------------------------------------


@torch.compile(fullgraph=True)
def _compiled_mamba_chunk_scan_combined_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None,  # noqa: UP006
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        cu_seqlens=cu_seqlens,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )


@torch.compile(fullgraph=True)
def _compiled_mamba_chunk_scan_combined_bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    dfinal_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None,  # noqa: UP006
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _mamba_chunk_scan_combined_bwd(
        dout,
        x,
        dt,
        A,
        B,
        C,
        out,
        chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        dfinal_states=dfinal_states,
        seq_idx=seq_idx,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )


@torch.library.custom_op(
    "mamba_ssm::ssm_chunk_scan_combined_fwd",
    mutates_args=(),
    device_types="cuda",
)
def ssm_chunk_scan_combined_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None,  # noqa: UP006
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out, out_x, dt_out, dA_cumsum, states, final_states, *rest = _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        cu_seqlens=cu_seqlens,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )

    return (
        out,
        out_x if out_x is not None else out.new_empty(0),
        rest[0] if cu_seqlens is not None else out.new_empty(0),
    )


@ssm_chunk_scan_combined_fwd.register_fake
def _ssm_chunk_scan_combined_fwd_fake(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None,  # noqa: UP006
):
    _, _, n_heads, head_dim = x.shape
    return (
        torch.empty_like(x),
        torch.empty_like(x) if z is not None else None,
        x.new_empty((cu_seqlens.size(0) - 1, n_heads, head_dim, B.size(0))) if cu_seqlens is not None else None,
    )


@torch.library.custom_op(
    "mamba_ssm::ssm_chunk_scan_combined_bwd",
    mutates_args=(),
    device_types="cuda",
)
def ssm_chunk_scan_combined_bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None,  # noqa: UP006
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(
        dout,
        x,
        dt,
        A,
        B,
        C,
        out,
        chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        dfinal_states=None,
        seq_idx=seq_idx,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )
    return (
        dx,
        ddt,
        dA,
        dB,
        dC,
        dD if dD is not None else dx.new_empty(0),
        dz if dz is not None else dx.new_empty(0),
        ddt_bias if ddt_bias is not None else dx.new_empty(0),
        dinitial_states if dinitial_states is not None else dx.new_empty(0),
    )


@ssm_chunk_scan_combined_bwd.register_fake
def _ssm_chunk_scan_combined_bwd_fake(
    dout: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None,  # noqa: UP006
):
    return (
        torch.empty_like(x),
        torch.empty_like(dt),
        torch.empty_like(A),
        torch.empty_like(B),
        torch.empty_like(C),
        torch.empty_like(D) if D is not None else None,
        torch.empty_like(z) if z is not None else None,
        torch.empty_like(dt_bias) if dt_bias is not None else None,
        torch.empty_like(initial_states) if initial_states is not None else None,
    )


def ssm_chunk_scan_combined_setup_context(
    ctx: FunctionCtx,
    inputs: tuple,
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, dt_limit = inputs
    out, out_x, state_varlen = output

    ctx.save_for_backward(out if z is None else out_x, x, dt, A, B, C, D, z, dt_bias, initial_states, seq_idx)
    ctx.dt_softplus = dt_softplus
    ctx.chunk_size = chunk_size
    ctx.dt_limit = dt_limit


def ssm_chunk_scan_combined_bridge(
    ctx: FunctionCtx,
    dout: torch.Tensor,
    dout_x: torch.Tensor,
    dout_state_varlen: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    out, x, dt, A, B, C, D, z, dt_bias, initial_states, seq_idx = ctx.saved_tensors

    dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = ssm_chunk_scan_combined_bwd(
        dout,
        x,
        dt,
        A,
        B,
        C,
        out,
        ctx.chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        dt_softplus=ctx.dt_softplus,
        dt_limit=ctx.dt_limit,
    )

    return (
        dx,
        ddt,
        dA,
        dB,
        dC,
        None,
        dD if D is not None else None,
        dz if z is not None else None,
        ddt_bias if dt_bias is not None else None,
        dinitial_states if initial_states is not None else None,
        None,
        None,
        None,
        None,
    )


# Register custom autograd function
torch.library.register_autograd(
    "mamba_ssm::ssm_chunk_scan_combined_fwd",
    ssm_chunk_scan_combined_bridge,
    setup_context=ssm_chunk_scan_combined_setup_context,
)


def mamba_chunk_scan_combined(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor = None,
    z: torch.Tensor = None,
    dt_bias: torch.Tensor = None,
    initial_states: torch.Tensor = None,
    seq_idx: torch.Tensor = None,
    cu_seqlens: torch.Tensor = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
) -> tuple[torch.Tensor, torch.Tensor]:
    out = _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        cu_seqlens=cu_seqlens,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )
    varlen_states = out[-1]
    out = out[0]
    if cu_seqlens is not None:
        return out, varlen_states
    return out


# ------------------------------------------------------------------------------
# Base Model
# ------------------------------------------------------------------------------


@dataclass
class InitArgs:
    dt_max: float = 0.1
    dt_min: float = 0.001

    dt_init_floor: float = 1e-4

    A_init_min: float = 1
    A_init_max: float = 16


@dataclass
class BaseMambaArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8

    state_dim: int = 128
    n_groups: int = 1
    conv_size: Optional[int] = None

    dt_bias: bool = False
    D_has_head_dim: bool = False
    learnable_init_states: bool = False

    ssm_chunk_size: int = 256

    vocab_size: int = -1

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256
    """
    Enforces that the SwiGLU hidden layer size is a multiple
    of large power of 2.
    """

    norm_eps: float = 1e-5

    init_use_depth: bool = False
    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    init_args: InitArgs = field(default_factory=InitArgs)
    seed: int = 42


class SSM(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        state_dim: int,
        n_heads: int,
        n_groups: int,
        conv_size: Optional[int],
        dt_bias: bool,
        D_has_head_dim: Optional[bool],
        learnable_init_states: bool,
        dt_limit: tuple[float, float] = (0.0, float("inf")),
        # Fused kernel and sharding options
        chunk_size: int = 256,
    ):
        super().__init__()

        self.dim = dim

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert self.hidden_dim % n_heads == 0, (
            f"Hidden dim must be divisible by n_heads: {self.hidden_dim} % {n_heads} != 0"
        )

        self.state_dim = state_dim
        self.head_dim = self.hidden_dim // n_heads
        self.n_heads = n_heads
        self.n_groups = n_groups

        self.dt_limit = dt_limit

        self.chunk_size = chunk_size

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.hidden_dim + 2 * self.n_groups * self.state_dim + self.n_heads
        self.in_proj = nn.Linear(dim, d_in_proj, bias=False)

        self.conv_size = conv_size
        self.conv_dim = None
        if conv_size is not None:
            self.conv_dim = self.hidden_dim + 2 * self.n_groups * self.state_dim
            assert (self.conv_dim % 8 == 0) and (conv_size in [2, 3, 4]), (
                "Causal conv1d only supports conv_size in [2, 3, 4] and hidden_dim/head_dim % 8 == 0, "
                f"got {self.conv_dim} and {conv_size}"
            )
            self.conv_dim = self.hidden_dim + 2 * self.n_groups * self.state_dim
            self.conv_weight = nn.Parameter(torch.empty((self.conv_dim, conv_size)))

        self.learnable_init_states = learnable_init_states
        if learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(n_heads, self.head_dim, state_dim))

        self.dt_bias = None
        if dt_bias:
            self.dt_bias = nn.Parameter(torch.empty(n_heads))
        self.A_log = nn.Parameter(torch.empty(n_heads))

        if D_has_head_dim is None:
            self.D = None
        elif D_has_head_dim:
            self.D = nn.Parameter(torch.ones(n_heads, self.head_dim))
        else:
            self.D = nn.Parameter(torch.ones(n_heads))

        self.out_proj = nn.Linear(self.hidden_dim, self.dim, bias=False)

        self.ssm_norm = RMSNorm(self.hidden_dim, eps=1e-5)

        self.dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

    def forward(
        self,
        x: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        zxbcdt = self.in_proj(x)

        # Causal conv1d path
        if self.conv_size is not None:
            z, xBC, dt = torch.split(
                zxbcdt,
                [
                    self.hidden_dim,
                    self.hidden_dim + 2 * self.n_groups * self.state_dim,
                    self.n_heads,
                ],
                dim=-1,
            )

            # conv1d = log_stats(self.conv_weight, "conv1d.w")
            conv1d = self.conv_weight
            # xBC = log_stats(xBC, "conv1d.in")

            if ssm_impl == "ssm":  # For training
                if hasattr(self, "cache"):
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0),
                        cu_seqlens,
                        state_len=self.cache.conv_cache.shape[-1],
                    )
                    self.cache.conv_cache.copy_(conv_varlen_states)

                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=conv1d,
                    bias=None,
                    activation="silu",
                    seq_idx=tok_idx,
                ).transpose(1, 2)

            elif ssm_impl == "ssm_update":  # For generation only
                xBC = causal_conv1d_update(
                    x=xBC.squeeze(0),
                    conv_state=self.cache.conv_cache,
                    weight=self.conv_weight,
                    bias=None,
                    activation="silu",
                ).unsqueeze(0)

            else:
                raise NotImplementedError(f"SSM implementation {ssm_impl} not supported")

            # xBC = log_stats(xBC, "conv1d.out")

            x, B, C = torch.split(
                xBC,
                [
                    self.hidden_dim,
                    self.n_groups * self.state_dim,
                    self.n_groups * self.state_dim,
                ],
                dim=-1,
            )
        else:
            z, x, B, C, dt = torch.split(
                zxbcdt,
                [
                    self.hidden_dim,
                    self.hidden_dim,
                    self.n_groups * self.state_dim,
                    self.n_groups * self.state_dim,
                    self.n_heads,
                ],
                dim=-1,
            )

        initial_states = None
        if self.learnable_init_states:
            initial_states = self.init_states.expand(bsz, -1, -1, -1)

        x = x.view(bsz, seq_len, self.n_heads, self.head_dim)  # (bsz, seq_len, n_heads, head_dim)

        # A_log = log_stats(self.A_log, "A_log")
        A_log = self.A_log
        A = -torch.exp(A_log.float())
        B = B.view(bsz, seq_len, self.n_groups, self.state_dim)  # (bsz, seq_len, ngroups, state_dim)
        C = C.view(bsz, seq_len, self.n_groups, self.state_dim)  # (bsz, seq_len, ngroups, state_dim)

        # A, B, C = log_stats(A, "A"), log_stats(B, "B"), log_stats(C, "C")  # For probing

        if ssm_impl == "ssm":  # For training
            y = mamba_chunk_scan_combined(
                x,
                dt,
                A,
                B,
                C,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=tok_idx,
                cu_seqlens=cu_seqlens,
                initial_states=initial_states,
                **self.dt_limit_kwargs,
            )  # (bsz, seq_len, n_heads, head_dim)

            if hasattr(self, "cache"):
                y, varlen_states = y
                self.cache.state_cache.copy_(varlen_states)

        elif ssm_impl == "ssm_update":  # For generation only
            x = x.squeeze(0)
            A = A[..., None, None].expand(self.n_heads, self.head_dim, self.state_dim)
            dt = dt.permute(1, 2, 0).expand(seq_len, self.n_heads, self.head_dim)
            D = self.D
            if D is not None and D.dim() == 1:
                D = D.unsqueeze(1).expand(self.n_heads, self.head_dim)
            B, C = B.squeeze(0), C.squeeze(0)
            y = selective_state_update(
                self.cache.state_cache,
                x,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=(
                    torch.zeros(self.n_heads, self.head_dim).to(x)
                    if self.dt_bias is None
                    else self.dt_bias.unsqueeze(1).expand(self.n_heads, self.head_dim)
                ),
                dt_softplus=True,
            ).unsqueeze(0)

        else:
            raise NotImplementedError(f"SSM implementation {ssm_impl} not supported")

        y = y.view(bsz, seq_len, self.hidden_dim)

        # Could be different activation function, including None, Mamba people post_norm here also
        # (sometime norm(z)*y or norm(z*y))
        # y = log_stats(y, "ssm_out")
        # y = self.ssm_norm(y)
        y = self.ssm_norm(y * F.silu(z))

        out = self.out_proj(y)

        return out

    def reset_parameters(self, init_std: float, factor: float, init_args: InitArgs) -> None:
        # Linear layers
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std / factor
        out_init_std = out_init_std / factor

        nn.init.trunc_normal_(
            self.in_proj.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )

        nn.init.trunc_normal_(
            self.out_proj.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )

        # SSM
        if self.dt_bias is not None:
            self.dt_bias.uniform_(init_args.dt_min, init_args.dt_max)
            self.dt_bias.clamp_(min=init_args.dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            self.dt_bias.data = self.dt_bias.data + torch.log(-torch.expm1(-self.dt_bias.data))

        if self.conv_size is not None:
            conv_std = init_std or (self.conv_size ** (-0.5))
            nn.init.trunc_normal_(
                self.conv_weight,
                mean=0.0,
                std=conv_std,
                a=-3 * conv_std,
                b=3 * conv_std,
            )

        if self.learnable_init_states:
            self.init_states.zero_()

        # Initialize A
        self.A_log.uniform_(init_args.A_init_min, init_args.A_init_max)
        self.A_log.log_()

        self.D.data.fill_(1.0)


class MambaBlock(nn.Module):
    def __init__(self, args: BaseMambaArgs):
        super().__init__()

        self.ssm_norm = RMSNorm(args.dim, args.norm_eps)
        self.ssm = SSM(
            dim=args.dim,
            hidden_dim=3 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            state_dim=args.state_dim,
            n_heads=args.n_heads,
            n_groups=args.n_groups,
            conv_size=args.conv_size,
            dt_bias=args.dt_bias,
            D_has_head_dim=args.D_has_head_dim,
            learnable_init_states=args.learnable_init_states,
            chunk_size=args.ssm_chunk_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        x = x + self.ssm(self.ssm_norm(x), tok_idx=tok_idx, cu_seqlens=cu_seqlens, ssm_impl=ssm_impl)
        return x

    def init_weights(self, init_std: float = None, factor: float = 1.0, init_args: InitArgs = None) -> None:
        if init_args is None:
            init_args = InitArgs()
        self.ssm_norm.reset_parameters()
        self.ssm.reset_parameters(init_std, factor, init_args)


class BaseMamba(nn.Module):
    def __init__(self, args: BaseMambaArgs):
        super().__init__()
        self.model_dim = args.dim
        self.init_base_std = args.init_base_std

        # self.init_args = args.init_args
        # self.init_std_factor = InitStdFactor(args.init_std_factor)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(MambaBlock(args))

    def forward(
        self,
        h: torch.Tensor,
        tok_idx: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, tok_idx=tok_idx, cu_seqlens=cu_seqlens, ssm_impl=ssm_impl)
        return h

    def reset_parameters(self) -> None:
        pass

    def init_weights(self) -> None:
        self.reset_parameters()
        for _depth, layer in enumerate(self.layers):
            factor = 1
            # factor = {
            #     InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
            #     InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
            #     InitStdFactor.DIM_RATIO: self.model_dim / 4096,
            #     InitStdFactor.DISABLED: 1.0,
            # }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)


# ------------------------------------------------------------------------------
# Language Model
# ------------------------------------------------------------------------------


@dataclass
class LMMambaArgs(BaseMambaArgs):
    seed: int = 42
    vocab_size: int = 0
    weight_tying: bool = False

    def __check_init__(self):
        assert self.vocab_size > 0, "vocab_size must be greater than 0"


def attention_flops_per_token(n_layers: int, seq_len: int, dim: int, causal: bool) -> float:
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(num_non_embed_params: int, n_layers: int, dim: int, seq_len: int) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(n_layers, seq_len, dim, True)


class StateCache(nn.Module):
    def __init__(
        self,
        bsz: int,
        n_heads: int,
        head_dim: int,
        state_dim: int,
        conv_size: int,
        conv_dim: int,
        dtype: str,
        device: str,
    ):
        super().__init__()
        state_shape = (bsz, n_heads, head_dim, state_dim)
        if conv_size is None:
            conv_shape = (0,)
        else:
            conv_shape = (bsz, conv_dim, conv_size)

        self.register_buffer(
            "conv_cache",
            torch.zeros(conv_shape, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "state_cache",
            torch.zeros(state_shape, dtype=dtype, device=device),
            persistent=False,
        )

    def reset(self) -> None:
        self.conv_cache.zero_()
        self.state_cache.zero_()


class LMMamba(BaseMamba):
    def __init__(self, args: LMMambaArgs) -> None:
        super().__init__(args)
        self.weight_tying = args.weight_tying

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )

        if args.weight_tying:
            self.output.weight = self.tok_embeddings.weight

        self.init_weights()

    def forward(
        self,
        token_values: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        h = self.tok_embeddings(token_values)

        h = super().forward(h, tok_idx=tok_idx, cu_seqlens=cu_seqlens, ssm_impl=ssm_impl)
        logits = self.output(self.norm(h))
        return logits

    def reset_parameters(self, init_std: float = None) -> None:
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.model_dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

    @torch.inference_mode()
    def init_weights(self) -> None:
        super().init_weights()

    def get_nb_flop(self) -> int:
        # TODO
        return 0


# Optional policy for activation checkpointing. With None, we stick to the default
# (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops() -> set:
    return {
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_mm.default,
        torch.ops.c10d_functional.reduce_scatter_tensor.default,
        torch.ops.mamba_ssm.ssm_chunk_scan_combined_fwd.default,
    }


if __name__ == "__main__":
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as mamba_chunk_scan_combined_ref

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    x = torch.randn(2, 3, 4, 5).cuda()
    dt = torch.randn(2, 3, 4).cuda()
    A = torch.randn(4).cuda()
    B = torch.randn(2, 3, 4, 5).cuda()
    C = torch.randn(2, 3, 4, 5).cuda()
    chunk_size = 2
    D = torch.randn(4, 5).cuda()
    z = torch.randn(2, 3, 4, 5).cuda()
    dt_bias = torch.randn(4).cuda()

    out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias)

    print(out.min(), out.max(), out.mean(), out.std())

    compiled_mamba_chunk_scan_combined = torch.compile(mamba_chunk_scan_combined)
    out = compiled_mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias)

    print(out.min(), out.max(), out.mean(), out.std())

    out_ref = mamba_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias)

    print(out_ref.min(), out_ref.max(), out_ref.mean(), out_ref.std())
