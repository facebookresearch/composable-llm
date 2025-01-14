"""
MinLSTM model

https://arxiv.org/abs/2410.01201

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..norm import RMSNorm
from .rnn_utils import conv1d, scan

# ------------------------------------------------------------------------------
# Base Model
# ------------------------------------------------------------------------------


@dataclass
class BaseMinLSTMArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 1

    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    conv_size: Optional[int] = None

    norm_eps: float = 1e-5

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"


class LSTM(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,  # h_t dim (state expansion)
        n_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        conv_size: Optional[int] = None,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % n_heads == 0, f"Hidden dim must be divisible by n_heads: {hidden_dim} % {n_heads} != 0"

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.conv_size = conv_size
        if conv_size is not None:
            assert ((self.hidden_dim) % 8 == 0) and (conv_size in [2, 3, 4]), (
                "Causal conv1d only supports conv_size in [2, 3, 4] and hidden_dim % 8 == 0, "
                f"got {self.hidden_dim} and {conv_size}"
            )
            self.conv_dim = 2 * self.hidden_dim
            self.conv_weight = nn.Parameter(torch.empty((self.conv_dim, conv_size)))

        self.w = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        self.wfi = nn.Linear(
            dim,
            2 * hidden_dim,
            bias=False,
        )

        self.wh_tilde = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, impl: str = "parallel") -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        w0 = self.w(x.view_as(x))

        fi = self.wfi(x.view_as(x)).transpose(1, 2)
        h_tilde = self.wh_tilde(x.view_as(x)).transpose(1, 2)

        if self.conv_size is not None:
            # conv1d_w = log_stats(self.conv_weight, "conv1d.w")
            conv1d_w = self.conv_weight
            fi = conv1d(
                x=fi,
                conv_weight=conv1d_w,
                cu_seqlens=cu_seqlens,
                impl=impl,
                cache=self.cache.conv_cache if hasattr(self, "cache") else None,
            )

        fi = fi.reshape(bsz * self.n_heads, 2 * self.head_dim, seq_len)
        h_tilde = h_tilde.reshape(bsz * self.n_heads, self.head_dim, seq_len)

        f, i = fi.chunk(2, dim=1)
        f, i = F.sigmoid(f), F.sigmoid(i)
        denom = 1 / (f + i + 1e-4)

        h = scan(
            a=(f * denom),
            b=(h_tilde * i * denom),
            cu_seqlens=cu_seqlens,
            impl=impl,
            cache=self.cache.state_cache if hasattr(self, "cache") else None,
        )

        h = h.view(bsz, self.hidden_dim, seq_len).transpose(1, 2)
        # h = log_stats(h, "hidden_state")

        h = h * F.silu(w0)

        out = self.wo(h)

        return out

    def reset_parameters(self, init_std: float, factor: float) -> None:
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std / factor
        out_init_std = out_init_std / factor

        for w in [self.wfi, self.wh_tilde]:
            nn.init.trunc_normal_(w.weight, std=in_init_std, a=-3 * in_init_std, b=3 * in_init_std)

        nn.init.trunc_normal_(self.wo.weight, std=out_init_std, a=-3 * in_init_std, b=3 * in_init_std)

        if self.conv_size is not None:
            conv_std = init_std or (self.conv_size ** (-0.5))
            nn.init.trunc_normal_(self.conv_weight, mean=0.0, std=conv_std, a=-3 * conv_std, b=3 * conv_std)


class LSTMBlock(nn.Module):
    def __init__(self, args: BaseMinLSTMArgs):
        super().__init__()

        self.lstm_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.lstm = LSTM(
            dim=args.dim,
            hidden_dim=3 * args.dim,
            n_heads=args.n_heads,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            conv_size=args.conv_size,
        )

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, impl: str = "parallel") -> torch.Tensor:
        x = x + self.lstm(self.lstm_norm(x), cu_seqlens=cu_seqlens, impl=impl)
        return x

    def init_weights(self, init_std: float, factor: float) -> None:
        self.lstm.reset_parameters(init_std, factor)
        self.lstm_norm.reset_parameters()


class BaseMinLSTM(nn.Module):
    def __init__(self, args: BaseMinLSTMArgs):
        super().__init__()

        self.dim = args.dim
        self.init_base_std = args.init_base_std
        # self.init_std_factor = InitStdFactor(args.init_std_factor)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(LSTMBlock(args))

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, impl: str = "parallel") -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cu_seqlens=cu_seqlens, impl=impl)
        return x

    def reset_parameters(self) -> None:
        pass

    def init_weights(self) -> None:
        self.reset_parameters()
        for _depth, layer in enumerate(self.layers):
            factor = 1
            layer.init_weights(self.init_base_std, factor)


# ------------------------------------------------------------------------------
# Language Model
# ------------------------------------------------------------------------------


@dataclass
class LMMinLSTMArgs(BaseMinLSTMArgs):
    seed: int = 42
    vocab_size: int = 0
    weight_tying: bool = False

    def __check_init__(self):
        assert self.vocab_size > 0, "Vocab size must be greater than 0"


class LMMinLSTM(BaseMinLSTM):
    def __init__(self, args: LMMinLSTMArgs) -> None:
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.seed = args.seed

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
        cu_seqlens: Optional[int] = None,
        impl: str = "parallel",
    ) -> torch.Tensor:
        h = self.tok_embeddings(token_values)

        h = super().forward(h, cu_seqlens=cu_seqlens, impl=impl)

        logits = self.output(self.norm(h))
        return logits

    def reset_parameters(self, init_std: int = None) -> None:
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
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

    def _get_no_recompute_ops(self):
        return get_no_recompute_ops()

    def get_nb_flop(self, **kwargs) -> int:
        # TODO
        return 0


def get_no_recompute_ops() -> None:
    return {
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_mm.default,
        torch.ops.c10d_functional.reduce_scatter_tensor.default,
        torch.ops.scan.scan_fwd.default,
    }
