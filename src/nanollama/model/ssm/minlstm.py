"""
MinLSTM model

https://arxiv.org/abs/2410.01201

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..norm import RMSNorm
from .utils_rnn import RNNBlockConfig, conv1d, scan


class LSTM(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,  # h_t dim (state expansion)
        nb_heads: int,
        conv_size: int = None,
    ):
        super().__init__()

        # dimension
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.nb_heads = nb_heads
        self.head_dim = hidden_dim // nb_heads
        assert hidden_dim % nb_heads == 0, f"Hidden dim must be divisible by nb_heads: {hidden_dim} % {nb_heads} != 0"

        # matrices
        self.W_in = nn.Linear(emb_dim, 4 * hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, emb_dim, bias=False)

        self.conv_size = conv_size
        if conv_size is not None:
            assert ((self.hidden_dim) % 8 == 0) and (conv_size in [2, 3, 4]), (
                "Causal conv1d only supports conv_size in [2, 3, 4] and hidden_dim % 8 == 0, "
                f"got {self.hidden_dim} and {conv_size}"
            )
            self.conv_weight = nn.Parameter(torch.empty((2 * self.hidden_dim, conv_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        tmp, fi = self.W_in(x).chunk(2, dim=-1)
        out1, h = tmp.chunk(2, dim=-1)

        fi = fi.transpose(1, 2)
        h = h.transpose(1, 2)

        if self.conv_size is not None:
            # conv1d_w = log_stats(self.conv_weight, "conv1d.w")
            fi = conv1d(x=fi, conv_weight=self.conv_weight, impl="parallel")

        fi = fi.reshape(bsz * self.nb_heads, 2 * self.head_dim, seq_len)
        h = h.reshape(bsz * self.nb_heads, self.head_dim, seq_len)

        f, i = fi.chunk(2, dim=1)
        f, i = F.sigmoid(f), F.sigmoid(i)
        denom = 1 / (f + i + 1e-4)

        h = scan(
            a=(f * denom),
            b=(h * i * denom),
            impl="parallel",
        )

        out2 = h.view(bsz, self.hidden_dim, seq_len).transpose(1, 2)
        # out2 = log_stats(out2, "hidden_state")

        out = F.silu(out1) * out2
        out = self.W_out(out)
        return out

    def reset_parameters(self, init_std: float, factor: float) -> None:
        """
        Weight initialization
        """
        # input
        in_std = init_std or (self.emb_dim ** (-0.5))
        in_std = in_std / factor
        nn.init.trunc_normal_(self.W_in.weight, std=in_std, a=-3 * in_std, b=3 * in_std)

        # output
        out_std = init_std or (self.hidden_dim ** (-0.5))
        out_std = out_std / factor
        nn.init.trunc_normal_(self.W_out.weight, std=out_std, a=-3 * in_std, b=3 * in_std)

        # convolution
        if self.conv_size is not None:
            conv_std = init_std or (self.conv_size ** (-0.5))
            nn.init.trunc_normal_(self.conv_weight, std=conv_std, a=-3 * conv_std, b=3 * conv_std)


class LSTMBlock(nn.Module):
    def __init__(self, config: RNNBlockConfig):
        super().__init__()

        self.lstm_norm = RMSNorm(config.emb_dim, eps=config.norm_eps)
        self.lstm = LSTM(
            emb_dim=config.emb_dim,
            hidden_dim=config.hidden_dim,
            nb_heads=config.nb_heads,
            conv_size=config.conv_size,
        )

    def reset_parameters(self, init_std: float, factor: float) -> None:
        """Weight initialization"""
        self.lstm.reset_parameters(init_std, factor)
        self.lstm_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.lstm(self.lstm_norm(x))
        return x

    def get_nb_flop(self, **kwargs) -> int:
        """
        TODO
        Number of flop to process a new token

        Parameters
        ----------
        mode:
            Whether to consider the forward, backward pass or both
        """
        return 0
