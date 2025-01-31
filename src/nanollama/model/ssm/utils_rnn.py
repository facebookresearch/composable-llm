"""
RNN utils

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from dataclasses import dataclass
from typing import Any

import torch

# from accelerated_scan.triton import scan as triton_scan
try:
    from accelerated_scan.ref import scan as ref_scan
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError as e:
    print(e)
    print(
        "Could not import FastRNN. This is likely due to the lack of installation of the ssm dependencies."
        "You may install them with `pip install .[ssm]."
    )
except OSError as e:
    print(e)
    print("Could not import FastRNN. This is likely due to missing OS environment variables.")

from .wrapper_causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from .wrapper_scan import scan as accelerated_scan

# ------------------------------------------------------------------------------
# Configuration class
# ------------------------------------------------------------------------------


@dataclass
class RNNBlockConfig:
    emb_dim: int = 0

    # Block parameters
    nb_heads: int = 0
    conv_size: int = None
    hidden_dim: int = 0
    ffn_dim: int = 0  # for Hawk
    norm_eps: float = 1e-5

    # Model parameters
    init_std: float = None

    def __check_init__(self):
        """Check validity of arguments."""
        assert self.emb_dim, "embedding dimension should be specified"
        assert self.nb_heads, "number of heads should be specified"


# ------------------------------------------------------------------------------
# 1D Convolution
# ------------------------------------------------------------------------------


def conv1d(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    cu_seqlens: torch.Tensor = None,
    impl: str = "parallel",
    cache: Any = None,
) -> torch.Tensor:
    if impl == "parallel":
        if cache is not None:
            conv_varlen_states = causal_conv1d_varlen_states(
                x.squeeze(0).transpose(0, 1), cu_seqlens, state_len=cache.shape[-1]
            )
            cache.copy_(conv_varlen_states)

        x = causal_conv1d_fn(
            x=x,
            weight=conv_weight,
            bias=None,
            activation="silu",
        )

    elif impl == "sequential":
        x = (
            causal_conv1d_update(
                x=x.squeeze(0).transpose(0, 1),
                conv_state=cache,
                weight=conv_weight,
                bias=None,
                activation="silu",
            )
            .transpose(0, 1)
            .unsqueeze(0)
        )

    return x


# ------------------------------------------------------------------------------
# Linear Scan
# ------------------------------------------------------------------------------


def _prepare_for_cache(
    a: torch.Tensor, b: torch.Tensor, cu_seqlen: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function reset the hidden state at the beginning of each sequence in the batch,
    so that the hidden state is not carried over between sequences.
    """
    num_seq = cu_seqlen.size(0) - 1
    pow_2_seqlen = max(2 ** (seq_len + num_seq - 2).bit_length(), 32)
    _a = torch.zeros(*a.shape[:2], pow_2_seqlen, device=a.device, dtype=a.dtype)
    _b = torch.zeros(*b.shape[:2], pow_2_seqlen, device=b.device, dtype=b.dtype)

    mask = torch.zeros(pow_2_seqlen, dtype=torch.bool, device=a.device)
    offsets = torch.arange(0, num_seq, device=a.device)
    mask[cu_seqlen[1:-1] + offsets[:-1]] = True
    mask[(cu_seqlen[-1] + offsets[-1]) :] = True
    mask = (~mask).nonzero().flatten()

    for tensor_with_reset, tensor in zip((_a, _b), (a, b)):
        tensor_with_reset[..., mask] = tensor

    return _a, _b, cu_seqlen[1:] + offsets - 1, mask


def sequential_step(states: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * states + b


def scan(
    a: torch.Tensor, b: torch.Tensor, cu_seqlens: torch.Tensor = None, impl: str = "parallel", cache: Any = None
) -> torch.Tensor:
    if impl == "parallel":
        if cache is not None:
            # For accelerated_scan give me illegal memory access error when seqlen > ~2048
            a, b, last_state_idx, mask = _prepare_for_cache(a, b, cu_seqlens, a.size(2))

            h = ref_scan(
                a.contiguous(),
                b.contiguous(),
            )

            cache.copy_(h[:, :, last_state_idx])
            h = h[:, :, mask]
        else:
            h = accelerated_scan(
                a.contiguous(),
                b.contiguous(),
            )

    elif impl == "sequential":
        h = sequential_step(cache, a, b)
        cache.copy_(h)

    return h
