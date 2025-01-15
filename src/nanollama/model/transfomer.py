"""
Transformer model.

Notes
-----
Comments abbreviations:
    B: batch size
    S: sequence length
    D: embedding dimension
    H: number of heads

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocklm import BlockLanguageModel, BlockLanguageModelConfig
from .feedforward import FeedForward
from .norm import RMSNorm

# -----------------------------------------------------------------------------
# Attention Layer
# -----------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """
    Self-attention layer.

    Parameters
    ----------
    seq_len: int
        maximum sequence length
    emb_dim: int
        embedding dimensionality of the input
    nb_heads: int
        number of attention heads (should divide emb_dim)
    rope_theta: float
        rotational positional encoding parameter
    """

    def __init__(
        self,
        seq_len: int,
        emb_dim: int,
        nb_heads: int,
        rope_theta: float,
    ):
        super().__init__()

        # dimensions
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.nb_heads = nb_heads
        self.head_dim = self.emb_dim // self.nb_heads
        assert self.emb_dim % self.nb_heads == 0, "embedding dimension must be divisible by number of heads"

        # matrices
        self.qkv_mat = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=False)
        self.output = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        # rotational positional encoding
        self.theta = rope_theta
        self.register_buffer("rope_modulator", self._get_rope_modulator(self.seq_len, self.head_dim, self.theta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Self attention

        Parameters
        ----------
        x: torch.Tensor (B, S, D)
            input sequence

        Returns
        -------
        z: torch.Tensor (B, S, D)
            output sequence
        """
        # dimensions
        bsz, seq_len, _ = x.size()

        # Query, key, value: (B, S, D) @ (D, 3 * D) -> (B, S, 3 * D) -> (B, S, D) * 3
        q, k, v = self.qkv_mat(x).chunk(3, dim=-1)

        # reformating: (B, S, D) -> (B, S, H, D / H) -> (B, H, S, D / H)
        q, k, v = map(lambda t: t.view(bsz, seq_len, self.nb_heads, self.head_dim).transpose(1, 2), (q, k, v))

        # rope formatting
        q, k = map(lambda t: self._rope_view(t), (q, k))

        # KV cache implementation at inference time
        if hasattr(self, "kv_cache"):
            k, v = self.kv_cache.update(k, v)

        # Flash attention implementation
        # ... -> (B, H, S, D / H)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # reformating: (B, H, S, D / H) -> (B, S, H, D / H) -> (B, S, D)
        z = z.transpose(1, 2).reshape(bsz, seq_len, self.emb_dim)

        # output layer: (B, L, D) @ (D, D) -> (N, L, D)
        z = self.output(z)
        return z

    def reset_parameters(self, init_std: float, factor: float) -> None:
        """
        Weight initialization
        """
        init_std = init_std or (self.emb_dim ** (-0.5))
        nn.init.trunc_normal_(
            self.qkv_mat.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )

    @staticmethod
    def _get_rope_modulator(seq_len: int, dim: int, theta: float) -> torch.Tensor:
        """
        Returns the rope modulator for the attention mechanism.

        Parameters
        ----------
        seq_len:
            sequence length
        dim:
            embedding dimension
        theta:
            rope angle parameter

        Returns
        -------
        rope_modulator:
            tensor of shape (seq_len, dim, 2) whose (t, k) element is
            .. math::
                \\cos(\\frac{2 \\pi t}{\\theta^{(2k / d)}}),
                \\sin(\\frac{2 \\pi t}{\\theta^{(2k / d)}})
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim - 1, 2) / dim))
        t = torch.arange(seq_len) * 2 * math.pi
        angles = torch.outer(t, freqs)
        cos, sin = angles.cos(), angles.sin()
        return torch.stack((cos, sin), dim=-1)

    def _rope_view(self, qk: torch.Tensor) -> torch.Tensor:
        """
        Recast tensor to complex numbers and apply rotational position filter.
        """
        B, H, S, dim = qk.size()
        assert S <= self.rope_modulator.size(0), "sequence length is too long for rope attention"

        rm = self.rope_modulator[:S].view(1, 1, S, dim // 2, 2)
        qk = qk.reshape(B, H, S, dim // 2, 2)

        # (x1 * cos - x2 * sin, x2 * cos + x1 * sin)
        # out = ((qk[..., 0] + qk[..., 1] * 1j) * (rm[..., 0] + rm[..., 1] * 1j))
        # out = torch.view_as_real(out)
        out = (qk[..., 0] * rm[..., 0] - qk[..., 1] * rm[..., 1], qk[..., 0] * rm[..., 1] + qk[..., 1] * rm[..., 0])
        out = torch.stack((out[0], out[1]), dim=-1)

        return out.type_as(qk).view((B, H, S, dim))


# -----------------------------------------------------------------------------
# Configuration Class
# -----------------------------------------------------------------------------


@dataclass
class TransformerBlockConfig:
    seq_len: int = 0
    emb_dim: int = 0

    # Transformer block parameters
    nb_heads: int = 0
    rope_theta: float = 10_000
    hidden_dim: int = None
    norm_eps: float = 1e-5

    # Transformer parameters
    init_std: float = None

    def __post_init__(self):
        # hidden feed-forward dimension
        if self.hidden_dim is None:
            self.hidden_dim = 4 * self.emb_dim

    def __check_init__(self):
        """Check validity of arguments."""
        assert self.seq_len, "sequence length should be specified"
        assert self.emb_dim, "embedding dimension should be specified"
        assert self.nb_heads, "number of heads should be specified"
        assert self.emb_dim // (2 * self.nb_heads), "embedding dimension should be divisible by 2 * number of heads"


# -------------------------------------------------------------------------------
# Transformer Block
# -------------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Transformer block.

    Parameters
    ----------
    config: configuration class containing arguments for SelfAttention and FeedForward
    """

    def __init__(self, config: TransformerBlockConfig):
        super().__init__()

        self.attn = SelfAttention(
            seq_len=config.seq_len, emb_dim=config.emb_dim, nb_heads=config.nb_heads, rope_theta=config.rope_theta
        )
        self.ffn = FeedForward(emb_dim=config.emb_dim, hidden_dim=config.hidden_dim)
        self.attn_norm = RMSNorm(config.emb_dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.emb_dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.attn(self.attn_norm(x))
        out = out + self.ffn(self.ffn_norm(out))
        return out

    def get_nb_flop(self, mode: str = "both", seq_len: int = None) -> int:
        """
        TODO
        Number of flop to process a new token

        Parameters
        ----------
        seq_len:
            Sequence length.
        mode:
            Whether to consider the forward, backward pass or both
        """
        mode_multiplier = dict(fwd=1, bwd=2.5, both=3.5)[mode]
        return 0 * mode_multiplier

    def reset_parameters(self, init_std: float, factor: float) -> None:
        """
        Weight initialization
        """
        self.attn.reset_parameters(init_std, factor)
        self.attn_norm.reset_parameters()
        self.ffn.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


# -----------------------------------------------------------------------------
# Transformer Architecture
# -----------------------------------------------------------------------------


@dataclass
class TransformerConfig(BlockLanguageModelConfig):
    block: TransformerBlockConfig = field(default_factory=TransformerBlockConfig)

    def __post_init__(self):
        """Inherit parameters from the block model configuration."""
        for attr in ["emb_dim", "norm_eps", "init_std"]:
            setattr(self.block, attr, getattr(self, attr))


class Transformer(BlockLanguageModel):
    """
    Decoder only transformer.

    Parameters
    ----------
    config: configuration class with
        n_layer: int
            number of transformer blocks
        weight_tying: bool
            whether to use weight tying between the token embedding and the output layer
        output_dropout: float
            dropout probability for the embeddings layer
        norm: str
            type of normalization layer. Options are "layer", "rms"
        norm_bias: bool
            whether to use bias in the normalization layer
        and the parameter to initialize TransformerBlock and Embedding

    See Also
    --------
    Embedding
    TransformerBlock
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config, block=TransformerBlock)
