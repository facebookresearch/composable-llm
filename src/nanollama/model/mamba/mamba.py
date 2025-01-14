# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass

import torch
from torch import nn

from ..utils import RMSNorm
from .core_mamba import BaseMamba, BaseMambaArgs


@dataclass
class LMMambaArgs(BaseMambaArgs):
    seed: int = 42

    vocab_size: int = 0
    weight_tying: bool = False

    loss_reduction: str = "mean"


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
        self.loss_reduction = args.loss_reduction

        assert args.vocab_size > 0

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
        # tok_idx: Optional[torch.Tensor] = None,
        # cu_seqlens: Optional[torch.Tensor] = None,
        # ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        h = self.tok_embeddings(token_values)

        h = super().forward(h)
        #     h, tok_idx=tok_idx, cu_seqlens=cu_seqlens, ssm_impl=ssm_impl
        # )

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
