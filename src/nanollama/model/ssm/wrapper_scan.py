"""
Wrapper around `scan` layer from the `accelerated_scan` library.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import torch

# from accelerated_scan.triton import scan as triton_scan
from torch.autograd.function import FunctionCtx

try:
    from accelerated_scan.warp import warpscan_backward, warpscan_forward
except ImportError as e:
    print(e)
    print(
        "Could not import FastRNN. This is likely due to the lack of installation of the ssm dependencies."
        "You may install them with `pip install .[ssm]."
    )
except OSError as e:
    print(e)
    print("Could not import FastRNN. This is likely due to missing OS environment variables.")


@torch.library.custom_op(
    "scan::scan_fwd",
    mutates_args=(),
    device_types="cuda",
)
def scan_fwd(
    gates: torch.Tensor,
    tokens: torch.Tensor,
    reverse: bool = False,
) -> torch.Tensor:
    B, dim, seq_len = gates.shape
    assert tokens.shape == (B, dim, seq_len)
    assert gates.is_contiguous()
    assert tokens.is_contiguous()

    output = torch.zeros_like(tokens)
    warpscan_forward(gates, tokens, output, reverse)
    return output


@scan_fwd.register_fake
def _scan_fwd_fake(gates: torch.Tensor, tokens: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    return torch.empty_like(tokens)


@torch.library.custom_op(
    "accelerated_scan::scan_bwd",
    mutates_args=(),
    device_types="cuda",
)
def scan_bwd(
    dout: torch.Tensor,
    states: torch.Tensor,
    gates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dout = dout.contiguous()
    assert states.is_contiguous()
    assert gates.is_contiguous()

    d_gates = torch.empty_like(gates)
    d_tokens = torch.empty_like(gates)
    warpscan_backward(gates, states, dout, d_gates, d_tokens)

    return d_gates, d_tokens


@scan_bwd.register_fake
def _scan_bwd_fake(dout: torch.Tensor, states: torch.Tensor, gates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(gates), torch.empty_like(gates)


def scan_setup_context(ctx: FunctionCtx, inputs: tuple[torch.Tensor, torch.Tensor, bool], output: torch.Tensor) -> None:
    gates, tokens, reverse = inputs
    ctx.save_for_backward(gates, output)


def scan_bwd_bridge(ctx: FunctionCtx, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
    gates, states = ctx.saved_tensors
    d_gates, d_tokens = scan_bwd(dout, states, gates)

    return d_gates, d_tokens, None


torch.library.register_autograd(
    "scan::scan_fwd",
    scan_bwd_bridge,
    setup_context=scan_setup_context,
)


def scan(gates: torch.Tensor, tokens: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    return scan_fwd(gates, tokens, reverse)
