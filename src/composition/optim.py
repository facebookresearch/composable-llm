"""
Optimization Managers

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import math
from dataclasses import dataclass
from functools import partial

from torch import nn
from torch.optim import AdamW, Optimizer, lr_scheduler

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    # total number of update steps
    steps: int = -1
    # number of gradient accumulation before update
    grad_acc_steps: int = 1

    # AdamW parameters
    lr: float = 3e-4
    weight_decay: float = 0.1
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95

    # gradient clipping
    clip: float = 1.0

    # scheduler parameters
    scheduler: str = "cosine"
    warmup: int = 2000
    lr_min_ratio: float = 0.1


def init_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """
    Build optimizer and Scheduler
    """
    return AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        eps=config.epsilon,
        fused=True,  # Faster optim.step but can throw errors
    )


@dataclass
class OptimizerState:
    # nb of steps taken by the optimizer
    step: int
    # nb of accumulation steps done since last optimizer step
    acc_step: int


def init_optimizer_state():
    """
    Initialize the scheduler state
    """
    return OptimizerState(step=0, acc_step=0)


# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------


def init_scheduler(optimizer, config: OptimizerConfig) -> lr_scheduler.LambdaLR:
    """
    Initialize the scheduler state
    """
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        partial(
            lr_cosine,
            warmup=config.warmup,
            steps=config.steps,
            min_ratio=config.lr_min_ratio,
        ),
    )
    return scheduler


def lr_cosine(
    step: int,
    warmup: int,
    steps: int,
    min_ratio: float,
) -> float:
    """
    Cosine learning rate scheduler with warmup
    """
    if step < warmup:
        lr = float(step) / warmup
    elif step <= steps:
        s = float(step - warmup) / (steps - warmup)
        lr = min_ratio + 0.5 * (1 - min_ratio) * (math.cos(math.pi * s) + 1)
    else:
        lr = min_ratio
    return lr
