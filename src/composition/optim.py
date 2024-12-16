import logging
import math
from dataclasses import dataclass
from functools import partial

from torch import nn
from torch.optim import AdamW, lr_scheduler

logger = logging.getLogger()


@dataclass
class OptimizerConfig:
    # Optimizer parameters
    grad_acc_steps: int = 1
    steps: int = 1000

    # AdamW parameters
    lr: float = 3e-4
    weight_decay: float = 0.1
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95
    clip: float = 1.0

    # scheduler parameters
    scheduler: str = "cosine"
    warmup: int = 2000
    lr_min_ratio: float = 0.1

    exp_factor: float = 0.5


def lr_cosine(
    step: int,
    warmup: int,
    n_steps: int,
    min_ratio: float,
) -> float:
    if step < warmup:
        lr = float(step) / warmup
    elif step <= n_steps:
        s = float(step - warmup) / (n_steps - warmup)
        lr = min_ratio + 0.5 * (1 - min_ratio) * (math.cos(math.pi * s) + 1)
    else:
        lr = min_ratio
    return lr


def build_optimizer(model: nn.Module, config: OptimizerConfig, n_steps: int):
    logger.info("Starting build of optimizer...")
    # optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        eps=config.epsilon,
        fused=True,  # Faster optim.step but can throw errors
    )

    # scheduler
    if config.scheduler == "cosine":
        lr_fn = partial(
            lr_cosine,
            warmup=config.warmup,
            n_steps=n_steps,
            min_ratio=config.lr_min_ratio,
        )
    else:
        raise NotImplementedError(f"Unknown scheduler: {config.scheduler}")
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_fn)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler
