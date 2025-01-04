"""
Evaluation Script

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from dataclasses import dataclass, field
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ...nanollama.data.hdf5 import DataConfig, FileEvaluator
from ...nanollama.distributed import get_local_rank

logger = getLogger("nanollama")


@dataclass
class EvaluationConfig:
    period: int = 1

    data: DataConfig = field(default_factory=DataConfig)


def loss_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = preds.size(-1)
    return F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1))


@torch.no_grad()
def launch_evaluation(config: EvaluationConfig, model: nn.Module) -> None:
    loss = 0
    nb_chunks = 0
    with FileEvaluator(config.data) as loader:
        for batch, _ in loader:
            batch = batch.to(get_local_rank())
            X_batch = batch[:, :-1]
            y_batch = batch[:, 1:]

            # evaluate
            preds = model(X_batch)
            loss += loss_func(preds, y_batch)
            nb_chunks += 1

        if isinstance(model, DDP):
            logger.info(f"Model running on {model.module.embeddings.weight.device}")
            logger.info(f"Batch on {batch.device}")
        else:
            logger.info(f"Model running on {model.embeddings.weight.device}")
            logger.info(f"Batch on {batch.device}")

    loss /= nb_chunks
    return loss.item()
