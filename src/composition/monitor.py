import logging
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    # logging
    name: str = "difformer"
    dump_dir: str = ""

    # reproducibility
    seed: int = 42

    # garbage collection
    gc_collect_freq: int = 1000

    # evaluation
    async_eval_gpus: Optional[int] = None

    # profiling
    # checkpointing


class MonitorContext:
    def __init__(self, config: MonitorConfig, model: nn.Module):
        # initialize all context

        self.nb_params = sum([p.numel() for p in model.parameters()])
        logger.info(f"Model built with {self.nb_params:,} parameters")

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass
