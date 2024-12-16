import logging
from dataclasses import dataclass, field
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

    # checkpointing


class MonitorsManager:
    def __init__(self, config: MonitorConfig):
        pass
        # initialize all context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def report_model(self, model: nn.Module):
        self.nb_params = sum([p.numel() for p in model.parameters()])
        logger.info(f"Model built with {self.nb_params:,} parameters")
