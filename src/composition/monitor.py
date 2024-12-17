import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    # logging
    name: str = "comp_default"
    dir: str = ""

    # reproducibility
    seed: int = 42

    # garbage collection
    gc_collect_freq: int = 1000

    # evaluation
    async_eval_gpus: Optional[int] = None

    # probing
    # profiling

    def __post_init__(self):
        if not self.dir:
            self.dir = str(Path.home() / "logs" / self.name)


class MonitorsManager:
    def __init__(self, config: MonitorConfig):
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def report_model(self, model: nn.Module):
        self.nb_params = sum([p.numel() for p in model.parameters()])
        logger.info(f"Model built with {self.nb_params:,} parameters")
