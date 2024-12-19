import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from .distributed import get_global_rank
from .train import TrainState
from .utils import trigger_update
from .wandb import WandbConfig, WandbManager

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    # logging
    name: str = "composition_default"
    dir: str = ""
    overwrite: bool = False  # whether to overwrite logging directory
    log_period: int = 100
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # reproducibility
    seed: int = 42

    # garbage collection frequency
    gc_period: int = 1000

    # evaluation
    async_eval_gpus: Optional[int] = None

    # probing
    # profiling

    def __post_init__(self):
        if not self.dir:
            self.dir = str(Path.home() / "logs" / self.name)
            print(f"Logging directory set to {self.dir}")


class MonitorsManager:
    def __init__(self, config: MonitorConfig):
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        self.log_dir = Path(config.dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.gc_period = config.gc_period

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.wandb = None
        if get_global_rank() == 0:
            if config.wandb.active:
                self.wandb = WandbManager(config.wandb, log_dir=self.log_dir)

    def __enter__(self):

        # disable garbage collection
        gc.disable()

        # enter wandb context
        if self.wandb is not None:
            self.wandb.__enter__()

        return self

    def report_objects(
        self, model: nn.Module, optimizer: Optimizer, scheduler: lr_scheduler.LambdaLR, state: TrainState
    ):
        """
        Report the objects to monitor.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = state

        self.nb_params = sum([p.numel() for p in model.parameters()])
        logger.info(f"Model built with {self.nb_params:,} parameters")

    def __call__(self):

        # manual garbage collection
        if trigger_update(self.state, self.gc_period):
            logger.info("garbage collection")
            gc.collect()

    def report_metrics(self, metrics: dict):
        """
        Report the metrics to monitor.
        """
        logger.info(f"DataLoader time: {metrics['data_time']}, Model time: {metrics['model_time']}")
        logger.info(f"Step: {metrics['step']}, Loss: {metrics['loss']}")

        if self.wandb:
            self.wandb.report_metrics(metrics, step=metrics["step"])

    def __exit__(self, exc_type, exc_value, traceback):
        gc.collect()

        if self.wandb is not None:
            self.wandb.__exit__(exc_type, exc_value, traceback)
