import logging
import time
from dataclasses import dataclass
from pathlib import Path, PosixPath

import torch
import torch.profiler as profiler

from ..cluster import get_rank
from ..train import TrainState

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------
# Individual Profilers
# -------------------------------------------------------------------------------


class HeavyProfiler:
    """
    Wrapper around Pytorch Profiler, highly detailed, yet heavy.
    """

    ACTIVITIES = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    def __init__(self, path: PosixPath, wait: int, steps: int):
        self.path = path
        self.profiler = profiler.profile(
            activities=self.ACTIVITIES,
            schedule=torch.profiler.schedule(
                skip_first=0,
                wait=max(wait - 1, 0),
                warmup=min(wait, 1),
                active=steps,
                repeat=1,
            ),
            on_trace_ready=self._on_trace,
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
        )

    def __enter__(self):
        self.profiler.__enter__()
        logger.info(f"Pytorch profiler active. Traces will be saved at {self.path}")

    def _on_trace(self, prof: profiler.profile):
        prof.export_chrome_trace(str(self.path))
        logger.info(f"Pytorch profiler traces saved to {self.path}")
        self.profiler.__exit__(None, None, None)
        self.profiler = None

    def __call__(self):
        if self.profiler:
            self.profiler.step()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)


class LightProfiler:
    """
    Minimal profiler.
    """

    def __init__(self, path: PosixPath, wait: int, steps: int):
        self.path = path
        self.start_step = wait
        self.end_step = wait + steps
        self.step = 0
        self.active = False

        # various placeholder
        self.device = None
        self.capacity = None
        self.times = {}
        self.state = None

    def __enter__(self):
        logger.info(f"Light profiler active. Traces will be saved at {self.path}")
        self.device = torch.device(get_rank())
        self.capacity = torch.cuda.get_device_properties(self.device).total_memory
        self.file = open(self.path, "w")
        self.start_timer()

    def start_timer(self):
        """Start a timer"""
        if self.device:  # act as an active flag
            self.time = time.time()

    def end_timer(self, name: str, sync=False):
        """End timer and report time"""
        if self.device:  # act as an active flag
            if sync:
                torch.cuda.synchronize(get_rank())
            self.times[name] = time.time() - self.time

    def report_objects(self, train_state: TrainState):
        self.state = train_state

    def __call__(self):
        if self.step >= self.start_step and self.step < self.end_step:
            # write csv header
            if self.step == self.start_step:
                header = list(self.times.keys()) + [
                    "ts",
                    "step",
                    "acc_step",
                    "mem",
                    "mem_reserved",
                    "num_alloc_retries",
                    "num_ooms",
                    "mem_capacity",
                ]
                self.file.write(",".join(header) + "\n")

            cuda_info = torch.cuda.memory_stats(self.device)

            data = list(self.times.values()) + [
                round(time.time(), 6),
                self.state.optim.step,
                self.state.optim.acc_step,
                cuda_info["active_bytes.all.peak"],
                cuda_info["reserved_bytes.all.peak"],
                cuda_info["num_alloc_retries"],
                cuda_info["num_ooms"],
                self.capacity,
            ]
            self.file.write(",".join([str(x) for x in data]) + "\n")

            torch.cuda.reset_peak_memory_stats()

        if self.step == self.end_step:
            self.__exit__(None, None, None)

        self.step += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device is None:
            return

        self.file.close()
        logger.info(f"Light profiler traces saved to {self.path}")

        # free placeholders
        self.device = None
        self.capacity = None
        self.times = {}
        self.state = None

    @staticmethod
    def to_gib(memory):
        return memory / (1024**3)

    @staticmethod
    def to_ratio(memory, device_capacity):
        return memory / device_capacity


# -------------------------------------------------------------------------------
# Generic Wrapper
# -------------------------------------------------------------------------------


@dataclass
class ProfilerConfig:
    active: bool = True
    path: str = ""
    wait: int = 1
    steps: int = 10
    heavy: bool = False


class Profiler:
    """
    Profiler Context

    Note
    ----
    Implementation is compatible with the simultaneous usage of multiple profilers
    """

    def __init__(self, config: ProfilerConfig):
        self.profilers = []
        self.light = None
        if not config.active:
            return

        self.path = Path(config.path)
        rank = get_rank()
        ts = time.strftime("%Y%m%d_%H%M%S")

        self.path.mkdir(parents=True, exist_ok=True)
        if config.heavy:
            self.profilers.append(
                HeavyProfiler(self.path / f"heavy_{rank}_{ts}.pt.trace.json", config.wait, config.steps)
            )
        else:
            self.light = LightProfiler(self.path / f"light_{rank}_{ts}.csv", config.wait, config.steps)
            self.profilers.append(self.light)

    def __enter__(self):
        for prof in self.profilers:
            prof.__enter__()
        return self

    def __call__(self):
        for prof in self.profilers:
            prof()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for prof in self.profilers:
            prof.__exit__(exc_type, exc_val, exc_tb)

    def start_timer(self):
        if self.light:
            self.light.start_timer()

    def end_timer(self, *args, **kwargs):
        if self.light:
            self.light.end_timer(*args, **kwargs)

    def report_objects(self, train_state: TrainState):
        if self.light:
            self.light.report_objects(train_state)
