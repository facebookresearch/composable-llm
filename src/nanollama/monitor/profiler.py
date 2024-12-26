import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path, PosixPath

import torch
import torch.profiler as profiler

from ..cluster.utils import get_rank

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------
# Individual Profilers
# -------------------------------------------------------------------------------


class PytorchProfiler:
    """
    Wrapper around Pytorch Profiler, highly detailed, yet heavy weight.
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

    def __init__(self, path: PosixPath, wait: int, steps: int, perfetto: bool = False) -> None:
        self.path = path
        self.start_step = wait
        self.end_step = wait + steps
        self.step = 0
        self.active = False

        # wether to use perfetto format
        self.perfetto = perfetto

        # various placeholder
        self.device = None
        self.device_capacity = None
        self.times = {}

    def __enter__(self):
        logger.info(f"Light profiler active. Traces will be saved at {self.path}")
        self.device = torch.device(get_rank())
        self.device_capacity = torch.cuda.get_device_properties(self.device).total_memory
        self.file = open(self.path, "a")
        self.file.write("[\n")  # Start the JSON array
        self.start_timer()
        self.active = True

    def start_timer(self):
        """Start a timer"""
        if self.active:
            self.time = time.time()

    def end_timer(self, name: str):
        """End timer and report time"""
        if self.active and self.time:
            self.times[name] = time.time() - self.time

    def __call__(self):
        if self.step == self.start_step:
            self.__enter__()

        if self.step == self.end_step:
            self.__exit__(None, None, None)

        if self.active:
            cuda_info = torch.cuda.memory_stats(self.device)

            max_mem = cuda_info["active_bytes.all.peak"]
            mem_reserved = cuda_info["reserved_bytes.all.peak"]

            event = self.times | {
                "step": self.step,
                "mem_gib": self.to_gib(max_mem),
                "mem_ratio": self.to_ratio(max_mem),
                "mem_reserved_gib": self.to_gib(mem_reserved),
                "mem_reserved_ratio": self.to_ratio(mem_reserved),
                "num_alloc_retries": cuda_info["num_alloc_retries"],
                "num_ooms": cuda_info["num_ooms"],
            }
            if self.perfetto:
                event = {
                    "name": "light",
                    "ph": "C",
                    "ts": round(time.time() * 1e6),  # Convert to microseconds
                    "pid": 0,
                    "tid": 0,
                    "args": event,
                }
            else:
                event["timestamp"] = time.time()

            json.dump(event, self.file, indent=2)
            if self.step < self.end_step - 1:
                self.file.write(",\n")

            torch.cuda.reset_peak_memory_stats()

        self.step += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.active:
            return
        self.file.write("\n]\n")
        self.file.close()

        logger.info(f"Light profiler traces saved to {self.path}")

        # free placeholders
        self.device = None
        self.device_capacity = None
        self.events = []
        self.times = {}
        self.active = False

    @staticmethod
    def to_gib(memory):
        return memory / (1024**3)

    def to_ratio(self, memory):
        return memory / self.device_capacity


# -------------------------------------------------------------------------------
# Generic Wrapper
# -------------------------------------------------------------------------------


@dataclass
class ProfilerConfig:
    active: bool = True
    path: str = ""
    wait: int = 1
    steps: int = 0
    pytorch: bool = False
    light: bool = True
    perfetto: bool = False


class Profiler:
    def __init__(self, config: ProfilerConfig):
        self.profilers = []
        self.light = None
        if not config.active:
            return

        self.path = Path(config.path)
        rank = get_rank()

        self.path.mkdir(parents=True, exist_ok=True)
        if config.pytorch:
            self.pytorch = PytorchProfiler(self.path / f"pytorch_{rank}.pt.trace.json", config.wait, config.steps)
            self.profilers.append(self.pytorch)
        if config.light:
            self.light = LightProfiler(
                self.path / f"light_{rank}.json", config.wait, config.steps, perfetto=config.perfetto
            )
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
