"""
Profiler

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import csv
import json
import time
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path, PosixPath
from types import TracebackType

import numpy as np
import torch
import torch.profiler as profiler

from ..distributed import get_local_rank, get_rank
from ..utils import TrainState

logger = getLogger(__name__)


# -------------------------------------------------------------------------------
# Individual Profilers
# -------------------------------------------------------------------------------


class BaseProfiler:
    def __init__(self):
        pass

    def __enter__(self):
        """Function called when entering context."""
        pass

    def __call__(self):
        """Main function ran by the Profiler"""
        pass

    def start_timer(self) -> None:
        """Start a timer"""
        pass

    def end_timer(self, name: str, **kwargs) -> None:
        """End timer and report time"""
        pass

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """Function called when exiting context"""
        pass


class HeavyProfiler(BaseProfiler):
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
            on_trace_ready=self.update,
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
        )

    def __enter__(self):
        self.profiler.__enter__()
        logger.info(f"Pytorch profiler active. Traces will be saved at {self.path}")

    def __call__(self) -> None:
        """
        Call step function when profiler is active
        """
        if self.profiler:
            self.profiler.step()

    def update(self, prof: profiler.profile) -> None:
        """
        Log profiler traces
        """
        prof.export_chrome_trace(str(self.path))
        logger.info(f"Pytorch profiler traces saved to {self.path}")
        self.profiler.__exit__(None, None, None)
        self.profiler = None

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        if self.profiler:
            self.profiler.__exit__(exc, value, tb)


class LightProfiler(BaseProfiler):
    """
    Minimal profiler.
    """

    def __init__(self, path: PosixPath, wait: int, steps: int, state: TrainState):
        self.path = path
        self.start_step = wait
        if steps < 0:
            self.end_step = float("inf")
        else:
            self.end_step = wait + steps
        self.step = 0
        self.active = False

        # device
        self.device = torch.device(get_local_rank())
        self.capacity = torch.cuda.get_device_properties(self.device).total_memory

        # various placeholder
        self.times = {}
        self.state = state

    def __enter__(self):
        logger.info(f"Light profiler active. Traces will be saved at {self.path}")
        self.file = open(self.path, "w")
        self.start_timer()

    def __call__(self) -> None:
        """
        Call update function when profiler is active
        """
        if self.step >= self.start_step and self.step <= self.end_step:
            # log profiler traces
            cuda_info = torch.cuda.memory_stats(self.device)

            metrics = self.times | {
                "ts": round(time.time(), 6),
                "step": self.state.optim.step,
                "acc_step": self.state.optim.acc_step,
                "mem": cuda_info["active_bytes.all.peak"],
                "mem_reserved": cuda_info["reserved_bytes.all.peak"],
                "num_alloc_retries": cuda_info["num_alloc_retries"],
                "num_ooms": cuda_info["num_ooms"],
                "mem_capacity": self.capacity,
            }

            print(json.dumps(metrics), file=self.file, flush=True)

            torch.cuda.reset_peak_memory_stats()

        if self.step == self.end_step:
            self.__exit__(None, None, None)

        self.step += 1

    def start_timer(self) -> None:
        if self.device:  # act as an active flag
            self.time = time.time()

    def end_timer(self, name: str, sync: bool = False) -> None:
        if self.device:  # act as an active flag
            if sync:
                torch.cuda.synchronize(self.device)
            self.times[name] = time.time() - self.time

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        if self.device is None:
            return

        self.file.close()
        logger.info(f"Light profiler traces saved to {self.path}")

        # free placeholders
        self.device = None
        self.times = {}
        self.state = None

    # ---------------------------------------------------------------------------
    # Trace Visualization
    # ---------------------------------------------------------------------------

    @classmethod
    def get_traces(cls, path: PosixPath) -> dict[int, dict[str, np.ndarray]]:
        """
        Get traces from csv files

        Example
        -------
        ```python
        from pathlib import Path
        import matplotlib.pyplot as plt
        from nanollama.monitor.profiler import LightProfiler

        path = <your path to profiler traces>
        res = LightProfiler.get_traces(path)
        xlabel = 'step'
        keys = list(res[0].keys())
        for key in keys:
            plt.figure()
            for rank in res:
                data = res[rank]
                plt.plot(data[xlabel], data[key], label=f"rank {rank}")
            plt.legend(); plt.title(key); plt.xlabel(xlabel)
        ```
        """
        res = {}
        for file_path in path.glob("*.csv"):
            rank = int(str(file_path.name).split("_")[1])
            header, data = cls._csv_to_numpy(file_path)
            res[rank] = cls._process_data(header, data)
        return res

    @staticmethod
    def _csv_to_numpy(file_path: PosixPath) -> tuple[list[str], np.ndarray]:
        with open(file_path, newline="") as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)
            data = np.array([row for row in csvreader], dtype=float)
        return header, data

    @staticmethod
    def _process_data(header: list[str], data: np.ndarray) -> dict[str, np.ndarray]:
        res = {}
        index = -1
        assert header[index] == "mem_capacity"
        capacity = data[0, index]
        for i, key in enumerate(header):
            res[key] = data[:, i]
            if key in ["mem", "mem_reserved"]:
                res[key + "_gib"] = res[key] / (1024**3)
                res[key + "_ratio"] = res[key] / capacity
        return res


# -------------------------------------------------------------------------------
# Generic Wrapper
# -------------------------------------------------------------------------------


@dataclass
class ProfilerConfig:
    active: bool = True
    wait: int = 1
    steps: int = 1
    heavy: bool = False
    path: str = field(init=False, default="")

    def __check_init__(self):
        """Check validity of arguments."""
        assert self.path, "path was not set"


class Profiler:
    """
    Profiler Context

    Note
    ----
    Implementation is compatible with the simultaneous usage of multiple profilers
    """

    def __init__(self, config: ProfilerConfig, state: TrainState = None):
        self.profilers: list[BaseProfiler] = []
        self.light = None
        if not config.active:
            return

        self.path = Path(config.path)
        rank = get_rank()

        self.path.mkdir(parents=True, exist_ok=True)
        if config.heavy:
            path = self._unique_path(self.path, f"hprof_{rank}_", ".pt.trace.json")
            self.profilers.append(HeavyProfiler(path, config.wait, config.steps))
        else:
            path = self._unique_path(self.path, f"prof_{rank}_", ".jsonl")
            self.profilers.append(LightProfiler(path, config.wait, config.steps, state=state))

    def __enter__(self) -> "Profiler":
        for prof in self.profilers:
            prof.__enter__()
        return self

    def __call__(self) -> None:
        """
        Call profilers
        """
        for prof in self.profilers:
            prof()

    def start_timer(self) -> None:
        """
        Start a timer
        """
        for prof in self.profilers:
            prof.start_timer()

    def end_timer(self, name: str, **kwargs) -> None:
        """
        End timer and report time
        """
        for prof in self.profilers:
            prof.end_timer(name, **kwargs)

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        for prof in self.profilers:
            prof.__exit__(exc, value, tb)

    @staticmethod
    def _unique_path(path: PosixPath, prefix: str, suffix: str) -> PosixPath:
        i = 0
        while i < 1000:
            i += 1
            file_path = path / f"{prefix}{i}{suffix}"
            if file_path.exists():
                continue
            return file_path
        raise ValueError("Could not find unique path")
