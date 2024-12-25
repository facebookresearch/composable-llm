import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.profiler as profiler

from ..cluster.utils import is_master_process

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------
# Individual Profilers
# -------------------------------------------------------------------------------


class TimelineProfiler:
    ACTIVITIES = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    def __init__(self, path: str, wait: int, steps: int):
        self.path = path
        self.profiler = profiler.profile(
            activities=self.ACTIVITIES,
            schedule=torch.profiler.schedule(
                skip_first=0,
                wait=0,
                warmup=wait,
                active=steps,
                repeat=1,
            ),
            on_trace_ready=self._on_trace,
            profile_memory=False,
            record_shapes=False,
            with_stack=True,
            with_flops=False,
        )

    def __enter__(self):
        self.profiler.__enter__()
        logger.info(f"Profiling active. Traces will be saved at {self.path}")

    def _on_trace(self, prof: profiler.profile):
        prof.export_chrome_trace(str(self.path))
        logger.info(f"Timeline traces saved to {self.path}")

    def __call__(self):
        self.profiler.step()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        logger.info("Exiting Timeline Profiler")


class MemoryProfiler:
    def __init__(self, path: str, wait: int, steps: int) -> None:
        self.path = path
        self.start_step = wait
        self.end_step = wait + steps
        self.step = 0
        # flag to manually trigger context entering and exiting
        self.active = False

    def __enter__(self):
        if not self.active:
            return
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )
        logger.info(f"Memory Profiler active. Traces will be saved at {self.path}")

    def __call__(self):
        if self.step == self.start_step:
            self.active = True
            self.__enter__()
        if self.step == self.end_step:
            self.__exit__(None, None, None)
            self.active = False
        self.step += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.active:
            return
        snapshot = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(False)

        # No data was recorded - avoids a `ValueError` in `trace_plot`
        if all(len(t) == 0 for t in snapshot["device_traces"]):
            logger.info("No memory allocation recorded.")
            return

        # Dump to disk
        with open(self.path, "w") as f:
            f.write(torch.cuda._memory_viz.trace_plot(snapshot, device=None, plot_segments=False))
        logger.info(f"Memory snapshot saved to {self.path}")


# -------------------------------------------------------------------------------
# Generic Wrapper
# -------------------------------------------------------------------------------


@dataclass
class ProfilerConfig:
    active: bool = False
    path: str = ""
    wait: int = 1
    steps: int = 10
    repeat: int = 1
    export_to_wandb: bool = False


class Profiler:
    def __init__(self, config: ProfilerConfig):
        self.active = config.active and is_master_process()

        self.path = Path(config.path)
        self.wandb = config.export_to_wandb

        self.profiler = []
        if not self.active:
            return

        self.path.mkdir(parents=True, exist_ok=True)
        self.profilers = [
            TimelineProfiler(self.path / "timeline.json", config.wait, config.steps),
            MemoryProfiler(self.path / "memory.json", config.wait, config.steps),
        ]

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


# from dataclasses import namedtuple

# GPUMemStats = namedtuple(
#     "GPUMemStats",
#     [
#         "max_active_gib",
#         "max_active_pct",
#         "max_reserved_gib",
#         "max_reserved_pct",
#         "num_alloc_retries",
#         "num_ooms",
#         "power_draw",
#     ],
# )

# class GPUProfiler:
#     """Profiler that monitors GPU memory usage."""

#     def __init__(self, main_profiler: "_Profiler", device: str = "cuda:0") -> None:
#         self.main_profiler = main_profiler
#         self.device = torch.device(device)
#         self.device_name = torch.cuda.get_device_name(self.device)
#         self.device_index = torch.cuda.current_device()
#         self.device_capacity = torch.cuda.get_device_properties(self.device).total_memory
#         self.device_capacity_gib = self._to_gib(self.device_capacity)
#         torch.cuda.reset_peak_memory_stats()
#         torch.cuda.empty_cache()
#         logger.info(
#             f"GPU capacity: {self.device_name} ({self.device_index}) "
#             f"with {self.device_capacity_gib:.2f} GiB memory"
#         )

#     def _to_gib(self, memory_in_bytes):
#         _gib_in_bytes = 1024 * 1024 * 1024
#         return memory_in_bytes / _gib_in_bytes

#     def _to_pct(self, memory):
#         return 100 * memory / self.device_capacity

#     def get_peak_stats(self):
#         cuda_info = torch.cuda.memory_stats(self.device)
#         max_active = cuda_info["active_bytes.all.peak"]
#         max_active_gib = self._to_gib(max_active)
#         max_active_pct = self._to_pct(max_active)
#         max_reserved = cuda_info["reserved_bytes.all.peak"]
#         max_reserved_gib = self._to_gib(max_reserved)
#         max_reserved_pct = self._to_pct(max_reserved)
#         num_retries = cuda_info["num_alloc_retries"]
#         num_ooms = cuda_info["num_ooms"]
#         power_draw = torch.cuda.power_draw()

#         if num_retries > 0:
#             logger.warning(f"{num_retries} CUDA memory allocation retries.")
#         if num_ooms > 0:
#             logger.warning(f"{num_ooms} CUDA OOM errors thrown.")

#         return GPUMemStats(
#             max_active_gib,
#             max_active_pct,
#             max_reserved_gib,
#             max_reserved_pct,
#             num_retries,
#             num_ooms,
#             power_draw=power_draw,
#         )

#     def __enter__(self):
#         logger.info(f"Starting GPUProfiler for {self.device_name}")
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         stats = self.get_peak_stats()
#         logger.info(f"GPU Memory Stats: {stats}")
#         self.main_profiler.summary.append(("GPU Memory Stats", str(stats)))

#     def step(self) -> None:
#         # Check if the current step is the last one in the schedule
#         if self.main_profiler.done_steps >= self.main_profiler.last_step:
#             self.__exit__(None, None, None)
