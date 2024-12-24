import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.profiler as profiler

from ..cluster.utils import is_master_process

logger = logging.getLogger()


@dataclass
class ProfilerConfig:
    active: bool = False
    path: str = ""
    warmup: int = 0
    steps: int = 10


# def perfetto_to_html(json_file, html_file):
#     import gzip
#     import os
#     import string

#     import viztracer

#     root = os.path.dirname(viztracer.__file__)
#     sub = {}

#     json_file = gzip.open(json_file) if ".gz" in str(json_file) else open(json_file)

#     with open(
#         os.path.join(root, "html/trace_viewer_embedder.html"), encoding="utf-8"
#     ) as f:
#         tmpl = f.read()

#     with open(os.path.join(root, "html/trace_viewer_full.html"), encoding="utf-8") as f:
#         sub["trace_viewer_full"] = f.read()

#     with json_file as j:
#         content = j.read()
#         if isinstance(content, bytes):
#             content = content.decode("utf-8")
#         sub["json_data"] = content.replace("</script>", "<\\/script>")  # type: ignore

#     with open(html_file, "w+", encoding="utf-8") as output_file:
#         output_file.write(string.Template(tmpl).substitute(sub))


class Profiler:
    def __init__(self, config: ProfilerConfig):
        self.path = Path(config.path)

        self.active = config.active and is_master_process()
        self.profiler = None
        if not self.active:
            self.profiler = None
            return

        self.profiler = profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=config.warmup, warmup=0, active=config.steps, repeat=1
            ),
            on_trace_ready=self._on_trace,
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Profiling active. Traces will be saved at {self.path}")

    def __enter__(self):
        if self.profiler:
            self.profiler.__enter__()
            return self

    def _on_trace(self, prof: profiler.profile):
        logger.info("Trace ready")
        prof.export_chrome_trace(str(self.path))
        # if wandb.run is not None:
        #     html_path = trace_file.replace(".json", ".html")
        #     perfetto_to_html(trace_file, html_path)
        #     wandb.log({"profile_trace": wandb.Html(html_path)})

    def __call__(self):
        if not self.active:
            return
        logger.info("Profiling call")
        self.profiler.step()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)
