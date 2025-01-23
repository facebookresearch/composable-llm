"""
Entropy computation

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import json
import logging
import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from types import TracebackType

import torch
import yaml
from numpy.random import default_rng

from ...nanollama.data.gssm import DataConfig as GSSMConfig
from ...nanollama.data.gssm import build_gssm, init_dataloader_state
from ...nanollama.data.hdf5 import DataConfig, FileEvaluator
from ...nanollama.distributed import ClusterConfig, ClusterManager, get_local_rank, get_rank, is_master_process
from ...nanollama.monitor import (
    PreemptionHandler,
)
from ...nanollama.utils import initialize_nested_object
from .hidden_markov_model import HMM

logger = getLogger("nanollama")

# ------------------------------------------------------------------------------
# Online Evaluation
# ------------------------------------------------------------------------------


@dataclass
class EntropyConfig:
    log_dir: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    gssm: GSSMConfig = field(default_factory=GSSMConfig)
    hmm: HMM = field(init=False)

    def __post_init__(self) -> None:
        """Check validity of arguments"""
        assert self.log_dir, "No logging directory set."
        self.log_dir = os.path.expandvars(self.log_dir)

        state = init_dataloader_state(self.gssm)
        rng = default_rng()
        rng.bit_generator.state = state.graph_rng_state
        node = build_gssm(self.gssm.gssm, rng=rng)
        self.hmm = HMM(top_node=node)

        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()


class EntropyComputer:
    """
    Evaluation manager

    Running evaluation into chunks in order to handle job preemption.

    Usage:
    ```python
    with EvalComputer(*args) as computer:
        while next(computer):
            pass
    """

    def __init__(self, config: EntropyConfig) -> None:
        self.data_config = config.data
        self.hmm = config.hmm
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"eval_{get_rank()}.jsonl"
        self.tmp_file = log_dir / f".{get_rank()}.tmp"

        self.step = 0
        self.loss = 0
        self.scaling = 0
        self.device = torch.device(get_local_rank()) if torch.cuda.is_available() else torch.device("cpu")

    def __enter__(self) -> "EntropyComputer":
        # logger.info("Evaluating model.")

        # retrieve previous computations
        if self.tmp_file.exists():
            with open(os.path.expandvars(self.tmp_file)) as f:
                data = json.loads(f)
                self.loss: float = data["loss"]
                self.scaling: float = data["scaling"]
                self.step: int = data["step"]
            # logger.info(f"Found previous evaluation at step {self.step}")

        # skip batches that were already evaluated
        self.loader = FileEvaluator(self.data_config).__enter__()
        for _ in range(self.step):
            next(self.loader)

        return self

    @torch.no_grad()
    def __next__(self) -> None:
        try:
            batch, _ = next(self.loader)

            entropy = self.hmm.entropy_of_observations(batch.T, device=self.device).mean().item() / (batch.size(1) - 1)

            # evaluate
            scaling = batch.size(0) / self.data_config.batch_size
            self.loss += scaling * entropy
            self.scaling += scaling
            self.step += 1

            return True

        except StopIteration:
            # rescale loss and save it
            self.loss /= self.scaling
            with open(os.path.expandvars(self.path), "a") as f:
                print(json.dumps({"loss": self.loss}), file=f, flush=True)

            # remove temporary file
            self.tmp_file.unlink()
            return False

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        # if the evaluation was interrupted, save the current state
        if self.tmp_file.exists():
            with open(os.path.expandvars(self.tmp_file), "w") as f:
                print(
                    json.dumps({"loss": self.loss, "scaling": self.scaling, "step": self.step}),
                    file=f,
                    flush=True,
                )

        self.loader.__exit__(exc, value, tb)


@torch.no_grad()
def eval(config: EntropyConfig) -> None:
    with ExitStack() as context_stack:
        preemption: PreemptionHandler = context_stack.enter_context(PreemptionHandler())
        context_stack.enter_context(ClusterManager(ClusterConfig()))
        computer: EntropyComputer = context_stack.enter_context(EntropyComputer(config))

        while next(computer):
            if preemption():
                logger.warning("Preemption flag set")
                break

            logger.debug(f"Evaluation. step: {computer.step} - loss: {round(computer.loss, 4):>7}")

    if is_master_process():
        logger.info(f"Test loss: {round(computer.loss, 4):>7}")

    logger.info("Evaluation done.")


def main() -> None:
    """
    Launch a evaluation job from configuration file specified by cli argument.

    Usage:
    ```
    python -m apps.my_app.eval apps/my_app/configs/my_config.yaml
    ```
    """
    import argparse

    logging.basicConfig(
        # level=logging.INFO,
        level=logging.DEBUG,
        format="[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # parse file configuration path
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", type=str, help="Path to configuration file")
    path = parser.parse_args().config

    # obtain configuration from file
    with open(os.path.expandvars(path)) as f:
        file_config = yaml.safe_load(f)
    if "run_config" in file_config:
        run_config = file_config.pop("run_config")
    else:
        run_config = file_config

    # casting logging directory to run_config
    if "launcher" in file_config:
        if "log_dir" in file_config["launcher"]:
            run_config["log_dir"] = file_config["launcher"]["log_dir"]

    # initialize configuration
    config = initialize_nested_object(EntropyConfig, run_config)

    # launch job
    eval(config)


if __name__ == "__main__":
    main()
