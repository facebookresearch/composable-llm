"""
Utility to estimate difficulty levels of GSSM configurations.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import json
import logging
import os
import zlib
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nanollama.data.gssm import DataConfig, OnlineDataLoader, init_dataloader_state
from nanollama.utils import initialize_nested_object

logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Difficulty Estimation
# ------------------------------------------------------------------------------


def get_compression_ratio(batch: np.ndarray, level: int = 9) -> float:
    """
    Get the compression ratio of a sequence.
    Estimate the entropy of a sequence by compressing it with gzip.

    Parameters
    ----------
    seq:
        Sequence to estimate the entropy of.
    level:
        Compression level to use, by default 9.

    Returns
    -------
    Entropy estimate.
    """
    bytes_batch = batch.tobytes()
    compressed_data = zlib.compress(bytes_batch, level=level)
    return len(compressed_data) / len(bytes_batch)


def estimate_config_difficulty(config: DataConfig, level: int = 9) -> float:
    """
    Estimate the difficulty of a GSSM configuration by compressing sequences generated by it.

    Parameters
    ----------
    data_config:
        Configuration of the GSSM data loader.
    level:
        Compression level to use, by default 9.

    Returns
    -------
    Difficulty estimate.
    """
    # get batch
    state = init_dataloader_state(config)
    dataloader = OnlineDataLoader(config, state)
    batch = next(dataloader.generator)

    # estimate difficulty
    return get_compression_ratio(batch, level=level)


# ------------------------------------------------------------------------------
# Loop Over Configurations
# ------------------------------------------------------------------------------


@dataclass
class RangeValue:
    min: float = 1
    max: float = 1
    num: int = 1
    values: list[float] = field(init=False)

    def __post_init__(self):
        if self.num == 1:
            self.values = [self.min]
        self.values = np.logspace(np.log10(self.min), np.log10(self.max), num=self.num).tolist()


@dataclass
class DifficultyEstimationConfig:
    # dataloader related
    batch_size: int = 256
    seq_len: int = 2048
    seed: int = 42

    # generate configurations from base configuration and ranges for alpha values
    gssm: dict[str, Any] = field(default_factory=dict)
    alpha_X: RangeValue = field(default_factory=RangeValue)
    alpha_Z: RangeValue = field(default_factory=RangeValue)

    # saving path
    path: str = ""

    # compression level
    level: int = 9

    def __post_init__(self):
        if not self.path:
            self.path = str(Path.home() / "logs" / "difficulty_estimation.jsonl")
        else:
            self.path = os.path.expandvars(self.path)


def estimate_difficulty(config: DifficultyEstimationConfig, task_id: int = 1, nb_tasks: int = 1) -> None:
    """
    Estimate the difficulty of a GSSM configuration by compressing sequences generated by it.

    Parameters
    ----------
    config:
        Configuration of the GSSM data loader.
    task_id:
        Task id in the job array.
    nb_tasks:
        Number of tasks in the job array.

    Returns
    -------
    Difficulty estimate.
    """
    # create parent directory if it does not exist
    Path(config.path).parent.mkdir(parents=True, exist_ok=True)

    # initialize data cofnig
    gssm = config.gssm
    for node in gssm["nodes"]:
        node["alpha"] = 1

    data_keys = config.__dataclass_fields__.keys() & DataConfig.__dataclass_fields__.keys()
    data_config = initialize_nested_object(
        DataConfig,
        {"gssm": config.gssm}  # base gssm configuration
        | {key: getattr(config, key) for key in data_keys},  # dataloader configuration
    )

    # iterate over values of alpha_X and alpha_Z
    for i, (alpha_X, alpha_Z) in enumerate(product(config.alpha_X.values, config.alpha_Z.values)):
        if i % nb_tasks != (task_id - 1):
            continue

        # specialize base configuration accordingly
        for node in data_config.gssm.nodes:
            if node.observed:
                node.alpha = alpha_X
            else:
                node.alpha = alpha_Z

        logger.info(f"Estimating difficulty for alpha_X={alpha_X:.2e}, alpha_Z={alpha_Z:.2e}.")
        difficulty = {
            "difficulty": estimate_config_difficulty(data_config, level=config.level),
            "alpha_X": alpha_X,
            "alpha_Z": alpha_Z,
        }
        logger.info(f"Difficulty: {difficulty['difficulty']:.2f}")

        with open(config.path, "a") as f:
            print(json.dumps(difficulty), file=f, flush=True)


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------


def main() -> None:
    """
    Launch a difficulty estimation job from configuration file specified by cli argument.

    Usage:
    ```
    python -m --task-id 1 --nb-tasks 4 difficulty src/apps/my_app/my_config.yaml
    ```
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # parse file configuration path and task id
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument("--task-id", type=int, default=1, help="Task id in the job array.")
    parser.add_argument("--nb-tasks", type=int, default=1, help="Number of tasks in the job array.")
    args = parser.parse_args()
    path = args.config

    # load configuration from file
    with open(path) as f:
        all_configs = yaml.safe_load(f)

    logger.info(f"Running task {args.task_id} out of {args.nb_tasks}.")
    for name, config in all_configs.items():
        logger.info(f"Launching difficulty estimation with configuration: {name}")

        # initialize configuration
        config = initialize_nested_object(DifficultyEstimationConfig, config)

        # launch training
        estimate_difficulty(config, task_id=args.task_id, nb_tasks=args.nb_tasks)


if __name__ == "__main__":
    main()
