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
import torch
import yaml

from nanollama.data.gssm import DataConfig, OnlineDataLoader, init_dataloader_state
from nanollama.launcher import get_configs_from_grid
from nanollama.utils import flatten_config, initialize_nested_object

from .hidden_markov_model import HMM

logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Difficulty Estimation
# ------------------------------------------------------------------------------


def gzip_loss(config: DataConfig, level: int = 9) -> float:
    """
    Estimate the entropy of a GSSM configuration by compressing sequences generated by it with gzip.

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
    compressed_data = zlib.compress(batch.tobytes(), level=level)
    return len(compressed_data) / batch.size


def hmm_loss(config: DataConfig) -> float:
    state = init_dataloader_state(config)
    dataloader = OnlineDataLoader(config, state)
    batch = next(dataloader.generator)

    hmm = HMM(top_node=dataloader.node)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    entropys = hmm.entropy_of_observations(batch.T, device=device)
    return entropys.mean().item()


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
    seeds: list[int] = field(default_factory=lambda: [0])

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


def estimate_alphas_entropy(config: DifficultyEstimationConfig, task_id: int = 1, nb_tasks: int = 1) -> None:
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

    # iterate over values of alpha_X, alpha_Z and seed
    for i, (alpha_X, alpha_Z, seed) in enumerate(product(config.alpha_X.values, config.alpha_Z.values, config.seeds)):
        if i % nb_tasks != (task_id - 1):
            continue

        # set seed
        data_config.seed = seed

        # specialize base configuration accordingly
        for node in data_config.gssm.nodes:
            if node.observed:
                node.alpha = alpha_X
            else:
                node.alpha = alpha_Z

        logger.info(f"Estimating difficulty for alpha_X={alpha_X:.2e}, alpha_Z={alpha_Z:.2e}, seed={seed}.")
        difficulty = {
            "seed": seed,
            # "difficulty_hmm": hmm_loss(data_config),
            "difficulty_hmm": 0,
            "difficulty_gzip": gzip_loss(data_config, level=config.level),
            "alpha_X": alpha_X,
            "alpha_Z": alpha_Z,
        }
        logger.info(f"Difficulty: {difficulty['difficulty_hmm']:.2f}, {difficulty['difficulty_gzip']:.2f}")

        with open(config.path, "a") as f:
            print(json.dumps(difficulty), file=f, flush=True)


# ------------------------------------------------------------------------------
# Result parser
# ------------------------------------------------------------------------------


def read_results(path: str) -> dict[str, float]:
    alpha_X, alpha_Z, difficulty_hmm, difficulty_gzip = [], [], [], []
    with open(path) as f:
        for line in f:
            try:
                res = json.loads(line)
            except Exception as e:
                print(e)
                continue
            for key in ["alpha_X", "alpha_Z", "difficulty_hmm", "difficulty_gzip"]:
                locals()[key].append(res[key])

    alpha_X = np.array(alpha_X)
    alpha_Z = np.array(alpha_Z)
    difficulty_hmm = np.array(difficulty_hmm)
    difficulty_gzip = np.array(difficulty_gzip)

    all_alpha_X = np.unique(alpha_X)
    all_alpha_Z = np.unique(alpha_Z)

    ave_difficulty_hmm = {}
    ave_difficulty_gzip = {}
    for alphaX, alphaZ in product(all_alpha_X, all_alpha_Z):
        idx = (alpha_X == alphaX) & (alpha_Z == alphaZ)
        ave_difficulty_hmm[(float(alphaX), float(alphaZ))] = float(difficulty_hmm[idx].mean())
        ave_difficulty_gzip[(float(alphaX), float(alphaZ))] = float(difficulty_gzip[idx].mean())

    return ave_difficulty_hmm, ave_difficulty_gzip


# ------------------------------------------------------------------------------
# Main functions
# ------------------------------------------------------------------------------


def main(path: str, task_id: int, nb_tasks: int, bsz: int) -> None:
    """
    Launch a difficulty estimation job from configuration file specified by cli argument.

    Usage:
    ```
    python -m apps.my_app.difficulty --task-id 1 --nb-tasks 4 main src/apps/my_app/my_config.yaml
    ```
    """
    # load configuration from file
    with open(path) as f:
        all_configs = yaml.safe_load(f)

    logger.info(f"Running task {task_id} out of {nb_tasks}.")
    for name, config in all_configs.items():
        logger.info(f"Launching difficulty estimation with configuration: {name}")

        # initialize configuration
        config = initialize_nested_object(DifficultyEstimationConfig, config)
        if bsz:
            config.batch_size = bsz

        # launch training
        estimate_alphas_entropy(config, task_id=task_id, nb_tasks=nb_tasks)


def estimate_entropy(path: str, task_id: int, nb_tasks: int, bsz: int) -> None:
    """
    Launch a difficulty estimation job from configuration file specified by cli argument.

    Usage:
    ```
    python -m apps.my_app.difficulty --task-id 1 --nb-tasks 4 entropy src/apps/my_app/my_config.yaml
    ```
    """
    with open(path) as f:
        file_config = yaml.safe_load(f)

    # initialize configuration
    log_dir = Path(os.path.expandvars(file_config["launcher"]["log_dir"]))

    grid_config = {"data": file_config["launcher"]["grid"]["data"]}
    run_config = {"data": file_config["run_config"]["data"]}
    all_configs = get_configs_from_grid(run_config, grid_config)

    all_nodes = flatten_config(grid_config)["data.gssm.nodes"]

    for i, config_dict in enumerate(all_configs):
        if i % nb_tasks != (task_id - 1):
            continue

        config = initialize_nested_object(DataConfig, config_dict["data"], inplace=False)
        if bsz:
            config.batch_size = bsz
        nodes = flatten_config(config_dict)["data.gssm.nodes"]

        difficulty = {
            "seed": config.seed,
            # "difficulty_hmm": hmm_loss(config),
            "difficulty_hmm": 0,
            "difficulty_gzip": gzip_loss(config),
            "num:data.gssm.node": all_nodes.index(nodes),
            "data.gssm.node": nodes,
        }

        with open(log_dir / "difficulty.jsonl", "a") as f:
            print(json.dumps(difficulty), file=f, flush=True)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="Difficulty Estimation Tool")
    parser.add_argument("command", type=str, help="Whether to run the main or entropy command.")
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument("--task-id", type=int, default=1, help="Task id in the job array.")
    parser.add_argument("--nb-tasks", type=int, default=1, help="Number of tasks in the job array.")
    parser.add_argument("--bsz", type=int, default=0, help="Number of sample to compute entropy estimate.")
    args = parser.parse_args()

    if args.command == "main":
        main(args.config, args.task_id, args.nb_tasks, args.bsz)
    elif args.command == "entropy":
        estimate_entropy(args.config, args.task_id, args.nb_tasks, args.bsz)
    else:
        parser.print_help()
