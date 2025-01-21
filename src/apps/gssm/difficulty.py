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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
import yaml

from nanollama.data.gssm import DataConfig, OnlineDataLoader, init_dataloader_state
from nanollama.launcher import get_configs_from_grid
from nanollama.monitor.checkpoint import EvalCheckpointer
from nanollama.utils import flatten_config

from .hidden_markov_model import HMM
from .train_onfly import TrainingConfig, train_config_from_run_config

logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Difficulty Estimation
# ------------------------------------------------------------------------------


def get_dataloader(config: DataConfig):
    state = init_dataloader_state(config)
    return OnlineDataLoader(config, state)


def iterate_batches(dataloader, data_seed=19924):
    new_rng_state = np.random.default_rng(seed=data_seed).bit_generator.state
    dataloader.rng_state = new_rng_state
    yield from dataloader.generator


def gzip_loss(config: DataConfig, n_batches: int, level: int = 9) -> float:
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
    gen = iterate_batches(get_dataloader(config))

    # estimate difficulty
    entropys = []
    for _ in range(n_batches):
        batch = next(gen)
        compressed_data = zlib.compress(batch.tobytes(), level=level)
        entropys.append(len(compressed_data) / batch.size)
    return np.mean(entropys)


def hmm_loss(config: DataConfig, n_batches: int) -> float:
    dataloader = get_dataloader(config)
    gen = iterate_batches(dataloader)

    hmm = HMM(top_node=dataloader.node)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    entropys = []
    for _ in range(n_batches):
        batch = next(gen)
        entropys.append(hmm.entropy_of_observations(batch.T, device=device).mean().item())
    return np.mean(entropys) / (config.seq_len - 1)


def model_loss(config: TrainingConfig, ckpt_path: str, n_batches: int) -> float:
    def loss_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = preds.size(-1)
        return torch.nn.functional.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1))

    model = config.model_gen(config.model)
    if config.cluster.compile_model:
        model = torch.compile(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    gen = iterate_batches(get_dataloader(config.data))
    losses = []
    try:
        with EvalCheckpointer(model, ckpt_path):
            for _ in range(n_batches):
                batch = next(gen)
                batch = torch.tensor(batch).to(device)
                logits = model(batch)
                losses.append(loss_func(logits[:, :-1], batch[:, 1:]).item())

    except FileNotFoundError:
        logger.info(f"No checkpoint found at {ckpt_path}")
        losses.append(np.nan)

    return np.mean(losses)

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


def estimate_entropy(path: str, task_id: int, nb_tasks: int, bsz: int, n_batches: int) -> None:
    """
    Launch a difficulty estimation job from training configuration file specified by cli argument.

    The configuration is the same as the one to launch train_onfly

    Usage:
    ```
    python -m apps.my_app.difficulty --task-id 1 --nb-tasks 4 entropy src/apps/my_app/my_config.yaml
    ```
    """
    with open(path) as f:
        file_config = yaml.safe_load(f)

    # initialize configuration
    log_dir = Path(os.path.expandvars(file_config["launcher"]["log_dir"]))

    grid_config = file_config["launcher"]["grid"]
    run_config = file_config["run_config"]
    all_configs = get_configs_from_grid(run_config, grid_config)

    all_nodes = flatten_config(grid_config)["data.gssm.nodes"]

    for i, config_dict in tqdm.tqdm(enumerate(all_configs)):
        if i % nb_tasks != (task_id - 1):
            continue

        config = train_config_from_run_config(config_dict)

        if bsz:
            config.data.batch_size = bsz
        nodes = flatten_config(config_dict)["data.gssm.nodes"]

        # add this for model_loss
        ckpt_path = log_dir / "checkpoints" / str(i + 1)

        difficulty = {
            "seed": config.data.seed,
            "loss": model_loss(config, ckpt_path, n_batches),
            "difficulty_hmm": hmm_loss(config.data, n_batches),
            "difficulty_gzip": gzip_loss(config.data, n_batches),
            "num:data.gssm.node": all_nodes.index(nodes),
            "data.gssm.node": nodes,
        }

        write_dir = log_dir / "metrics" / str(i + 1)
        os.makedirs(write_dir, exist_ok=True)
        with open(write_dir / "metrics.json", "w") as f:
            print(json.dumps(difficulty), file=f, flush=True)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="Difficulty Estimation Tool")
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument("--task-id", type=int, default=1, help="Task id in the job array.")
    parser.add_argument("--nb-tasks", type=int, default=1, help="Number of tasks in the job array.")
    parser.add_argument("--bsz", type=int, default=0, help="batch size to compute entropy estimate.")
    parser.add_argument(
        "--n-batches",
        type=int,
        default=10,
        help="number of batches to compute entropy estimate.",
    )
    args = parser.parse_args()

    estimate_entropy(args.config, args.task_id, args.nb_tasks, args.bsz, args.n_batches)
