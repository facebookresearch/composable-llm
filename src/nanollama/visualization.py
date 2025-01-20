import json
from logging import getLogger
from pathlib import PosixPath
from typing import Any

import numpy as np
import yaml

from .utils import flatten_config

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Configuration Utilities
# ------------------------------------------------------------------------------


def extract_config_info(log_dir: PosixPath, task_id: int, keys: list[str], num_keys: list[str]) -> dict[str, Any]:
    """
    Extract configuration informations.

    Parameters
    ----------
    log_dir:
        Path to logging directory.
    task_id:
        Id of the task to extract information from.
    keys:
        The configuration keys to extract.
    num_keys:
        The keys to extract as numbers.
    """
    res = {}

    # configuration information
    config_path = log_dir / "tasks" / f"{task_id}.yaml"
    with open(config_path) as f:
        config = flatten_config(yaml.safe_load(f))
    for key in keys:
        res[key] = config[f"run_config.{key}"]
    for key in num_keys:
        val = config[f"run_config.{key}"]
        all_val = config[f"launcher.grid.{key}"]
        res[key] = all_val.index(val)

    # number of parameters
    metric_path = log_dir / "metrics" / str(task_id)
    filepath = metric_path / "info_model.jsonl"
    with open(filepath) as f:
        res["nb_params"] = json.loads(f.readline())['model_params']
    return res


# ------------------------------------------------------------------------------
# Metrics Utilities
# ------------------------------------------------------------------------------


def jsonl_to_numpy(path: str, keys: list[str]) -> dict[str, np.ndarray]:
    """
    Convert a jsonl file to a dictionnary of numpy array

    Parameters
    ----------
    path:
        Path to the jsonl file
    keys:
        List of keys to extract from the jsonl file

    Returns
    -------
    A dictionnary of numpy array
    """
    data: dict[str, list] = {key: [] for key in keys}
    with open(path) as f:
        # read jsonl as a csv with missing values
        for line in f:
            values: dict[str, Any] = json.loads(line)
            for key in keys:
                data[key].append(values.get(key, None))
    return {k: np.array(v) for k, v in data.items()}


def get_keys(path: str, readall: bool = True) -> list[str]:
    """
    Get keys from a jsonl file

    Parameters
    ----------
    path:
        Path to the jsonl file
    readall:
        Wether to read all lines of the file or the first one only

    Returns
    -------
    keys:
        List of keys in the jsonl file
    """
    keys = set()
    with open(path) as f:
        for line in f:
            keys |= json.loads(line).keys()
            if not readall:
                break
    return list(keys)


def get_losses(metric_path: str, steps: list, eval: bool = False) -> dict[str, float]:
    """
    Get the loss for the given metric path.

    Parameters
    ----------
    metric_path:
        The path to the metric files.
    world_size:
        The number of processes.
    steps:
        The steps to consider.
    eval:
        Whether to consider the evaluation or training loss.

    Returns
    -------
    The loss for the given metric path.
    """
    res = {}
    prefix = "eval" if eval else "raw"

    # compute the loss
    loss = None
    world_size = 0
    for filepath in metric_path.glob(f"{prefix}_*.jsonl"):
        keys = ["loss", "step"]
        data = jsonl_to_numpy(filepath, keys=keys)
        if loss is None:
            loss = data["loss"]
        else:
            loss += data["loss"]
        world_size += 1
    logger.info(f"Directory {metric_path} World_size: {world_size}")
    loss /= world_size

    # extract statistics
    step = data["step"]
    for snapshot in steps:
        idx = step == snapshot
        res[f"loss_{snapshot}"] = loss[idx].item()
    res["best"] = loss.min().item()
    return res
