import json
from logging import getLogger
from pathlib import PosixPath
from typing import Any

import numpy as np
import pandas as pd
import yaml
import os

from .utils import flatten_config

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Configuration Utilities
# ------------------------------------------------------------------------------


def extract_config_info(
    log_dir: PosixPath, task_id: int, keys: list[str], num_keys: list[str], copy_num: bool = False
) -> dict[str, Any]:
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
    copy_num:
        Whether to copy the original values of the numerical keys.
    """
    res = {}

    # configuration information
    config_path = log_dir / "tasks" / f"{task_id}.yaml"
    with open(os.path.expandvars(config_path)) as f:
        config = flatten_config(yaml.safe_load(f))
    for key in keys:
        res[key] = config[f"run_config.{key}"]
    for key in num_keys:
        val = config[f"run_config.{key}"]
        all_val = config[f"launcher.grid.{key}"]
        res[f"num:{key}"] = all_val.index(val)
        if copy_num:
            res[key] = val

    # number of parameters
    metric_path = log_dir / "metrics" / str(task_id)
    filepath = metric_path / "info_model.jsonl"
    with open(os.path.expandvars(filepath)) as f:
        res["nb_params"] = json.loads(f.readline())["model_params"]
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
    with open(os.path.expandvars(path)) as f:
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
    with open(os.path.expandvars(path)) as f:
        for line in f:
            keys |= json.loads(line).keys()
            if not readall:
                break
    return list(keys)


def get_losses(metric_path: PosixPath, steps: list, eval: bool = False) -> dict[str, float]:
    """
    Get the loss for the given metric path.

    Parameters
    ----------
    metric_path:
        Path to metric files.
    steps:
        Training steps to snapshot the loss.
    eval:
        Whether to consider the testing or training loss.

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
    logger.debug(f"Directory {metric_path} World_size: {world_size}")
    loss /= world_size

    # extract statistics
    step = data["step"]
    for snapshot in steps:
        idx = step == snapshot
        res[f"loss_{snapshot}"] = loss[idx].item()
    res["best"] = loss.min().item()
    return res



# ------------------------------------------------------------------------------
# Postprocessing Utilities
# ------------------------------------------------------------------------------


def get_task_ids(log_dir: PosixPath) -> list[int]:
    """
    Get the task ids from the given log directory.

    Parameters
    ----------
    log_dir:
        Path to logging directory.

    Returns
    -------
    The list of task ids.
    """
    task_ids = [int(p.name) for p in (log_dir / "metrics").glob("*") if p.is_dir()]
    task_ids.sort()
    return task_ids


def process_results(
    log_dir: PosixPath, keys: list[str], num_keys: list[str], steps: list[int], eval: bool, copy_num: bool = False
) -> None:
    """
    Process the results of the given experiments.

    Parameters
    ----------
    log_dir:
        Path to logging directory.
    keys:
        The configuration keys to extract.
    num_keys:
        The keys to extract as numbers.
    steps:
        Training steps to snapshot the loss.
    eval:
        Whether to consider the testing or training loss.
    copy_num:
        Whether to copy the original values of the numerical keys.
    """
    logger.info(f"Processing results in {log_dir}")
    all_task_ids = get_task_ids(log_dir)
    for task_id in all_task_ids:
        try:
            metric_path = log_dir / "metrics" / str(task_id)
            res = extract_config_info(log_dir, task_id, keys, num_keys, copy_num=copy_num)
            res |= get_losses(metric_path, steps, eval=eval)
            # res |= get_metrics(metric_path)

            with open(os.path.expandvars(metric_path / "process.json"), "w") as f:
                print(json.dumps(res, indent=4), file=f, flush=True)
        except Exception as e:
            print(log_dir / "metrics" / str(task_id))
            logger.error(f"Error processing task {task_id}: {e}")
            continue


def get_processed_results(log_dir: PosixPath) -> pd.DataFrame:
    """
    Load multiple JSON files into a single pandas DataFrame.

    Parameters
    ----------
    log_dir:
        Path to logging directory.

    Returns
    -------
    A DataFrame containing the data from all the loaded JSON files.
    """

    logger.info(f"Loading processed results in {log_dir}")
    data = []
    for file in log_dir.rglob("process.json"):
        try:
            with open(os.path.expandvars(file)) as f:
                json_data = json.load(f)
            data.append(json_data)
        except Exception as e:
            print(f"Error loading file {file}: {str(e)}")

    return pd.DataFrame(data)
