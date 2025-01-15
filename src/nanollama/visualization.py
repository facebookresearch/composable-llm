import csv
import json
from pathlib import PosixPath
from typing import Any

import numpy as np

# ------------------------------------------------------------------------------
# Utilities
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


# ------------------------------------------------------------------------------
# (Deprecated) Trace Visualization
# ------------------------------------------------------------------------------


def get_traces(path: PosixPath) -> dict[int, dict[str, np.ndarray]]:
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
        header, data = _csv_to_numpy(file_path)
        res[rank] = _process_data(header, data)
    return res


def _csv_to_numpy(file_path: PosixPath) -> tuple[list[str], np.ndarray]:
    with open(file_path, newline="") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        data = np.array([row for row in csvreader], dtype=float)
    return header, data


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
