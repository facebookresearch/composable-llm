import csv
import json
from pathlib import PosixPath

import numpy as np

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def jsonl_to_numpy(path: str) -> dict[str, np.ndarray]:
    """
    Convert a jsonl file to a dictionnary of numpy array

    Parameters
    ----------
    path:
        Path to the jsonl file
    """
    data: dict[str, list] = {}
    updated = {}
    with open(path) as f:
        for line in f:
            for key, value in json.loads(line).items():
                if key not in updated:
                    data[key] = []
                data[key].append(value)
                updated[key] = True
            for key in updated:
                if not updated[key]:
                    data[key].append(None)
                updated[key] = False
    # return {k: np.array(v) for k, v in data.items()}
    res = {}
    for k, v in data.items():
        if k == "batch_idx":
            continue
        res[k] = np.array(v)
    return res


# -----------------------------------------------------------------------------
# (Deprecated) Trace Visualization
# -----------------------------------------------------------------------------


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
