"""
Scaling plots for all experiments.

Look at best training/testing loss for a given seed.
Report it against the number of parameters or the number of data.
Plot scaling laws.
"""

# %% Import
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nanollama.visualization import get_processed_results, jsonl_to_numpy, process_results

HOME_DIR = Path("/checkpoint/vivc/icml/")
name = "data"

# %% Process results
keys = ["data.n_data", "grid_id"]
num_keys = []
steps = [100, 300, 1000, 3000]

for exp in range(1, 5):
    log_dir = HOME_DIR / "logs" / f"exp{exp}" / name
    process_results(log_dir, keys, num_keys, steps, eval=True, copy_num=True)

# %% Load processed results in a DataFrame
all_data = []
for exp in range(1, 5):
    log_dir = HOME_DIR / "logs" / f"exp{exp}" / name
    data = get_processed_results(log_dir)

    # retrieve corresponding entropy esimates
    for key in ["gzip", "hmm"]:
        filepath = log_dir.parent / f"{key}.jsonl"
        keys = [f"{key}_difficulty", "grid_id"]
        difficulty = pd.DataFrame(jsonl_to_numpy(filepath, keys))
        data = data.merge(difficulty, left_on=["grid_id"], right_on=["grid_id"], how="left")
    all_data.append(data)

# ------------------------------------------------------------------------------
# Generate plots from the DataFrame
# ------------------------------------------------------------------------------

# %% show loss vs entropy

difficulty_key = "hmm_difficulty"
# difficulty_key = "gzip_difficulty"

for data in all_data:
    plt.figure()
    # make a line y = x
    plt.plot([0, 3], [0, 3], color="black", ls=":")
    for nb_params in data["nb_params"].unique():
        tmp = data[data["nb_params"] == nb_params]
        plt.scatter(tmp[difficulty_key], tmp["best"], label=f"{nb_params} params")
    plt.xlabel(difficulty_key)
    plt.ylabel("test loss")
    plt.legend()

# %% (DEPRECATED - TO BE CORRECTED) show loss vs alpha in the first experiment

plt.figure()
data = all_data[0]
plt.plot([0, 3], [0, 3], color="black", ls=":")
for index, name in zip([range(7), range(7, 13)], ["alpha_X", "alpha_Z"]):
    mask = None
    for i in index:
        if mask is None:
            mask = all_data[0]["num:data.gssm.nodes"] == i
        else:
            mask |= all_data[0]["num:data.gssm.nodes"] == i
    tmp = data[mask]

    for nb_params in data["nb_params"].unique():
        _tmp = tmp[tmp["nb_params"] == nb_params]
        plt.scatter(_tmp[difficulty_key], _tmp["best"], label=f"{nb_params} params, varying {name}")
plt.xlabel(difficulty_key)
plt.ylabel("test loss")
plt.legend()

# %% (DEPRECATED - TO BE CORRECTED) show loss vs graph type

for exp in range(4):
    # show loss vs nb of parameters
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    data = all_data[exp]
    all_nodes = np.unique(data["num:data.gssm.nodes"])
    all_params = np.unique(data["nb_params"])

    for node in all_nodes:
        data_node = data[data["num:data.gssm.nodes"] == node]
        for nb_params in all_params[-1:]:
            tmp = data_node[data_node["nb_params"] == nb_params]
            ax.scatter(tmp[difficulty_key], tmp["loss"], label=f"graph {node}, {nb_params} params")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"exp={exp}")
    ax.set_xlabel(difficulty_key)
    ax.set_ylabel("test loss")
    ax.legend()
    plt.show()

# %%
