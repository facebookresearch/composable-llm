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

HOME_DIR = Path("/private/home/nolte/")
name = "onfly"

# %% Process results
keys = ["model.emb_dim", "data.seed"]
num_keys = ["data.gssm.nodes"]
steps = [30, 100, 300, 1000]

for exp in range(1,2):
    log_dir = HOME_DIR / "logs" / f"exp{exp}" / name
    process_results(log_dir, keys, num_keys, steps, eval=False, copy_num=True)

# %% Load processed results in a DataFrame
all_data = []
for exp in range(1, 2):
    log_dir = HOME_DIR / "logs" / f"exp{exp}" / name
    data = get_processed_results(log_dir)

    config_id = data.groupby("num:data.gssm.nodes")["data.gssm.nodes"].first()
    data.drop(columns=["seed", "num:data.gssm.node", "data.gssm.nodes"], inplace=True)

    all_data.append(data)


# ------------------------------------------------------------------------------
# Generate plots from the DataFrame
# ------------------------------------------------------------------------------

# %% show loss vs entropy

difficulty_key = "difficulty_hmm"
# difficulty_key = "difficulty_gzip"

for data in all_data:
    plt.figure()
    # make a line y = x
    plt.plot([0, 3], [0, 3], color="black", ls=":")
    for nb_params in data["nb_params"].unique():
        tmp = data[data["nb_params"] == nb_params]
        plt.scatter(tmp[difficulty_key], tmp["loss"], label=f"{nb_params} params")
    plt.xlabel(difficulty_key)
    plt.ylabel("test loss")
    plt.legend()

# %% show loss vs alpha in the first experiment

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
        plt.scatter(_tmp[difficulty_key], _tmp["loss"], label=f"{nb_params} params, varying {name}")
plt.xlabel(difficulty_key)
plt.ylabel("test loss")
plt.legend()

# %% show loss vs graph type

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
