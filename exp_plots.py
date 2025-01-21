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

LOG_DIR = Path("/checkpoint/vivc/icml/")
CODE_DIR = Path("/private/home/vivc/code/composable-llm/")
name = "data"
scaling_key = "data.n_data"
keys = [scaling_key, "grid_id"]

# name = "params"
# scaling_key = "nb_params"
# keys = ["grid_id"]

# %% Process results
num_keys = []
steps = [100, 300, 1000, 3000]

for exp in range(1, 5):
    log_dir = LOG_DIR / "logs" / f"exp{exp}" / name
    process_results(log_dir, keys, num_keys, steps, eval=True, copy_num=True)

# %% Load processed results in a DataFrame
all_data = []
for exp in range(1, 5):
    log_dir = LOG_DIR / "logs" / f"exp{exp}" / name
    data = get_processed_results(log_dir)

    # retrieve corresponding entropy esimates
    for key in ["gzip", "hmm"]:
        filepath = log_dir.parent / f"{key}.jsonl"
        keys = [f"{key}_difficulty", "grid_id"]
        difficulty = pd.DataFrame(jsonl_to_numpy(filepath, keys))
        data = data.merge(difficulty, left_on=["grid_id"], right_on=["grid_id"], how="left")

    # retrieve graph information
    prefix = CODE_DIR / "src" / "apps" / "gssm" / "configs" / f"experiment{exp}"
    filepath = prefix / "map_grid_id_gssm_id.jsonl"
    keys = ["grid_id", "node_id", "seed"]
    graph_info = pd.DataFrame(jsonl_to_numpy(filepath, keys))
    data = data.merge(graph_info, left_on=["grid_id"], right_on=["grid_id"], how="left")

    all_data.append(data)

# ------------------------------------------------------------------------------
# Generate plots from the DataFrame
# ------------------------------------------------------------------------------

# %% show loss vs entropy

# difficulty_key = "hmm_difficulty"
difficulty_key = "gzip_difficulty"


for exp, data in enumerate(all_data):
    plt.figure()
    # make a line y = x
    # plt.plot([0, 3], [0, 3], color="black", ls=":")
    all_scales = data[scaling_key].unique()
    nb_data = len(all_scales)
    all_graph = data["node_id"].unique()
    for i, grid_id in enumerate(all_graph):
        data1 = data[data["node_id"] == grid_id]
        for alpha, scale in enumerate(all_scales):
            alpha = (alpha + 1) / nb_data
            data2 = data1[data1[scaling_key] == scale]
            plt.scatter(
                data2[difficulty_key],
                data2["best"],
                label=f"{scale} {scaling_key} {grid_id}",
                color=f"C{i}",
                alpha=alpha,
            )
        # print(f"graph {grid_id} done")
    plt.xlabel(difficulty_key)
    plt.ylabel("test loss")
    plt.title(f"exp={exp + 1}")
    # plt.legend()

# %% show loss vs alpha in the first experiment

plt.figure()
data = all_data[0]
# plt.plot([0, 3], [0, 3], color="black", ls=":")
for index, name in zip([range(7), range(7, 13)], ["alpha_X", "alpha_Z"]):
    mask = None
    for i in index:
        if mask is None:
            mask = all_data[0]["node_id"] == i
        else:
            mask |= all_data[0]["node_id"] == i
    tmp = data[mask]

    for nb_params in data["nb_params"].unique()[7:8]:
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
    all_nodes = np.unique(data["node_id"])
    all_params = np.unique(data["nb_params"])

    for node in all_nodes:
        data_node = data[data["node_id"] == node]
        for nb_params in all_params[-1:]:
            tmp = data_node[data_node["nb_params"] == nb_params]
            ax.scatter(tmp[difficulty_key], tmp["loss_300"], label=f"graph {node}, {nb_params} params")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"exp={exp}")
    ax.set_xlabel(difficulty_key)
    ax.set_ylabel("test loss")
    ax.legend()
    plt.show()

# %%
