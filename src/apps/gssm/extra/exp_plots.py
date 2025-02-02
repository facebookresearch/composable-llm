"""
Scaling plots for all experiments.

Look at best training/testing loss for a given seed.
Report it against the number of parameters or the number of data.
Plot scaling laws.
"""

# %% Import
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nanollama.visualization import get_processed_results, jsonl_to_numpy, process_results, read_indented_jsonl
from nanollama.utils import flatten_config

LOG_DIR = Path("/checkpoint/vivc/icml/")
CODE_DIR = Path("/private/home/vivc/code/composable-llm/")
name = "data"
scaling_key = "data.n_data"
keys = [scaling_key, "grid_id"]

data_range = range(1, 9)
data_range = [1, 2, 3, 4, 5, 7, 8]

# name = "params"
# scaling_key = "nb_params"
# keys = ["grid_id"]

# %% Process results
num_keys = []
steps = [100, 300, 1000, 3000]

for exp in data_range:
    log_dir = LOG_DIR / "logs" / f"exp{exp}" / name
    process_results(log_dir, keys, num_keys, steps, eval=True, copy_num=True)

# %% Load processed results in a DataFrame
all_data = []
for exp in data_range:
    log_dir = LOG_DIR / "logs" / f"exp{exp}" / name
    data = get_processed_results(log_dir)

    # retrieve corresponding entropy estimates
    key = "hmm"
    filepath = log_dir.parent / f"{key}.jsonl"
    keys = [f"{key}_difficulty", "grid_id"]
    difficulty = pd.DataFrame(jsonl_to_numpy(filepath, keys))
    data = data.merge(difficulty, left_on=["grid_id"], right_on=["grid_id"], how="left")

    # retrieve graph information
    prefix = CODE_DIR / "src" / "apps" / "gssm" / "configs" / f"experiment{exp}"
    filepath = prefix / ".gssm_id_path.jsonl"
    nodepath = prefix / ".gssm_id_config.jsonl"
    keys = ["grid_id", "gssm_id", "seed"]
    graph_info = pd.DataFrame(jsonl_to_numpy(filepath, keys))
    def make_nice(cfg):
        cfg["nodes"] = {n["name"]: n for n in cfg["nodes"]}
        return flatten_config(cfg)
    id_to_config = [make_nice(x) for x in read_indented_jsonl(nodepath)]
    nodes_info = pd.DataFrame(id_to_config)
    graph_info = graph_info.merge(nodes_info, left_on=["gssm_id"], right_on=["gssm_id"], how="left")
    data = data.merge(graph_info, left_on=["grid_id"], right_on=["grid_id"], how="left")

    all_data.append(data)

# ------------------------------------------------------------------------------
# Generate plots from the DataFrame
# ------------------------------------------------------------------------------

# %% show loss vs entropy

difficulty_key = "hmm_difficulty"


for exp, data in zip(data_range, all_data):
    plt.figure()
    # make a line y = x
    # plt.plot([2.5, 5], [2.5, 5], color="black", ls=":")
    all_scales = data[scaling_key].unique()[:3]
    nb_data = len(all_scales)
    all_graph = data["gssm_id"].unique()
    for i, grid_id in enumerate(all_graph):
        data1 = data[data["gssm_id"] == grid_id]
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
            # alpha = 0.8
            # xaxis = data2[difficulty_key].values
            # order = xaxis.argsort()
            # xaxis = xaxis[order]
            # yaxis = data2["best"].values[order]
            # plt.plot(
            #     xaxis,
            #     yaxis,
            #     label=f"{scale} {scaling_key} {grid_id}",
            #     color=f"C{i}",
            #     alpha=alpha,
            # )
        # print(f"graph {grid_id} done")
    plt.xlabel(difficulty_key)
    plt.ylabel("test loss")
    plt.title(f"exp={exp}")
    # plt.loglog()
    # plt.legend()

# %% show loss vs alpha in the first experiment

plt.figure()
data = all_data[0]
# plt.plot([0, 3], [0, 3], color="black", ls=":")
for index, name in zip([range(7), range(7, 13)], ["alpha_X", "alpha_Z"]):
    mask = None
    for i in index:
        if mask is None:
            mask = all_data[0]["gssm_id"] == i
        else:
            mask |= all_data[0]["gssm_id"] == i
    tmp = data[mask]

    for scale in data[scaling_key].unique()[:3]:
        _tmp = tmp[tmp[scaling_key] == scale]
        _name = {"X": "Z", "Z": "X"}[name[-1]]
        _key = {"data.n_data": "data"}[scaling_key]
        plt.scatter(_tmp[difficulty_key], _tmp["best"], label=rf"{scale} {_key}, $\alpha_{_name}$" + r"$=10^{-3}$")
plt.xlabel(difficulty_key)
plt.ylabel("test loss")
plt.legend()
# plt.loglog()

# %%
