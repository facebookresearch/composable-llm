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

HOME_DIR = Path("/private/home/vivc/")
name = "onfly"

# %% Process results
keys = ["model.emb_dim", "data.seed"]
num_keys = ["data.gssm.nodes"]
steps = [30, 100, 300, 1000, 2000]

for exp in range(1, 5):
    log_dir = HOME_DIR / "logs" / f"exp{exp}" / name
    process_results(log_dir, keys, num_keys, steps, eval=False, copy_num=True)

# %% Load processed results in a DataFrame
all_data = []
for exp in range(1, 5):
    log_dir = HOME_DIR / "logs" / f"exp{exp}" / name
    data = get_processed_results(log_dir)

    filepath = log_dir / "difficulty.jsonl"
    keys = ["difficulty_hmm", "difficulty_gzip", "seed", "num:data.gssm.node"]
    difficulty = pd.DataFrame(jsonl_to_numpy(filepath, keys))

    merged_df = data.merge(
        difficulty, left_on=["data.seed", "num:data.gssm.nodes"], right_on=["seed", "num:data.gssm.node"], how="left"
    )
    config_id = merged_df.groupby("num:data.gssm.nodes")["data.gssm.nodes"].first()
    merged_df.drop(columns=["seed", "num:data.gssm.node", "data.gssm.nodes"], inplace=True)

    all_data.append(merged_df)

# %% Generate plots from the DataFrame
for exp in range(4):
    # show loss vs nb of parameters
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    data = all_data[exp]
    data.drop(columns=["data.gssm.nodes"], inplace=True)
    all_nodes = np.unique(data["num:data.gssm.nodes"])

    for node in all_nodes:
        data_node = data[data["num:data.gssm.nodes"] == node]
        data_node = data_node.groupby(
            [
                "nb_params",
            ]
        ).mean()
        ax.plot(data_node.index.get_level_values("nb_params"), data_node["best"], label=f"nodes={node}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"exp={exp}")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Best train loss")
    ax.legend()
    plt.show()

    # show loss vs entropy
    plt.figure()
    for nb_params in data["nb_params"].unique():
        tmp = data[data["nb_params"] == nb_params]
        plt.scatter(tmp["difficulty_gzip"], tmp["best"], label=f"{nb_params} params")
    plt.xlabel("difficulty_gzip")
    plt.ylabel("best loss")
    plt.legend()

# %%
