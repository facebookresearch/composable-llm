"""
Scaling plots for all experiments.

Look at best training/testing loss for a given seed.
Report it against the number of parameters or the number of data.
Plot scaling laws.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nanollama.visualization import get_processed_results, process_results

HOME_DIR = Path("/private/home/vivc/")
name = "onfly"

# %%
keys = ["model.emb_dim", "data.seed"]
num_keys = ["data.gssm.nodes"]
steps = [30, 100, 300, 1000, 2000]

for exp in range(1, 5):
    log_dir = HOME_DIR / "logs" / f"exp{exp}" / name
    process_results(log_dir, keys, num_keys, steps, eval=False)

# %%
all_data = [get_processed_results(HOME_DIR / "logs" / f"exp{exp}" / name) for exp in range(1, 5)]

for exp in range(4):
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

# %%
