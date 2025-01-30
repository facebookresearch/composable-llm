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
import json
import os
from nanollama.visualization import get_processed_results, jsonl_to_numpy, process_results

LOG_DIR = Path("/private/home/nolte/")

# %% Load processed results in a DataFrame
all_data = {}
for exp in range(1, 3):
    all_data[exp] = []
    log_dir = LOG_DIR / "logs" / f"exp{exp}" / "onfly/metrics"
    for path in log_dir.glob("*"):
        filepath = path / "metrics.json"
        with open(os.path.expandvars(filepath), "r") as f:
          all_data[exp].append(json.loads(f.read()))

dfs = {k: pd.DataFrame(d) for k,d in all_data.items()}
# ------------------------------------------------------------------------------
# Generate plots from the DataFrame
# ------------------------------------------------------------------------------
# %%
dfs[2]

# %% show loss vs entropy
plt.scatter(dfs[1].difficulty_hmm, dfs[1].loss) 
plt.plot([0,4], [0,4])
plt.show()

plt.scatter(dfs[2].difficulty_hmm, dfs[2].loss) 
plt.plot([1,3], [1,3])
plt.show()

# %%
(df.loss - df.difficulty_hmm < 0).any()
# %%



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
# plt.loglog()

# %%
