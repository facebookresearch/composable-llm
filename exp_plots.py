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
import numpy as np

from nanollama.visualization import get_processed_results, jsonl_to_numpy, process_results, read_indented_jsonl
from nanollama.utils import flatten_config

LOG_DIR = Path("/checkpoint/nolte/icml/")
CODE_DIR = Path("/private/home/nolte/projects/composable-llm/")
# name = "data"
# scaling_key = "data.n_data"
name = "params"
scaling_key = "model.emb_dim"
keys = [scaling_key, "grid_id"]

data_range = [16]

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
# %%

def plot_for_exp12():
  """
  this is a plot for compositionality
  """
  difficulty_key = "hmm_difficulty"

  scaling_key = "nodes.Z1.state_dim"
  color_key = "nodes.Z1.state_dim"

  def fix_label(key):
      if key == "nodes.Z1.state_dim":
          return "$|Z_i|=$"

  for exp, data in zip(data_range, all_data):
      plt.figure()
      selection = data["nodes.X.alpha"].isin(['1e-2', '1e-1'])
      selection &= data["data.n_data"].isin([100_000])
      selection &= data["nodes.X.alpha"].isin(['1e-1'])
      print(selection.sum())
      # make a line y = x
      # plt.plot([2.5, 5], [2.5, 5], color="black", ls=":")
      all_graph = data[color_key].unique()
      for i, scaling_val in enumerate(all_graph):
          data1 = data[(data[color_key] == scaling_val) & selection]
          plt.scatter(
              data1[difficulty_key],
              data1["best"] - data1[difficulty_key],
              label=f"{fix_label(scaling_key)} {scaling_val}",
              color=f"C{i}",
              # alpha=alpha,
          )
      plt.xlabel("Entropy", fontsize=14)
      plt.ylabel("KL Divergence", fontsize=14)
      plt.yscale('log')
      plt.legend(loc='lower right', fontsize=12)
      plt.tight_layout()
      plt.savefig(f"figures/exp{exp}_kl_vs_entropy.pdf")
      # plt.loglog()

plot_for_exp12()

# %%
def plot_for_exp13_nb_data():
  """
  plot for computational hardness
  """
  difficulty_key = "hmm_difficulty"
  color_key = "nodes.Z1.state_dim"
  alpha_key = "data.n_data"

  def fix_label(key):
      if key == "nodes.Z1.state_dim":
          return "|$Z$|="

  for exp, data in zip(data_range, all_data):
      plt.figure()
      selection = data["nodes.X.alpha"].isin(['1e-2', '2e-3'])
      # selection &= data["nodes.Z1.alpha"].isin(['1e-2', '2e-3'])
      selection &= data["nodes.X.state_dim"].isin([128])
      # selection &= data["data.n_data"].isin([100_000])
      # make a line y = x
      # plt.plot([2.5, 5], [2.5, 5], color="black", ls=":")
      alpha_scales = data[alpha_key].unique()[[0,1,2,4]]
      print(alpha_scales)
      all_graph = data[color_key].unique()[[0,1,2]]
      for alpha_val in alpha_scales:
        for i, color_val in enumerate(all_graph):
            data1 = data[(data[color_key] == color_val) & selection & (data[alpha_key] == alpha_val)]
            plt.scatter(
                data1[difficulty_key],
                data1["best"] - data1[difficulty_key],
                label=f"{fix_label(color_key)} {color_val}, #data $10^{int(np.log10(alpha_val))}$",
                color=f"C{i}",
                alpha=alpha_val**.3/max(alpha_scales)**.3,
            )
      plt.xlabel("Entropy", fontsize=14)
      plt.ylabel("KL Divergence", fontsize=14)
      plt.yscale('log')
      plt.legend(loc='best', bbox_to_anchor=(1.01, 1.02), fontsize=12)
      plt.tight_layout()
      plt.savefig(f"figures/exp{exp}_kl_vs_entropy_nb_data.pdf")

plot_for_exp13_nb_data()

# %%
def plot_for_exp13_nb_params():
  """
  plot for computational hardness
  """
  difficulty_key = "hmm_difficulty"
  color_key = "nodes.Z1.state_dim"
  alpha_key = "model.emb_dim"

  def fix_label(key):
      if key == "nodes.Z1.state_dim":
          return "|$Z$|="

  for exp, data in zip(data_range, all_data):
      plt.figure()
      selection = data["nodes.X.alpha"].isin(['1e-2', '2e-3'])
      # selection &= data["nodes.Z1.alpha"].isin(['1e-2', '2e-3'])
      selection &= data["nodes.X.state_dim"].isin([128])
      # make a line y = x
      # plt.plot([2.5, 5], [2.5, 5], color="black", ls=":")
      alpha_scales = data[alpha_key].unique()[2:]
      print(alpha_scales)
      all_graph = data[color_key].unique()[[0,1,2]]
      for alpha_val in alpha_scales:
        for i, color_val in enumerate(all_graph):
            data1 = data[(data[color_key] == color_val) & selection & (data[alpha_key] == alpha_val)]
            plt.scatter(
                data1[difficulty_key],
                data1["best"] - data1[difficulty_key],
                label=f"{fix_label(color_key)} {color_val}, model $d = {alpha_val}$",
                color=f"C{i}",
                alpha=alpha_val/max(alpha_scales),
            )
      plt.xlabel("Entropy", fontsize=14)
      plt.ylabel("KL Divergence", fontsize=14)
      plt.yscale('log')
      plt.legend(loc='best', bbox_to_anchor=(1.01, 1.02), fontsize=12)
      plt.tight_layout()
      plt.savefig(f"figures/exp{exp}_kl_vs_entropy_nb_params.pdf")

plot_for_exp13_nb_params()

# %%

def plot_for_exp14_nb_data():
  """
  plot for graph sparsity
  """
  difficulty_key = "hmm_difficulty"
  color_key = "nodes.Z3.parents"
  alpha_key = "data.n_data"

  def fix_legend(key,val):
      if key == "nodes.Z3.parents":
        if len(val) == 2:
          return "dense"
        else:
          return "sparse"

  for exp, data in zip(data_range, all_data):
      plt.figure()
      # selection = data["nodes.X.alpha"].isin(['1e-2', '2e-3'])
      # selection &= data["nodes.Z1.alpha"].isin(['1e-2', '2e-3'])
      selection = data["nodes.X.state_dim"].isin([128])
      # selection &= data["nodes.Z1.alpha"].isin(['1e-1', '2e-1'])
      # selection &= data["data.n_data"].isin([100_000])
      # make a line y = x
      # plt.plot([2.5, 5], [2.5, 5], color="black", ls=":")
      alpha_scales = data[alpha_key].unique()[[1,2,4]]
      print(alpha_scales)
      all_graph = data[color_key].apply(tuple).unique()[::-1]
      for alpha_val in alpha_scales:
        for i, color_val in enumerate(all_graph):
            # data1 = data[(data[color_key] == color_val) & selection & (data[alpha_key] == alpha_val)]
            data1 = data[(data[color_key].apply(tuple) == color_val) & selection & (data[alpha_key] == alpha_val)]
            plt.scatter(
                data1[difficulty_key],
                data1["best"] - data1[difficulty_key],
                label=f"{fix_legend(color_key, color_val)}, #data $10^{int(np.log10(alpha_val))}$",
                color=f"C{i}",
                alpha=alpha_val**.3/max(alpha_scales)**.3,
            )
      plt.xlabel("Entropy", fontsize=14)
      plt.ylabel("KL Divergence", fontsize=14)
      plt.yscale('log')
      plt.legend(loc='best', bbox_to_anchor=(1.01, 1.02), fontsize=12)
      plt.tight_layout()
      plt.savefig(f"figures/exp{exp}_kl_vs_entropy_nb_data.pdf")

plot_for_exp14_nb_data()# %%

# %%

def plot_for_exp14_nb_params():
  """
  plot for graph sparsity
  """
  difficulty_key = "hmm_difficulty"
  color_key = "nodes.Z3.parents"
  alpha_key = "model.emb_dim"

  def fix_legend(key,val):
      if key == "nodes.Z3.parents":
        if len(val) == 2:
          return "dense"
        else:
          return "sparse"

  for exp, data in zip(data_range, all_data):
      plt.figure()
      # selection = data["nodes.X.alpha"].isin(['1e-2', '2e-3'])
      # selection &= data["nodes.Z1.alpha"].isin(['1e-2', '2e-3'])
      selection = data["nodes.X.state_dim"].isin([128])
      # selection &= data["nodes.Z1.alpha"].isin(['1e-1', '2e-1'])
      # selection &= data["data.n_data"].isin([100_000])
      # make a line y = x
      # plt.plot([2.5, 5], [2.5, 5], color="black", ls=":")
      alpha_scales = data[alpha_key].unique()[[-5,-3, -1]]
      print(alpha_scales)
      all_graph = data[color_key].apply(tuple).unique()[[1,0]] # turn colors around
      for alpha_val in alpha_scales:
        for i, color_val in enumerate(all_graph):
            # data1 = data[(data[color_key] == color_val) & selection & (data[alpha_key] == alpha_val)]
            data1 = data[(data[color_key].apply(tuple) == color_val) & selection & (data[alpha_key] == alpha_val)]
            plt.scatter(
                data1[difficulty_key],
                data1["best"] - data1[difficulty_key],
                label=f"{fix_legend(color_key, color_val)}, model $d = {alpha_val}$",
                color=f"C{i}",
                alpha=alpha_val**.7/max(alpha_scales)**.7,
            )
      plt.xlabel("Entropy", fontsize=14)
      plt.ylabel("KL Divergence", fontsize=14)
      plt.yscale('log')
      plt.legend(loc='best', bbox_to_anchor=(1.01, 1.02), fontsize=12)
      plt.tight_layout()
      plt.savefig(f"figures/exp{exp}_kl_vs_entropy_nb_params.pdf")

plot_for_exp14_nb_params()# %%

# %%

def plot_for_exp16_nb_data():
  """
  plot for graph sparsity
  """
  difficulty_key = "hmm_difficulty"
  color_key = "nodes.Z1.mode"
  alpha_key = "data.n_data"

  def fix_legend(key,val):
      if key == "nodes.Z1.mode":
        return val.replace("dead", "inactive")

  for exp, data in zip(data_range, all_data):
      plt.figure()
      # selection = data["nodes.X.alpha"].isin(['1e-2', '2e-3'])
      # selection &= data["nodes.Z1.alpha"].isin(['1e-2', '2e-3'])
      selection = data["nodes.X.state_dim"].isin([32])
      # selection &= ((data["nodes.Z1.alpha"].apply(float) > 3.2e-2) | (data[color_key] == "default"))
      # selection &= data["data.n_data"].isin([100_000])
      # make a line y = x
      # plt.plot([2.5, 5], [2.5, 5], color="black", ls=":")
      alpha_scales = data[alpha_key].unique()[[1,2,4]]
      print(alpha_scales)
      all_graph = data[color_key].unique()
      for alpha_val in alpha_scales:
        for i, color_val in enumerate(all_graph):
            # data1 = data[(data[color_key] == color_val) & selection & (data[alpha_key] == alpha_val)]
            data1 = data[(data[color_key] == color_val) & selection & (data[alpha_key] == alpha_val)]
            plt.scatter(
                data1[difficulty_key],
                data1["best"] - data1[difficulty_key],
                label=f"{fix_legend(color_key, color_val)}, #data $10^{int(np.log10(alpha_val))}$",
                color=f"C{i}",
                alpha=alpha_val**.3/max(alpha_scales)**.3,
            )
      plt.xlabel("Entropy", fontsize=14)
      plt.ylabel("KL Divergence", fontsize=14)
      plt.yscale('log')
      plt.legend(loc='best', bbox_to_anchor=(1.01, 1.02), fontsize=12)
      plt.tight_layout()
      plt.savefig(f"figures/exp{exp}_kl_vs_entropy_nb_data.pdf")

plot_for_exp16_nb_data()

# %%

def plot_for_exp16_nb_params():
  """
  plot for graph sparsity
  """
  difficulty_key = "hmm_difficulty"
  color_key = "nodes.Z1.mode"
  alpha_key = "model.emb_dim"

  def fix_legend(key,val):
      if key == "nodes.Z1.mode":
        return val.replace("dead", "inactive")

  for exp, data in zip(data_range, all_data):
      plt.figure()
      # selection = data["nodes.X.alpha"].isin(['1e-2', '2e-3'])
      # selection &= data["nodes.Z1.alpha"].isin(['1e-2', '2e-3'])
      selection = data["nodes.X.state_dim"].isin([32])
      # selection &= data["nodes.Z1.alpha"].isin(['1e-1', '2e-1'])
      # selection &= data["data.n_data"].isin([100_000])
      # make a line y = x
      # plt.plot([2.5, 5], [2.5, 5], color="black", ls=":")
      alpha_scales = data[alpha_key].unique()[[-5,-3,-1]]
      print(alpha_scales)
      all_graph = data[color_key].unique()
      for alpha_val in alpha_scales:
        for i, color_val in enumerate(all_graph):
            # data1 = data[(data[color_key] == color_val) & selection & (data[alpha_key] == alpha_val)]
            data1 = data[(data[color_key] == color_val) & selection & (data[alpha_key] == alpha_val)]
            plt.scatter(
                data1[difficulty_key],
                data1["best"] - data1[difficulty_key],
                label=f"{fix_legend(color_key, color_val)}, model $d = {alpha_val}$",
                color=f"C{i}",
                alpha=alpha_val**.7/max(alpha_scales)**.7,
            )
      plt.xlabel("Entropy", fontsize=14)
      plt.ylabel("KL Divergence", fontsize=14)
      plt.yscale('log')
      plt.legend(loc='best', bbox_to_anchor=(1.01, 1.02), fontsize=12)
      plt.tight_layout()
      plt.savefig(f"figures/exp{exp}_kl_vs_entropy_nb_params.pdf")

plot_for_exp16_nb_params()# %%

# %%
