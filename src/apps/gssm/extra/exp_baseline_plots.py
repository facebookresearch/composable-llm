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

from nanollama.visualization import get_processed_results, jsonl_to_numpy, process_results

import os

import pickle


# name = "params"
# scaling_key = "nb_params"
# keys = ["grid_id"]

# %% Process results
num_keys = []
steps = [100, 300, 1000, 3000]

def get_data(LOG_DIR, process_data=False, exp_post_fix=''):
    name = "data"
    scaling_key = "data.n_data"
    keys = [scaling_key, "grid_id"]

    CODE_DIR = Path("/private/home/vivc/code/composable-llm/")
    if process_data:
        for exp in range(1, 5):
            log_dir = LOG_DIR / "logs" / f"exp{exp}{exp_post_fix}" / name
            process_results(log_dir, keys, num_keys, steps, eval=True, copy_num=True)

    # %% Load processed results in a DataFrame
    all_data = []
    for exp in range(1, 5):
        log_dir = LOG_DIR / "logs" / f"exp{exp}{exp_post_fix}" / name
        data = get_processed_results(log_dir)

        # retrieve corresponding entropy esimates
        for key in ["gzip"]:#, "hmm"]:
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
    return all_data


# collect results
tf_data = get_data(Path("/checkpoint/vivc/icml/"))  
lstm_data = get_data(Path("/private/home/jianyuzhang/composable-llm/"), process_data=True, exp_post_fix='_minlstm')


# save 

with open('/private/home/jianyuzhang/composable-llm/lstm_data.pkl', 'wb') as file:       
    pickle.dump(lstm_data, file) 

with open('/private/home/jianyuzhang/composable-llm/tf_data.pkl', 'wb') as file:       
    pickle.dump(tf_data, file) 


# reload to plot

with open('/private/home/jianyuzhang/composable-llm/lstm_data.pkl', 'rb') as file:       
    lstm_data = pickle.load( file) 

with open('/private/home/jianyuzhang/composable-llm/tf_data.pkl', 'rb') as file:       
    tf_data = pickle.load(file) 
# ------------------------------------------------------------------------------
# Generate plots from the DataFrame
# ------------------------------------------------------------------------------

# %% show loss vs entropy

#difficulty_key = "hmm_difficulty"

def plot(all_data):
    difficulty_key = "gzip_difficulty"
    scaling_key = "data.n_data"
    keys = [scaling_key, "grid_id"]


    for exp, data in enumerate(all_data):
        plt.figure()
        # make a line y = x
        # plt.plot([0, 3], [0, 3], color="black", ls=":")
        all_scales = data[scaling_key].unique()[-1:]
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
        # plt.loglog()
        # plt.legend()
        plt.savefig("tmp.pdf")
        os.system("imgcat tmp.pdf --height 32")

def plot_two_models(all_data1, all_data2):
    difficulty_key = "gzip_difficulty"
    scaling_key = "data.n_data"
    keys = [scaling_key, "grid_id"]

    print(len(all_data1), len(all_data2))

    plt.figure(figsize=(12,12))
    #for exp, data in enumerate(all_data):
    for exp in range(len(all_data1)):
        plt.subplot(2,2,exp+1)


        data = all_data1[exp]
        # make a line y = x
        # plt.plot([0, 3], [0, 3], color="black", ls=":")
        all_scales = data[scaling_key].unique()[-1:]
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

        data = all_data2[exp]
        # make a line y = x
        # plt.plot([0, 3], [0, 3], color="black", ls=":")
        all_scales = data[scaling_key].unique()[-1:]
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
                    marker="x",
                )
  
        plt.xlabel(difficulty_key)
        plt.ylabel("test loss")
        plt.title(f"exp={exp + 1}")
    # plt.loglog()
    # plt.legend()
    plt.tight_layout()
    plt.savefig("tmp.pdf")
    os.system("imgcat tmp.pdf --height 64")


plot_two_models(tf_data, lstm_data)
