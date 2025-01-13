"""
Do the automatic parsing of the grid

Do automatic parsing of wether or not we are in the infinite data regime.


- Scaling plots for all experiments.
     Look at best training/testing loss for a given seed.
     Report it against the number of parameters or the number of data.
     Do the scaling.
     Average across seeds

Read the config to get the seed
"""

# %%

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from nanollama.utils import flatten_config
from nanollama.visualization import jsonl_to_numpy


def get_statistics(exp: int, names: list[str], nb_tasks: int, best: bool) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Get the statistics for the given experiments and names.

    Parameters
    ----------
    exp:
        The experiment number.
    names:
        The names of the experiments.
    nb_tasks:
        The number of tasks.
    best:
        Whether to take the best loss values.

    Returns
    -------
    The experiments statistics, organized by name, task_id, and statistics.
    """
    all_res = {}
    for name in names:
        res = {}
        log_dir = Path.home() / "logs" / f"exp{exp}" / name

        for task_id in range(nb_tasks):
            task_id += 1
            config_path = log_dir / "tasks" / f"{task_id}.yaml"

            # configuration
            with open(config_path) as f:
                config = flatten_config(yaml.safe_load(f))
            data_seed = config["run_config.data.seed"]
            model_seed = config["run_config.orchestration.utils.seed"]
            try:
                nb_data = config["run_config.data.n_data"]
            except KeyError:
                nb_data = None

            metric_path = log_dir / "metrics" / str(task_id)

            # number of parameters
            for filepath in metric_path.glob("info_model.jsonl"):
                nb_params = jsonl_to_numpy(filepath, keys=["model_params"])["model_params"][0]

            # training results
            try:
                loss = None
                for rank in range(8):
                    filepath = metric_path / f"raw_{rank}.jsonl"
                    keys = ["loss"]
                    rank_loss = jsonl_to_numpy(filepath, keys=keys)["loss"]
                    if loss is None:
                        loss = rank_loss
                    else:
                        loss += rank_loss
                loss /= 8
                best_train_loss = loss.min()
            except (FileNotFoundError, ValueError) as e:
                print(e)
                best_train_loss = None

            # testing results
            try:
                loss = None
                for rank in range(8):
                    filepath = metric_path / f"eval_{rank}.jsonl"
                    keys = ["loss"]
                    rank_loss = jsonl_to_numpy(filepath, keys=keys)["loss"]
                    if loss is None:
                        loss = rank_loss
                    else:
                        loss += rank_loss
                loss /= 8
                if best:
                    best_test_loss = loss.min()
            except (FileNotFoundError, ValueError):
                best_test_loss = None

            res[task_id] = {
                "data_seed": data_seed,
                "model_seed": model_seed,
                "nb_data": nb_data,
                "nb_params": nb_params,
                "best_train_loss": best_train_loss,
                "best_test_loss": best_test_loss,
            }
        all_res[name] = res

    return all_res


def extract_useful_info(exp: int, name: str) -> Any:
    log_dir = Path.home() / "logs" / f"exp{exp}" / name
    task_id = 2
    config_path = log_dir / "tasks" / f"{task_id}.yaml"
    with open(config_path) as f:
        config = flatten_config(yaml.safe_load(f))

    # get keys
    data_seeds = config.get("launcher.grid.data.seed", None)
    if data_seeds is None:
        data_seeds = [config.get("run_config.data.seed")]
    model_seeds = config.get("launcher.grid.orchestration.utils.seed", None)
    if model_seeds is None:
        model_seeds = [config.get("run_config.orchestration.utils.seed")]

    # wether training with finite data
    if "run_config.data.path" in config:
        xkey = "nb_data"
        ykey = "best_test_loss"
    # or with infinite data
    else:
        xkey = "nb_params"
        ykey = "best_train_loss"

    return xkey, ykey, data_seeds, model_seeds


# %%

exp_dict = {
    1: [
        {"names": ["small_X", "small_Z"], "nb_tasks": 72},
        {"names": ["onfly_small_X", "onfly_small_Z"], "nb_tasks": 64},
    ],
    2: [
        {"names": ["one_node", "two_nodes", "four_nodes", "eight_nodes"], "nb_tasks": 72},
        {"names": ["onfly_one_node", "onfly_two_nodes", "onfly_four_nodes", "onfly_eight_nodes"], "nb_tasks": 64},
    ],
    3: [
        {"names": ["easy", "medium", "hard", "dense"], "nb_tasks": 72},
        {"names": ["onfly_easy", "onfly_medium", "onfly_hard", "onfly_dense"], "nb_tasks": 64},
    ],
    4: [
        {"names": ["base", "slow", "dead", "context"], "nb_tasks": 72},
        {"names": ["onfly_base", "onfly_slow", "onfly_dead", "onfly_context"], "nb_tasks": 64},
    ],
}

for exp in range(4):
    exp += 1
    for exp_config in exp_dict[exp]:
        names = exp_config["names"]
        nb_tasks = exp_config["nb_tasks"]

        all_res = get_statistics(exp, names, nb_tasks, best=True)

        xkey, ykey, data_seeds, model_seeds = extract_useful_info(exp, names[0])

        for data_seed in data_seeds:
            for model_seed in model_seeds:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                for i, name in enumerate(names):
                    res = all_res[name]
                    xaxis = []
                    yaxis = []
                    for local_res in res.values():
                        if local_res["model_seed"] == model_seed and local_res["data_seed"] == data_seed:
                            xaxis.append(local_res[xkey])
                            yaxis.append(local_res[ykey])

                    order = np.argsort(xaxis)
                    yaxis = np.array(yaxis, float)[order]
                    xaxis = np.array(xaxis, float)[order]

                    ind = np.invert(np.isnan(yaxis))

                    # if ind.any():
                    #     yaxis -= yaxis[ind].min()

                    ax.plot(xaxis[ind], yaxis[ind], color=f"C{i}", label=name)

                ax.set_title(f"Data seed: {data_seed}, Model seed: {model_seed}")
                ax.set_xlabel(xkey)
                ax.set_ylabel(ykey)
                ax.legend()
                ax.loglog()

                save_dir = Path("savings") / f"exp{exp}"
                save_dir.mkdir(exist_ok=True, parents=True)

                if name.startswith("onfly"):
                    prefix = "onfly_"
                else:
                    prefix = ""
                fig.savefig(save_dir / f"{prefix}data{data_seed}_model{model_seed}.png")
                # fig.savefig(save_dir / f"norm_{prefix}data{data_seed}_model{model_seed}.png")

# %%
