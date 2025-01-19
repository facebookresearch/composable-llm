"""
Scaling plots for all experiments.

Look at best training/testing loss for a given seed.
Report it against the number of parameters or the number of data.
Plot scaling laws.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from nanollama.utils import flatten_config
from nanollama.visualization import jsonl_to_numpy

HOME_DIR = Path("/private/home/vivc/")


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
        log_dir = HOME_DIR / "logs" / f"exp{exp}" / name

        for task_id in range(nb_tasks):
            task_id += 1
            config_path = log_dir / "tasks" / f"{task_id}.yaml"

            # configuration
            with open(config_path) as f:
                config = flatten_config(yaml.safe_load(f))

            data_seed = config["run_config.data.seed"]

            # data configuration
            onfly_config = True
            try:
                nodes = config["run_config.data.gssm.nodes"]
            except AttributeError:
                nodes = config["run_config.data.path"]
                onfly_config = False

            metric_path = log_dir / "metrics" / str(task_id)

            # scaling parameters
            scaling_variable = None
            if onfly_config:
                # number of parameters when onfly
                filepath = metric_path / "info_model.jsonl"
                scaling_variable = jsonl_to_numpy(filepath, keys=["model_params"])["model_params"][0]
            else:
                # number of data otherwise
                scaling_variable = config["run_config.data.n_data"]

            # training results
            try:
                loss = None
                for rank in range(8):
                    filepath = metric_path / f"raw_{rank}.jsonl"
                    keys = ["loss", "step"]
                    data = jsonl_to_numpy(filepath, keys=keys)
                    rank_loss = data["loss"]
                    if loss is None:
                        loss = rank_loss
                    else:
                        loss += rank_loss
                loss /= 8
                best_train_loss = loss.min()
            except Exception as e:
                print(e, name, task_id)
                best_train_loss = None

            # testing results
            if onfly_config:
                best_test_loss = None
            else:
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

            res[task_id] = {
                "data_seed": data_seed,
                "nodes": nodes,
                "scaling_var": scaling_variable,
                "best_train_loss": best_train_loss,
                "best_test_loss": best_test_loss,
            }
        all_res[name] = res

    return all_res


def extract_useful_info(exp: int, name: str) -> Any:
    log_dir = HOME_DIR / "logs" / f"exp{exp}" / name
    task_id = 1
    config_path = log_dir / "tasks" / f"{task_id}.yaml"
    with open(config_path) as f:
        config = flatten_config(yaml.safe_load(f))

    # get keys
    data_seeds = config.get("launcher.grid.data.seed", None)
    if data_seeds is None:
        data_seeds = [config.get("run_config.data.seed")]
    graphs = config.get("launcher.grid.data.gssm.nodes", None)

    # wether training with finite data
    if "run_config.data.path" in config:
        xkey = "nb_data"
        ykey = "best_test_loss"
    # or with infinite data
    else:
        xkey = "nb_params"
        ykey = "best_train_loss"

    return xkey, ykey, data_seeds, graphs


exp_dict = {
    1: [
        {"names": ["onfly"], "nb_tasks": 234},
    ],
    2: [
        {"names": ["onfly"], "nb_tasks": 72},
    ],
    3: [
        {"names": ["onfly"], "nb_tasks": 72},
    ],
    4: [
        {"names": ["onfly"], "nb_tasks": 72},
    ],
}


for exp in range(4):
    exp += 1
    for exp_config in exp_dict[exp]:
        names = exp_config["names"]
        nb_tasks = exp_config["nb_tasks"]

        all_res = get_statistics(exp, names, nb_tasks, best=True)

        xkey, ykey, data_seeds, all_nodes = extract_useful_info(exp, names[0])

        for data_seed in data_seeds:
            for nodes in all_nodes:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                for i, name in enumerate(names):
                    res = all_res[name]
                    xaxis = []
                    yaxis = []
                    for local_res in res.values():
                        if local_res["nodes"] == nodes and local_res["data_seed"] == data_seed:
                            xaxis.append(local_res["scaling_var"])
                            yaxis.append(local_res[ykey])

                    order = np.argsort(xaxis)
                    yaxis = np.array(yaxis, float)[order]
                    xaxis = np.array(xaxis, float)[order]

                    ind = np.invert(np.isnan(yaxis))

                    if ind.any():
                        yaxis -= yaxis[ind].min() - 1e-4

                    ax.plot(xaxis[ind], yaxis[ind], color=f"C{i}", label=name)

                ax.set_title(f"Data seed: {data_seed}, Nodes: {nodes}")
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
                fig.savefig(save_dir / f"{prefix}data{data_seed}_node{nodes}.png")
                # fig.savefig(save_dir / f"norm_{prefix}data{data_seed}_node{nodes}.png")
