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


def get_statistics(exp: int, name: str, nb_tasks: int) -> dict[str, dict[str, Any]]:
    """
    Get the statistics for the given experiments and names.

    Parameters
    ----------
    exp:
        The experiment number.
    name:
        The name of the experiments.
    nb_tasks:
        The number of tasks (TODO: find it automatically).

    Returns
    -------
    The experiments statistics, organized by name, task_id, and statistics.
    """
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
            best_test_loss = loss.min()

        res[task_id] = {
            "data_seed": data_seed,
            "nodes": nodes,
            "scaling_var": scaling_variable,
            "best_train_loss": best_train_loss,
            "best_test_loss": best_test_loss,
        }

    return res


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


exp_list = [
    {"exp": 1, "name": "onfly", "nb_tasks": 234},
    {"exp": 2, "name": "onfly", "nb_tasks": 72},
    {"exp": 3, "name": "onfly", "nb_tasks": 72},
    {"exp": 4, "name": "onfly", "nb_tasks": 72},
]

for exp_config in exp_list:
    exp = exp_config["exp"]
    name = exp_config["name"]
    nb_tasks = exp_config["nb_tasks"]

    res = get_statistics(exp, name, nb_tasks)

    xkey, ykey, data_seeds, all_nodes = extract_useful_info(exp, name)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for i, nodes in enumerate(all_nodes):
        xaxis, yaxis, nb = None, None, None
        # average losses over seeds
        for data_seed in data_seeds:
            local_x, local_y = [], []
            for local_res in res.values():
                if local_res["nodes"] == nodes and local_res["data_seed"] == data_seed:
                    local_x.append(local_res["scaling_var"])
                    local_y.append(local_res[ykey])

            order = np.argsort(local_x)
            local_y = np.array(local_y, float)[order]

            # local_y -= local_y.min() - 1e-4  # fake entropy removal
            if xaxis is None:
                xaxis = np.array(local_x, float)[order]
                yaxis = np.array(local_y, float)[order]
                nb = 0
            else:
                assert (xaxis == np.array(local_x, float)[order]).all()
                yaxis += local_y
            nb += 1

        yaxis /= nb

        ax.plot(xaxis, yaxis, color=f"C{i}", label=str(nodes))

    ax.set_title(f"Exp {exp}")
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
    fig.savefig(save_dir / f"{prefix}exp{exp}.png")
