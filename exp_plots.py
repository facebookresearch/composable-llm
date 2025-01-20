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

from nanollama.visualization import extract_config_info, get_losses

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
    keys = ["model.emb_dim", "data.seed"]
    num_keys = ["data.gssm.nodes"]
    steps = [30, 100, 300, 1000]

    for task_id in range(nb_tasks):
        task_id += 1

        log_dir = HOME_DIR / "logs" / f"exp{exp}" / name
        metric_path = log_dir / "metrics" / str(task_id)

        local_res = extract_config_info(log_dir, task_id, keys, num_keys)
        local_res |= get_losses(metric_path, steps, eval=False)

        res[task_id] = local_res
    return res


exp_list = [
    # {"exp": 1, "name": "onfly", "nb_tasks": 234},
    {"exp": 2, "name": "onfly", "nb_tasks": 72},
    {"exp": 3, "name": "onfly", "nb_tasks": 72},
    {"exp": 4, "name": "onfly", "nb_tasks": 72},
]

for exp_config in exp_list:
    exp = exp_config["exp"]
    name = exp_config["name"]
    nb_tasks = exp_config["nb_tasks"]

    res = get_statistics(exp, name, nb_tasks)
    xkey = "nb_params"
    ykey = "best"

    node_configs = np.unique([local_res["data.gssm.nodes"] for local_res in res.values()]).tolist()
    data_seeds = np.unique([local_res["data.seed"] for local_res in res.values()]).tolist()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for i, node in enumerate(node_configs):
        xaxis: np.ndarray = None
        yaxis: np.ndarray = None
        nb = 0
        # average losses over seeds
        for data_seed in data_seeds:
            local_x, local_y = [], []
            for local_res in res.values():
                if local_res["data.gssm.nodes"] == node and local_res["data.seed"] == data_seed:
                    local_x.append(local_res[xkey])
                    local_y.append(local_res[ykey])

            order = np.argsort(local_x)
            local_y = np.array(local_y, float)[order]

            # local_y -= local_y.min() - 1e-4  # fake entropy removal
            if xaxis is None:
                xaxis = np.array(local_x, float)[order]
                yaxis = np.array(local_y, float)[order]
            else:
                assert (xaxis == np.array(local_x, float)[order]).all()
                yaxis += local_y
            nb += 1

        yaxis /= nb

        ax.plot(xaxis, yaxis, color=f"C{i}", label=node)

    ax.set_title(f"Data seed: {data_seed}, Nodes: {node}")
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
    fig.savefig(save_dir / f"{prefix}data{data_seed}_node{node}.png")
