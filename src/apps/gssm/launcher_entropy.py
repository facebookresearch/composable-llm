"""
Estimate entropy by sampling log-likelihoods

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import json
import os
import subprocess
import zlib
from pathlib import Path
from typing import Any

import h5py
import yaml

# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------


def read_indented_jsonl(filepath: str) -> list[dict[str, Any]]:
    data = []
    with open(filepath) as file:
        content = file.read()

    # split the content into individual JSON objects
    json_objects = content.split("}\n{")

    # adjust format
    if json_objects:
        json_objects[0] = json_objects[0] + "}"
        json_objects[-1] = "{" + json_objects[-1]
        for i in range(1, len(json_objects) - 1):
            json_objects[i] = "{" + json_objects[i] + "}"

    # parse each JSON object
    for json_str in json_objects:
        json_object = json.loads(json_str)
        data.append(json_object)
    return data


# ------------------------------------------------------------------------------
# Slurm launcher
# ------------------------------------------------------------------------------


SBATCH = """#!/bin/bash

# Logging configuration
#SBATCH --job-name=entropy{exp}
#SBATCH --output=/private/home/%u/logs/exp{exp}/entropy/logs/%a.log
#SBATCH --error=/private/home/%u/logs/exp{exp}/entropy/logs/%a.err
#SBATCH --open-mode=append

# Job specification
#SBATCH --partition=scavenge
#SBATCH --constraint=volta32gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --array=1-{nb_tasks}

# activate conda environment
eval "$({conda_exe} shell.bash hook)"
conda activate {conda_env_path}

# go to code directory
cd $CODE_DIR

python -m src.apps.gssm.entropy {config_path}
"""


def launch_entropy_estimate(exp: int, code_dir: str) -> None:
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    python_env = subprocess.check_output("which python", shell=True).decode("ascii").strip()
    conda_env = str(Path(python_env).parent.parent)

    code_dir = Path(code_dir)

    path = code_dir / f"src/apps/gssm/configs/experiment{exp}/entropy.yaml"
    with open(os.path.expandvars(path)) as f:
        config = yaml.safe_load(f)

    all_configs = []
    with open(code_dir / f"src/apps/gssm/configs/experiment{exp}/.gssm_id_path.jsonl") as f:
        for line in f:
            all_configs.append(json.loads(line))

    all_gssm = read_indented_jsonl(code_dir / f"src/apps/gssm/configs/experiment{exp}/.gssm_id_config.jsonl")

    nb_tasks = len(all_configs)
    log_dir = Path(os.path.expandvars(config["launcher"]["log_dir"]))
    config_path = str(log_dir / "tasks" / "$SLURM_ARRAY_TASK_ID.yaml")
    run_file = log_dir / "run.sh"

    (log_dir / "tasks").mkdir(parents=True, exist_ok=True)

    for i, conf in enumerate(all_configs):
        config["run_config"]["data"]["path"] = conf["path"]

        gssm_config = all_gssm[conf["gssm_id"]]
        assert gssm_config["gssm_id"] == conf["gssm_id"]
        gssm_config["batch_size"] = "FAKE"
        gssm_config["seq_len"] = "FAKE"
        gssm_config["seed"] = conf["seed"]

        config["run_config"]["gssm"] = gssm_config
        config["launcher"]["log_dir"] = str(log_dir / str(i))

        with open(log_dir / "tasks" / f"{i + 1}.yaml", "w") as f:
            yaml.dump(config, f)

        with open(run_file, "w") as f:
            f.write(
                SBATCH.format(
                    exp=exp,
                    code_dir=str(code_dir),
                    nb_tasks=nb_tasks,
                    conda_exe=conda_exe,
                    conda_env_path=conda_env,
                    config_path=config_path,
                )
            )

    os.system(f"sbatch {run_file}")


# ------------------------------------------------------------------------------
# Merging estimates
# ------------------------------------------------------------------------------


def merge_hmm_estimate(exp: int, code_dir: str) -> None:
    """
    Merge files created by the entropy launcher into single jsonl files.

    This is useful to homogenize the format of the entropy estimates.
    """
    save_path = os.path.expandvars(f"/checkpoint/$USER/icml/logs/exp{exp}/hmm.jsonl")
    with open(save_path, "w") as f:
        pass

    code_dir = Path(os.path.expandvars(code_dir))
    path = code_dir / f"src/apps/gssm/configs/experiment{exp}/entropy.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)

    all_configs = []
    with open(code_dir / f"src/apps/gssm/configs/experiment{exp}/.gssm_id_path.jsonl") as f:
        for line in f:
            all_configs.append(json.loads(line))

    log_dir = Path(os.path.expandvars(config["launcher"]["log_dir"]))

    # for all data files
    for conf in all_configs:
        # retrieve entropy estimates
        metric_dir = log_dir / str(conf["grid_id"])
        loss = None
        num = 0
        for file in metric_dir.glob("eval_*.jsonl"):
            with open(os.path.expandvars(file)) as f:
                if loss is None:
                    loss = json.load(f)["loss"]
                else:
                    loss += json.load(f)["loss"]
                num += 1
            loss /= num

        # save it to a jsonl file
        with open(os.path.expandvars(save_path), "a") as f:
            print(json.dumps({"grid_id": conf["grid_id"], "hmm_difficulty": loss}), file=f, flush=True)


# ------------------------------------------------------------------------------
# Gzip estimate
# ------------------------------------------------------------------------------


def gzip_estimate(exp: int, code_dir: str) -> None:
    code_dir = Path(os.path.expandvars(code_dir))

    save_path = os.path.expandvars(f"/checkpoint/$USER/icml/logs/exp{exp}/gzip.jsonl")
    with open(os.path.expandvars(save_path), "w") as f:
        pass

    all_configs = []
    with open(code_dir / f"src/apps/gssm/configs/experiment{exp}/.gssm_id_path.jsonl") as f:
        for line in f:
            all_configs.append(json.loads(line))

    for conf in all_configs:
        path = os.path.expandvars(conf["path"])
        with h5py.File(path) as f:
            data = f["data"][:, 1:]

        entropy = len(zlib.compress(data.tobytes(), level=9)) / data.size
        with open(os.path.expandvars(save_path), "a") as f:
            print(json.dumps({"grid_id": conf["grid_id"], "gzip_difficulty": entropy}), file=f, flush=True)


if __name__ == "__main__":
    code_dir = os.path.expandvars("$CODE_DIR")

    if input("Entropy calculation? (Y/N))").lower()[0] != "y":
        exp = int(input("Which experiment? "))
        print(f"Entropy calculation launch for experiment {exp}")
        launch_entropy_estimate(exp, code_dir)

    if input("gzip calculation? (Y/N))").lower()[0] != "y":
        exp = int(input("Which experiment? "))
        print(f"Running experiment {exp}")
        gzip_estimate(exp, code_dir)

    # merge the hmm estimate in the same format
    if input("merging hmm calculation? (Y/N))").lower()[0] != "y":
        exp = int(input("Which experiment? "))
        print(f"Merging hmm estimate for experiment {exp}")
        merge_hmm_estimate(exp, code_dir)
