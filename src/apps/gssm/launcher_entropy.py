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
from pathlib import Path
import getpass

import yaml

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

    path = f"{code_dir}/src/apps/gssm/configs/experiment{exp}/entropy.yaml"
    with open(os.path.expandvars(path)) as f:
        config = yaml.safe_load(f)

    all_configs = []
    with open(os.path.expandvars(config.pop("configs_path"))) as f:
        for line in f:
            all_configs.append(json.loads(line))

    nb_tasks = len(all_configs)
    log_dir = Path(os.path.expandvars(config["launcher"]["log_dir"]))
    config_path = str(log_dir / "tasks" / "$SLURM_ARRAY_TASK_ID.yaml")
    run_file = log_dir / "run.sh"

    (log_dir / "tasks").mkdir(parents=True, exist_ok=True)

    for i, conf in enumerate(all_configs):
        config["run_config"]["data"] |= conf["data"]
        config["run_config"]["gssm"] = conf["gssm"]
        config["launcher"]["log_dir"] = str(log_dir / str(i))

        with open(os.path.expandvars(log_dir / "tasks" / f"{i + 1}.yaml"), "w") as f:
            yaml.dump(config, f)

        with open(os.path.expandvars(run_file), "w") as f:
            f.write(
                SBATCH.format(
                    exp=exp,
                    code_dir=code_dir,
                    nb_tasks=nb_tasks,
                    conda_exe=conda_exe,
                    conda_env_path=conda_env,
                    config_path=config_path,
                )
            )

    os.system(f"sbatch {run_file}")


def merge_hmm_estimate(exp: int, code_dir: str) -> None:
    """
    Merge files created by the entropy launcher into single jsonl files.

    This is useful to homogenize the format of the entropy estimates.
    """
    save_path = f"/checkpoint/{getpass.getuser()}/icml/logs/exp{exp}/hmm.jsonl"
    with open(os.path.expandvars(save_path), "w") as f:
        pass

    path = f"{code_dir}/src/apps/gssm/configs/experiment{exp}/entropy.yaml"
    with open(os.path.expandvars(path)) as f:
        config = yaml.safe_load(f)

    all_configs = []
    with open(os.path.expandvars(config.pop("configs_path"))) as f:
        for line in f:
            all_configs.append(json.loads(line))

    nb_tasks = len(all_configs)
    log_dir = Path(os.path.expandvars(config["launcher"]["log_dir"]))

    # for all data files
    for i in range(nb_tasks):
        # retrieve entropy estimates
        metric_dir = log_dir / str(i)
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
            print(json.dumps({"grid_id": i, "hmm_difficulty": loss}), file=f, flush=True)


if __name__ == "__main__":
    code_dir = "$CODE_DIR"
    exp = int(input("which experiment? "))
    launch_entropy_estimate(exp, code_dir)
