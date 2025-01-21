import json
import os
import subprocess
from pathlib import Path

import yaml

SBATCH = """#!/bin/bash

# Logging configuration
#SBATCH --job-name=entropy1
#SBATCH --output=/private/home/%u/logs/exp{exp}/entropy/logs/%a.log
#SBATCH --error=/private/home/%u/logs/exp{exp}/entropy/logs/%a.err
#SBATCH --open-mode=append

# Job specification
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH --array=1-{nb_tasks}

# activate conda environment
eval "$({conda_exe} shell.bash hook)"
conda activate {conda_env_path}

# go to code directory
export PATH_TO_CODE_DIR={code_dir}
cd $PATH_TO_CODE_DIR

python -m src.apps.gssm.entropy {config_path}
"""


def launch_entropy_estimate(exp: int, code_dir: str) -> None:
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    python_env = subprocess.check_output("which python", shell=True).decode("ascii").strip()
    conda_env = str(Path(python_env).parent.parent)

    path = f"{code_dir}src/apps/gssm/configs/experiment{exp}/entropy.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)

    all_configs = []
    with open(config.pop("configs_path")) as f:
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

        config_path = log_dir / "tasks" / f"{i + 1}.yaml"

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with open(run_file, "w") as f:
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


if __name__ == "__main__":
    code_dir = "/private/home/vivc/code/composable-llm/"
    for exp in range(1, 5):
        launch_entropy_estimate(exp, code_dir)
