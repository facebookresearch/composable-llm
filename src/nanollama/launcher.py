"""
Script to launch jobs on a Slurm cluster.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import itertools
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import initialize_nested_dataclass

# -------------------------------------------------------------------------------
# Configuration Class
# -------------------------------------------------------------------------------


@dataclass
class SlurmConfig:
    # basic configuration
    partition: str = ""
    nodes: int = 1  # number of nodes to run the job on.
    nb_gpus: int = 1  # number of GPUs required per node.
    nb_cpus: int = 16  # number of CPUs allocated per GPU.
    mem: str = ""  # amount of memory to allocate per node.
    time: int = -1  # time limit of the job (in minutes).

    # time between USR signal and job terminaion (in seconds)
    signal_time: int = 120

    # extra configuration
    slurm_extra: str = field(init=False, default="")  # placeholder
    constraint: str = ""  # constraint on the nodes.
    exclude: str = ""  # nodes to exclude.
    account: str = ""
    qos: str = ""

    # cluster environment
    script_extra: str = ""

    def __manual_post_init__(self):
        self.slurm_extra = ""
        for name in ["exclude", "qos", "account", "constraint"]:
            val = getattr(self, name)
            if val:
                self.slurm_extra += f"#SBATCH --{name}={val}\n"

        # if partition, time or memory was not set
        priorities, max_times, memories = {}, {}, {}
        if self.partition == "" or self.time == -1 or self.mem == "":
            priorities, max_times, memories = self.extract_slurm_info()
        if self.partition == "":
            self.partition = min(priorities.keys(), key=lambda k: priorities[k]["job_factor"])
            print(f"No partition specified default to {self.partition}")
        if self.time == -1:
            self.time = max_times[self.partition]
            print(f"No time specified, default to {self.time} minutes")
        if self.mem == "":
            self.mem = memories[self.partition]
            print(f"No memory specified, default to {self.mem}MB")

    @staticmethod
    def extract_slurm_info() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        # retrieve partition max times (slow but run only once)

        print("Missing Slurm information, extracting them from `sinfo`.")
        sinfo = json.loads(subprocess.check_output("sinfo --json", shell=True))["sinfo"]
        priorities: dict[str, int] = {}
        max_times: dict[str, int] = {}
        memories: dict[str, int] = {}

        for info in sinfo:
            partition = info["partition"]["name"]
            if partition in priorities:
                continue

            priorities[partition] = info["partition"]["priority"]
            memories[partition] = info["memory"]["maximum"]  # in MB

            if info["partition"]["maximums"]["time"]["infinite"]:
                max_times[partition] = 14 * 24 * 60  # 14 days
            else:
                max_times[partition] = info["partition"]["maximums"]["time"]["number"]  # in minutes

        return priorities, max_times, memories


@dataclass
class LauncherConfig:
    name: str = "composition_default"

    dir: str = ""
    overwrite: bool = False
    copy_code: bool = True

    launcher: str = "sbatch"
    torchrun: bool = False
    python_env: str = "default"
    script: str = "apps.train"

    grid: dict[str, Any] = field(default_factory=dict)

    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    def __manual_post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        if not self.dir:
            self.dir = str(Path.home() / "logs" / self.name)
            print(f"No logging directory specified, default to {self.dir}")

        # recover python environment from the job was launched.
        if self.python_env:
            if self.python_env == "default":
                self.python_env = subprocess.check_output("which python", shell=True).decode("ascii").strip()
            else:
                self.python_env = f"{self.python_env}/bin/python"
            assert os.path.isfile(self.python_env)

        self.slurm.__manual_post_init__()


# -------------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------------


def copy_dir(input_dir: str, output_dir: str) -> None:
    rsync_cmd = (
        "rsync -ar --copy-links "
        "--exclude .git/ "
        # configuration and cache
        "--exclude .gitignore "
        "--exclude .vscode "
        "--exclude .ruff_cache "
        "--exclude '*.egg-info' "
        "--exclude '__pycache__' "
        "--exclude '*.md' "
        "--exclude '*.toml' "
        "--exclude '*.yaml' "
        # checkpoints and runs
        "--exclude logs/ "
        "--exclude savings/ "
        "--exclude wandb/ "
        # personal files and folders
        "--exclude '*.ipynb' "
        "--exclude 'tmp_*' "
        "--exclude tests/ "
        f"{input_dir}/ {output_dir}"
    )
    subprocess.call([rsync_cmd], shell=True)


# -------------------------------------------------------------------------------
# Grid job utilities
# -------------------------------------------------------------------------------


def _flatten_config(config: dict[str, Any], _parent_key: str = "") -> dict[str, Any]:
    """Flatten a nested configuration into a dot-separated format."""
    items = []
    for k, v in config.items():
        new_key = f"{_parent_key}.{k}" if _parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_config(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat configuration into a nested configuration."""
    nested = {}
    for key, value in config.items():
        keys = key.split(".")
        d = nested
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return nested


def get_configs_from_grid(config: dict[str, Any], grid_config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Get a set of configurations from a base configuration and a grid configuration.

    Parameters
    ----------
    config:
        The base configuration.
    grid_config:
        The grid configuration to launch a grid job.

    Returns
    -------
    List of configurations.
    """

    # get grid configurations as a list of flatten configs
    flatten_grid = _flatten_config(grid_config)
    keys, all_values = zip(*flatten_grid.items())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*all_values)]

    # merge on flatten config for simplicity
    config = _flatten_config(config)
    return [_unflatten_config(config | new_config) for new_config in all_configs]


# -------------------------------------------------------------------------------
# Job Launcher
# -------------------------------------------------------------------------------


LAUNCHER_SCRIPT = """#!/bin/bash

# Logging configuration
#SBATCH --job-name={name}
#SBATCH --output={log_dir}/logs/%j/main.out
#SBATCH --error={log_dir}/logs/%j/device_0.log
#SBATCH --open-mode=append
#SBATCH --mail-type=END
#SBATCH --mail-user=%u@meta.com

# Job specification
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={tasks}
#SBATCH --gres=gpu:{nb_gpus}
#SBATCH --cpus-per-gpu={nb_cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --distribution=block

# termination handling
#SBATCH --signal=USR1@{signal_time}

# slurm extra commands
{slurm_extra}
# cluster dependent commands
{script_extra}
# activate conda environment
eval "$({conda_exe} shell.bash hook)"
conda activate {conda_env_path}

{go_to_code_dir}

# launch the job
export OMP_NUM_THREADS=1
export LOG_DIR={log_dir}
{run_command}
"""


def launch_job(config: LauncherConfig, run_config: Any) -> None:
    """
    Launch a job on a Slurm cluster.

    Parameters
    ----------
    config:
        The configuration to launch the job.
    run_config:
        The training configuration of the job.
    """
    # alias
    slurm = config.slurm

    # logging directory
    dir = config.dir
    if os.path.exists(dir) and config.overwrite:
        confirm = input(
            f"Are you sure you want to delete the directory '{dir}'? This action cannot be undone. (yes/no): "
        )
        if confirm.upper().startswith("Y"):
            shutil.rmtree(dir)
            print(f"Directory '{dir}' has been deleted.")
        else:
            print("Operation cancelled.")
            return
    os.makedirs(dir, exist_ok=True)

    # copy code
    if config.copy_code:
        os.makedirs(f"{dir}/code", exist_ok=True)
        print(f"Copying code to {dir} ...", end="")
        copy_dir(os.getcwd(), f"{dir}/code")
        go_to_code_dir = f"cd {dir}/code"
    else:
        go_to_code_dir = ""
    print(" Done.")

    # write configs
    if config.grid:
        # handling potential grid run
        print("Writing grid configurations ...", end="")
        all_configs = get_configs_from_grid(run_config, config.grid)

        for i, nested_config in enumerate(all_configs, start=1):
            config_path = os.path.join(dir, f"config_{i}.yaml")
            with open(config_path, "w") as f:
                yaml.dump(nested_config, f, default_flow_style=False)

        slurm_extra = f"#SBATCH --array=1-{i}\n"
        config_path = "$LOG_DIR/config_$SLURM_ARRAY_TASK_ID.yaml"
    else:
        with open(f"{dir}/config.yaml", "w") as f:
            yaml.dump(run_config, f, default_flow_style=False)
        slurm_extra = ""
        config_path = "$LOG_DIR/config.yaml"

    # define proper conda environment
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    conda_env_path = str(Path(config.python_env).parent.parent)

    # aliases
    nodes = slurm.nodes
    nb_gpus = slurm.nb_gpus

    # define the run command
    if config.launcher == "sbatch":
        if config.torchrun:
            option_flags = f" --nproc_per_node={nb_gpus}" f" --nnodes={nodes}" " --node_rank=$SLURM_NODEID"
            run_command = f"torchrun {option_flags} -m {config.script} config={config_path}"
        else:
            run_command = f"srun python -u -m {config.script} config={config_path}"
    else:
        if config.torchrun:
            option_flags = f"--nproc_per_node={nb_gpus}"
            run_command = f"torchrun {option_flags} -m {config.script} config=$LOG_DIR/config.yaml"
        else:
            run_command = f"python -u -m {config.script} config=$LOG_DIR/config.yaml"

    bash_command = LAUNCHER_SCRIPT.format(
        name=config.name,
        log_dir=dir,
        partition=slurm.partition,
        nodes=nodes,
        tasks=nodes * nb_gpus,
        nb_gpus=nb_gpus,
        nb_cpus=slurm.nb_cpus,
        mem=slurm.mem,
        time=slurm.time,
        signal_time=slurm.signal_time,
        slurm_extra=slurm_extra + slurm.slurm_extra,
        script_extra=slurm.script_extra,
        conda_exe=conda_exe,
        conda_env_path=conda_env_path,
        go_to_code_dir=go_to_code_dir,
        run_command=run_command,
    )

    with open(f"{dir}/run.sh", "w") as f:
        f.write(bash_command)

    print(f"Launching job with `{config.launcher}` command.")
    os.system(f"{config.launcher} {dir}/run.sh")


def main() -> None:
    """
    Command line interface using OmegaConf

    Read argument from a config file specified by the `config` cli argument. E.g.,
    ```bash
    python -m launchers.stool script=src.apps.my_app.train config=src/apps/my_app/debug.yaml
    ```

    Non-specified arguments will be filled with the default values of the Config classes.
    """
    import argparse

    # Load config from path specified by the `config` cli argument
    parser = argparse.ArgumentParser(description="Launch job run.")
    parser.add_argument("config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    path = args.config

    with open(path) as f:
        file_config = yaml.safe_load(f)

    run_config = file_config.pop("run_config", None)
    file_config = file_config.pop("launcher", None)

    config = initialize_nested_dataclass(LauncherConfig, file_config)
    config.__manual_post_init__()

    # Launch job
    launch_job(config, run_config)


if __name__ == "__main__":
    main()
