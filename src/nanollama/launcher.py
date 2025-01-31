"""
Script to launch jobs on a Slurm cluster.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from .utils import flatten_config, initialize_nested_object, unflatten_config

logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Configuration Class
# ------------------------------------------------------------------------------


@dataclass
class SlurmConfig:
    # basic configuration
    partition: str = ""
    time: int = -1  # time limit of the job (in minutes).
    mem: str = ""  # amount of memory to allocate per node.
    nodes: int = 1  # number of nodes to run the job on.
    nb_gpus: int = 1  # number of GPUs required per node.
    nb_cpus: int = 10  # number of CPUs allocated per GPU.

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

    def __check_init__(self):
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
            logger.info(f"No partition specified default to {self.partition}")
        if self.time == -1:
            self.time = max_times[self.partition]
            logger.info(f"No time specified, default to {self.time} minutes")
        if self.mem == "":
            self.mem = memories[self.partition]
            logger.info(f"No memory specified, default to {self.mem}MB")

    @staticmethod
    def extract_slurm_info() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        # retrieve partition max times (slow but run only once)

        logger.info("Missing Slurm information, extracting them from `sinfo`.")
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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionnary to reinitialize it.
        """
        output = asdict(self)
        output.pop("slurm_extra")
        return output


@dataclass
class LauncherConfig:
    name: str = "composition_default"

    log_dir: str = ""
    overwrite: bool = False
    copy_code: bool = True

    launcher: str = "sbatch"
    torchrun: bool = False
    python_env: str = "default"
    script: str = ""

    grid: dict[str, Any] = field(default_factory=dict)

    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        assert self.script, "No script specified to run the job."

        for key in self.grid:
          if isinstance(self.grid[key], str):
              self.grid[key] = eval(self.grid[key])

        if not self.log_dir:
            self.log_dir = str(Path.home() / "logs" / self.name)
            logger.info(f"No logging directory specified, default to {self.log_dir}")
        else:
            self.log_dir = os.path.expandvars(self.log_dir)

        # recover python environment from the job was launched.
        if self.python_env:
            if self.python_env == "default":
                self.python_env = subprocess.check_output("which python", shell=True).decode("ascii").strip()
            else:
                self.python_env = f"{self.python_env}/bin/python"
            assert os.path.isfile(self.python_env)

        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------


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
        "--exclude 'core.*' "
        # personal files and folders
        "--exclude '*.ipynb' "
        "--exclude 'tmp_*' "
        "--exclude tests/ "
        f"{input_dir}/ {output_dir}"
    )
    subprocess.call([rsync_cmd], shell=True)


# ------------------------------------------------------------------------------
# Grid job utilities
# ------------------------------------------------------------------------------


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
    flatten_grid = flatten_config(grid_config)
    keys, all_values = zip(*flatten_grid.items())
    all_configs = [dict(zip(keys, v)) for v in product(*all_values)]

    # merge on flatten config for simplicity
    config = flatten_config(config)
    return [unflatten_config(config | new_config) for new_config in all_configs]


# ------------------------------------------------------------------------------
# Job Launcher
# ------------------------------------------------------------------------------


LAUNCHER_SCRIPT = """#!/bin/bash

# Logging configuration
#SBATCH --job-name={name}
#SBATCH --output={log_dir}/stdout/%j/main.out
#SBATCH --error={log_dir}/stdout/%j/main.err
#SBATCH --open-mode=append

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

# go to code directory
{go_to_code_dir}

# launch the job
export OMP_NUM_THREADS=1
{run_command}
"""


def launch_job(config: LauncherConfig, file_config: Any) -> None:
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
    run_config = file_config["run_config"]

    # logging directory
    log_dir = Path(config.log_dir)
    if log_dir.exists() and config.overwrite:
        confirm = input(
            f"Are you sure you want to delete the directory '{log_dir}'? This action cannot be undone. (yes/no): "
        )
        if confirm.upper().startswith("Y"):
            shutil.rmtree(log_dir)
            logger.info(f"Directory '{log_dir}' has been deleted.")
        else:
            logger.info("Operation cancelled.")
            return
    log_dir.mkdir(exist_ok=True, parents=True)

    # copy code
    if config.copy_code:
        code_dir = log_dir / "code"
        code_dir.mkdir(exist_ok=True)
        logger.info(f"Copying code to {code_dir}.")
        copy_dir(os.getcwd(), code_dir)
        go_to_code_dir = f"cd {code_dir}"
    else:
        go_to_code_dir = ""

    # write configs
    config_dir = log_dir / "tasks"
    config_dir.mkdir(exist_ok=True)
    if config.grid:
        # handling potential grid run
        logger.info("Writing grid configurations.")
        all_configs = get_configs_from_grid(run_config, config.grid)

        for i, nested_config in enumerate(all_configs, start=1):
            config_path = config_dir / f"{i}.yaml"
            file_config["run_config"] = nested_config
            with open(config_path, "w") as f:
                yaml.dump(file_config, f, default_flow_style=False)

        slurm_extra = f"#SBATCH --array=1-{i}\n"
        config_path = config_dir / "$SLURM_ARRAY_TASK_ID.yaml"
    else:
        config_path = config_dir / "0.yaml"
        with open(config_path, "w") as f:
            yaml.dump(file_config, f, default_flow_style=False)

        slurm_extra = ""
        config_path = config_dir / "0.yaml"

    # define proper conda environment
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    conda_env_path = str(Path(config.python_env).parent.parent)

    # aliases
    nodes = slurm.nodes
    nb_gpus = slurm.nb_gpus

    # define the run command
    if config.launcher == "sbatch":
        if config.torchrun:
            option_flags = f" --nproc_per_node={nb_gpus} --nnodes={nodes} --node_rank=$SLURM_NODEID"
            run_command = f"torchrun {option_flags} -m {config.script} {config_path}"
        else:
            run_command = f"srun python -u -m {config.script} {config_path}"
    else:
        if config.torchrun:
            option_flags = f"--nproc_per_node={nb_gpus}"
            run_command = f"torchrun {option_flags} -m {config.script} {config_path}"
        else:
            run_command = f"python -u -m {config.script} {config_path}"

    bash_command = LAUNCHER_SCRIPT.format(
        name=config.name,
        log_dir=log_dir,
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

    run_path = log_dir / "run.sh"
    with open(run_path, "w") as f:
        f.write(bash_command)

    logger.info(f"Launching job with `{config.launcher}` command.")
    os.system(f"{config.launcher} {run_path}")


def main() -> None:
    """
    Launch a training job (through slurm) from configuration file specified by cli argument.

    Usage:
    ```
    python -m launcher src/apps/my_app/my_config.yaml
    ```
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # parse file configuration path
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    path = os.path.expandvars(args.config)

    # obtain configuration from file
    with open(path) as f:
        file_config: dict[str, Any] = yaml.safe_load(f)

    # initialize configuration
    config = initialize_nested_object(LauncherConfig, file_config["launcher"], inplace=False)

    # launch job
    launch_job(config, file_config)


if __name__ == "__main__":
    main()
