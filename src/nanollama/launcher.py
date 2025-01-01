"""
Script to launch jobs on a Slurm cluster.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import itertools
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from .cluster import ClusterConfig
from .monitor import MonitorConfig

# -------------------------------------------------------------------------------
# Configuration Class
# -------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    data: Optional[Any] = None
    model: Optional[Any] = None
    optim: Optional[Any] = None

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)


@dataclass
class LauncherConfig:
    config: TrainingConfig = field(default_factory=TrainingConfig)
    launcher: str = "sbatch"
    torchrun: bool = False
    script: str = "apps.train"
    copy_code: bool = True
    python_env: str = "default"

    def __manual_post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # manual post initialization of all modules
        for module in self.config.__dict__.values():
            if hasattr(module, "__manual_post_init__"):
                module.__manual_post_init__()

        # recover python environment from the job was launched.
        if self.python_env:
            if self.python_env == "default":
                self.python_env = subprocess.check_output("which python", shell=True).decode("ascii").strip()
            else:
                self.python_env = f"{self.python_env}/bin/python"
            assert os.path.isfile(self.python_env)


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


def flatten_config(config: dict[str, Any], _parent_key: str = "") -> dict[str, Any]:
    """
    Flatten a nested configuration into a dot-separated format.

    Parameters
    ----------
    config:
        A nested configuration.

    Returns
    -------
    A flattened configuration.
    """
    items = []
    for k, v in config.items():
        new_key = f"{_parent_key}.{k}" if _parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a flat configuration into a nested configuration.

    Parameters
    ----------
    config:
        A flat configuration.

    Returns
    -------
    A nested configuration.
    """
    nested = {}
    for key, value in config.items():
        keys = key.split(".")
        d = nested
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return nested


def write_configs_from_grid(config: dict, grid_config: dict, path: str) -> int:
    """
    Write a set of configurations from a grid configuration.

    Parameters
    ----------
    config:
        The base configuration.
    grid_config:
        The grid configuration to launch a grid job.
    path:
        The path to write the configurations.

    Returns
    -------
    The number of configurations written.
    """

    # get grid configurations as a list of flatten configs
    flatten_grid = flatten_config(grid_config)
    keys, all_values = zip(*flatten_grid.items())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*all_values)]

    # merge on flatten config for simplicity
    config = flatten_config(config)

    for i, new_config in enumerate(all_configs, start=1):
        # update base configuration
        config |= new_config
        nested_config = unflatten_config(config)

        config_path = os.path.join(path, f"config_{i}.yaml")
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(nested_config))

    return i


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


def launch_job(config: LauncherConfig, grid: Optional[dict[str, Any]] = None) -> None:
    """
    Launch a job on a Slurm cluster.

    Parameters
    ----------
    config:
        The configuration to launch the job.
    grid:
        A grid configuration to launch a grid job.
    """
    # aliases
    monitor_config = config.config.monitor
    slurm_config = config.config.cluster.slurm

    # logging directory
    dir = monitor_config.dir
    if os.path.exists(dir) and monitor_config.overwrite:
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

    # handling potential grid run
    if grid:
        print("Writing grid configurations ...", end="")
        nb_configs = write_configs_from_grid(asdict(config.config), grid, dir)
        slurm_extra = f"#SBATCH --array=1-{nb_configs}\n"
        config_path = "$LOG_DIR/config_$SLURM_ARRAY_TASK_ID.yaml"
    else:
        # write config
        with open(f"{dir}/config.yaml", "w") as cfg:
            cfg.write(OmegaConf.to_yaml(config.config))
        slurm_extra = ""
        config_path = "$LOG_DIR/config.yaml"

    # define proper conda environment
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    conda_env_path = str(Path(config.python_env).parent.parent)

    # aliases
    nodes = slurm_config.nodes
    nb_gpus = slurm_config.nb_gpus

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
        name=monitor_config.name,
        log_dir=dir,
        partition=slurm_config.partition,
        nodes=nodes,
        tasks=nodes * nb_gpus,
        nb_gpus=nb_gpus,
        nb_cpus=slurm_config.nb_cpus,
        mem=slurm_config.mem,
        time=slurm_config.time,
        signal_time=slurm_config.signal_time,
        slurm_extra=slurm_extra + slurm_config.slurm_extra,
        script_extra=slurm_config.script_extra,
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
    # Load config from path specified by the `config` cli argument
    args = OmegaConf.from_cli()
    args.config = OmegaConf.load(args.config)
    grid = args.config.pop("grid", None)
    if grid:
        grid = OmegaConf.to_object(grid)

    # Default to default arguments for unspecified values
    default_config = OmegaConf.structured(LauncherConfig())
    config = OmegaConf.merge(default_config, args)
    config: LauncherConfig = OmegaConf.to_object(config)
    config.__manual_post_init__()

    # Launch job
    launch_job(config, grid)


if __name__ == "__main__":
    main()
