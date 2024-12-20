"""
Slurm tool script to launch jobs on a Slurm cluster.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
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
    checkpoint: Optional[Any] = None

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
        "--exclude '*.egg-info' "
        "--exclude '__pycache__' "
        "--exclude '*.md' "
        "--exclude '*.toml' "
        "--exclude '*.yaml' "
        # checkpoints and runs
        "--exclude dumps/ "
        "--exclude logs/ "
        "--exclude savings/ "
        # personal files and folders
        "--exclude '*.ipynb' "
        "--exclude 'tmp_*' "
        "--exclude tests/ "
        f"{input_dir}/ {output_dir}"
    )
    subprocess.call([rsync_cmd], shell=True)


# -------------------------------------------------------------------------------
# Job Launcher
# -------------------------------------------------------------------------------


LAUNCHER_SCRIPT = """#!/bin/bash

# Logging configuration
#SBATCH --job-name={name}
#SBATCH --output={log_dir}/logs/%j/main.out
#SBATCH --error={log_dir}/logs/%j/main.err
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


def launch_job(config: LauncherConfig):
    """
    Launch a job on a Slurm cluster.
    """
    # logging directory
    log_dir = config.config.monitor.dir
    if os.path.exists(log_dir) and config.config.monitor.overwrite:
        confirm = input(
            f"Are you sure you want to delete the directory '{log_dir}'? This action cannot be undone. (yes/no): "
        )
        if confirm.upper().startswith("Y"):
            shutil.rmtree(log_dir)
            print(f"Directory '{log_dir}' has been deleted.")
        else:
            print("Operation cancelled.")
            return
    os.makedirs(log_dir, exist_ok=True)

    # copy code
    if config.copy_code:
        os.makedirs(f"{log_dir}/code", exist_ok=True)
        print(f"Copying code to {log_dir} ...", end="")
        copy_dir(os.getcwd(), f"{log_dir}/code")
        go_to_code_dir = f"cd {log_dir}/code"
    else:
        go_to_code_dir = ""
    print(" Done.")

    # write config
    with open(f"{log_dir}/config.yaml", "w") as cfg:
        cfg.write(OmegaConf.to_yaml(config.config))

    # define proper conda environment
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    conda_env_path = str(Path(config.python_env).parent.parent)

    # aliases
    slurm_cfg = config.config.cluster.slurm
    nodes = slurm_cfg.nodes
    nb_gpus = slurm_cfg.nb_gpus

    # define the run command
    if config.launcher == "sbatch":
        if config.torchrun:
            option_flags = f"--nproc_per_node={nb_gpus}"
            run_command = f"torchrun {option_flags} -m {config.script} config=$LOG_DIR/config.yaml"
        else:
            run_command = f"srun python -u -m {config.script} config=$LOG_DIR/config.yaml"
    else:
        if config.torchrun:
            run_command = f"torchrun --nproc-per-node {nb_gpus} -m {config.script} config=$LOG_DIR/config.yaml"
        else:
            run_command = f"python -u -m {config.script} config=$LOG_DIR/config.yaml"

    bash_command = LAUNCHER_SCRIPT.format(
        name=config.config.monitor.name,
        log_dir=log_dir,
        partition=slurm_cfg.partition,
        nodes=nodes,
        tasks=nodes * nb_gpus,
        nb_gpus=nb_gpus,
        nb_cpus=slurm_cfg.nb_cpus,
        mem=slurm_cfg.mem,
        time=slurm_cfg.time,
        signal_time=slurm_cfg.signal_time,
        slurm_extra=slurm_cfg.slurm_extra,
        script_extra=slurm_cfg.script_extra,
        conda_exe=conda_exe,
        conda_env_path=conda_env_path,
        go_to_code_dir=go_to_code_dir,
        run_command=run_command,
    )

    with open(f"{log_dir}/run.sh", "w") as f:
        f.write(bash_command)

    print(f"Launching job with `{config.launcher}` command.")
    os.system(f"{config.launcher} {log_dir}/run.sh")


def main():
    """
    Command line interface using OmegaConf

    Read argument from a config file specified by the `config` cli argument. E.g.,
    ```bash
    python -m launchers.stool script=src.apps.train config=src/apps/debug.yaml
    ```

    Non-specified arguments will be filled with the default values of the Config classes.
    """
    # Load config from path specified by the `config` cli argument
    args = OmegaConf.from_cli()
    args.config = OmegaConf.load(args.config)

    # Default to default arguments for unspecified values
    default_cfg = OmegaConf.structured(LauncherConfig())
    config = OmegaConf.merge(default_cfg, args)
    config = OmegaConf.to_object(config)
    config.__manual_post_init__()

    # Launch job
    launch_job(config)


if __name__ == "__main__":
    main()
