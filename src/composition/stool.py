"""
Script tool to launch jobs on a Slurm cluster.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from .computing import ComputeConfig
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

    compute: ComputeConfig = field(default_factory=ComputeConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)


@dataclass
class LauncherConfig:
    run_config: TrainingConfig = field(default_factory=TrainingConfig)
    launcher: str = "sbatch"
    script: str = "apps.train"
    copy_code: bool = True
    python_env: str = "default"

    def __manual_post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
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
#SBATCH --output={dump_dir}/logs/%j.stdout
#SBATCH --error={dump_dir}/logs/%j.stderr
#SBATCH --open-mode=append
#SBATCH --mail-type=END
#SBATCH --mail-user=%u@meta.com

# Job specification
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={tasks}
#SBATCH --gres=gpu:{nb_gpu}
#SBATCH --cpus-per-gpu={nb_cpu}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --distribution=block

# termination handling
#SBATCH --signal=USR2@120

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
export LAUNCH_WITH="SBATCH"
export DUMP_DIR={dump_dir}
srun {log_output} python -u -m {script} config=$DUMP_DIR/run_config.yaml
"""


def launch_job(config: LauncherConfig):
    """
    Launch a job on a Slurm cluster.
    """
    dump_dir = config.run_config.monitor.dir
    os.makedirs(dump_dir, exist_ok=True)

    # copy code
    if config.copy_code:
        os.makedirs(f"{dump_dir}/code", exist_ok=True)
        print(f"Copying code to {dump_dir} ...", end="")
        copy_dir(os.getcwd(), f"{dump_dir}/code")
        go_to_code_dir = f"cd {dump_dir}/code"
    else:
        go_to_code_dir = ""
    print(" Done.")

    # write run_config
    with open(f"{dump_dir}/run_config.yaml", "w") as cfg:
        cfg.write(OmegaConf.to_yaml(config.run_config))

    # define proper conda environment
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    conda_env_path = str(Path(config.python_env).parent.parent)

    # log_output = "" if config.stdout else f"-o {dump_dir}/logs/output.log -e {dump_dir}/logs/error.log"

    nodes = config.run_config.compute.nodes
    nb_gpus = config.run_config.compute.nb_gpus

    bash_command = LAUNCHER_SCRIPT.format(
        name=config.run_config.monitor.name,
        dump_dir=dump_dir,
        partition=config.run_config.compute.partition,
        nodes=nodes,
        tasks=nodes * nb_gpus,
        nb_gpus=nb_gpus,
        nb_cpus=config.run_config.compute.nb_cpus,
        time=config.run_config.compute.time,
        mem=config.run_config.compute.mem,
        slurm_extra=config.run_config.compute.slurm_extra,
        script_extra=config.run_config.compute.script_extra,
        conda_exe=conda_exe,
        conda_env_path=conda_env_path,
        go_to_code_dir=go_to_code_dir,
        log_output="",
        tasks="",
        nodes_per_run="",
        script=config.script,
    )

    with open(f"{dump_dir}/run.sh", "w") as f:
        f.write(bash_command)

    print(f"Launching job with `{config.launcher}` command ...", end="")
    os.system(f"{config.launcher} {dump_dir}/run.sh")
    print(" Done.")


def main():
    """
    Command line interface using OmegaConf

    Read argument from a config file specified by the `config` cli argument. E.g.,
    ```bash
    python -m launchers.stool script=src.apps.train config=src/apps/debug.yaml
    ```

    Non-specified arguments will be filled with the default values of the Config classes.
    """
    # Load run_config from path specified by the `config` cli argument
    args = OmegaConf.from_cli()
    args.run_config = OmegaConf.load(args.config)
    del args.config

    # Load structured config
    default_cfg = OmegaConf.structured(LauncherConfig())
    config = OmegaConf.merge(default_cfg, args)
    config = OmegaConf.to_object(config)
    config.__manual_post_init__()

    # Launch job
    launch_job(config)


if __name__ == "__main__":
    main()
