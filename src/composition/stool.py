"""
Script tool to launch jobs on a Slurm cluster.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from .checkpoint import CheckpointConfig
from .computing import ComputeConfig
from .data import DataConfig
from .model import TransformerConfig
from .monitor import MonitorConfig
from .optim import OptimizerConfig

logger = logging.getLogger(__file__)

# -------------------------------------------------------------------------------
# Configuration Class
# -------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: TransformerConfig = field(default_factory=TransformerConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)


@dataclass
class LauncherConfig:
    run_config: TrainingConfig = None
    launcher: str = "bash"
    script: str = "apps.train"
    copy_code: bool = True
    python_env: str = "default"
    stdout: bool = False

    def __post_init__(self):
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
    print(f"Copying : {input_dir}\n" f"to      : {output_dir} ...")
    assert os.path.isdir(input_dir), f"{input_dir} is not a directory"
    assert os.path.isdir(output_dir), f"{output_dir} is not a directory"
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
    print(f"Copying command: {rsync_cmd}")
    subprocess.call([rsync_cmd], shell=True)
    print("Copy done.")


# -------------------------------------------------------------------------------
# Job Launcher
# -------------------------------------------------------------------------------


LAUNCHER_SCRIPT = """#!/bin/bash

# activate conda environment
eval "$({conda_exe} shell.bash hook)"
conda activate {conda_env_path}

# launch the job
cd {dump_dir}/code
export OMP_NUM_THREADS=1
python -u -m {script} config={dump_dir}/base_config.yaml
"""


def launch_job(config: LauncherConfig):
    """
    Launch a job on a Slurm cluster.
    """
    dump_dir = config.run_config.monitor.dir
    os.makedirs(dump_dir, exist_ok=True)

    if config.copy_code:
        os.makedirs(f"{dump_dir}/code", exist_ok=True)
        print("Copying code ...")
        copy_dir(os.getcwd(), f"{dump_dir}/code")

    with open(f"{dump_dir}/base_config.yaml", "w") as cfg:
        cfg.write(OmegaConf.to_yaml(config.run_config))

    conda_exe = os.environ.get("CONDA_EXE", "conda")
    conda_env_path = str(Path(config.python_env).parent.parent)

    log_output = "" if config.stdout else f"-o {dump_dir}/logs/output.log -e {dump_dir}/logs/error.log"

    bash_command = LAUNCHER_SCRIPT.format(
        script=config.script,
        dump_dir=dump_dir,
        conda_exe=conda_exe,
        conda_env_path=conda_env_path,
        log_output=log_output,
    )

    with open(f"{dump_dir}/run.sh", "w") as f:
        f.write(bash_command)

    logger.info(f"Launching job with command: {bash_command}")
    os.system(f"bash {dump_dir}/run.sh")
    logger.info("Job launched.")


def main():
    """
    Command line interface using OmegaConf

    Read argument from a config file specified by the `config` cli argument. E.g.,
    ```bash
    python -m apps.train config=apps/debug.yaml
    python -m launchers.stool script=apps.main.train config=apps/main/configs/lr3e3.yaml nodes=1
    ```

    Non-specified arguments will be filled with the default values of the Config classes.
    """
    args = OmegaConf.from_cli()
    args.run_config = OmegaConf.load(args.config)
    del args.config
    config = LauncherConfig(**args)
    launch_job(config)


if __name__ == "__main__":
    main()
