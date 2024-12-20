"""
OS Environment manager

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import os
import random
import subprocess
from dataclasses import asdict, dataclass

from .utils import is_slurm_job


@dataclass
class OsEnvironment:
    OMP_NUM_THREADS: str = "1"


def set_os_environment(config: OsEnvironment):
    """
    Set OS environment variables based on configuration.
    """
    env_vars = asdict(config)

    # # When using Triton, it attempts to locate prebuilt kernels in a cache
    # # located at ~/.triton/cache, but when that's backed by NFS this can fail
    # # with a "OSError: [Errno 116] Stale file handle" error. If we were to set
    # # it to a local directory it would belong to the first user who created it
    # # and it would fail for the job of any other successive user assigned to
    # # that machine. To avoid all this mess we use a temporary per-process cache.
    # triton_cache_dir = tempfile.mkdtemp()
    # atexit.register(shutil.rmtree, triton_cache_dir, ignore_errors=True)
    # env_vars["TRITON_CACHE_DIR"] = triton_cache_dir

    # # We change the tmp dir to /scratch in case it's slurm job
    # # This avoids filling up the host's usually limited tmpfs
    # # A full tmpfs leads to very slow creation of processes and weird bugs
    # if is_slurm_job():
    #     new_tmp = f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}"
    #     if os.path.exists(new_tmp):
    #         env_vars["TMP_DIR"] = new_tmp

    for name, value in env_vars.items():
        if os.environ.get(name) != str(value):
            os.environ[name] = str(value)
            print(f"WARNING: Setting {name} to {value}")

    # set up slurm environment
    if is_slurm_job():
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
        master_addr = hostnames.split()[0].decode("utf-8")

        MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
        job_id = int(os.environ["SLURM_JOB_ID"])
        rng = random.Random(job_id)
        master_port = rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
