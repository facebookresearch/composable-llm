"""
Estimate entropy from gzip compression

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import json
import zlib
from pathlib import Path
import getpass
import os

import h5py
import yaml


def gzip_estimate(exp: int, code_dir: str) -> None:
    save_path = f"/checkpoint/{getpass.getuser()}/icml/logs/exp{exp}/gzip.jsonl"
    with open(os.path.expandvars(save_path), "w") as f:
        pass

    path = f"{code_dir}/src/apps/gssm/configs/experiment{exp}/entropy.yaml"
    with open(os.path.expandvars(path)) as f:
        config = yaml.safe_load(f)

    all_configs = []
    with open(os.path.expandvars(config.pop("configs_path"))) as f:
        for line in f:
            all_configs.append(json.loads(line))

    for conf in all_configs:
        path = conf["data"]["path"]
        with h5py.File(os.path.expandvars(path)) as f:
            data = f["data"][:, 1:]

        entropy = len(zlib.compress(data.tobytes(), level=9)) / data.size
        with open(os.path.expandvars(save_path), "a") as f:
            print(json.dumps({"grid_id": int(Path(path).parent.name), "gzip_difficulty": entropy}), file=f, flush=True)


if __name__ == "__main__":
    from .launcher_entropy import merge_hmm_estimate

    code_dir = "$CODE_DIR"
    exp = int(input("Which experiment? "))

    print(f"Running experiment {exp}")
    gzip_estimate(exp, code_dir)

    # merge the hmm estimate in the same format

    print(f"Merging hmm estimate for experiment {exp}")
    merge_hmm_estimate(exp, code_dir)
