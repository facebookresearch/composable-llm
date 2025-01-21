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

import h5py
import yaml


def gzip_estimate(exp: int, code_dir: str) -> None:
    save_path = f"/checkpoint/vivc/icml/logs/exp{exp}/gzip.jsonl"
    with open(save_path, "w") as f:
        pass

    path = f"{code_dir}src/apps/gssm/configs/experiment{exp}/entropy.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)

    all_configs = []
    with open(config.pop("configs_path")) as f:
        for line in f:
            all_configs.append(json.loads(line))

    for conf in all_configs:
        path = conf["data"]["path"]
        with h5py.File(path) as f:
            data = f["data"][:, 1:]

        entropy = len(zlib.compress(data.tobytes(), level=9)) / data.size
        with open(save_path, "a") as f:
            print(json.dumps({"id": int(Path(path).parent.name), "entropy": entropy}), file=f, flush=True)


if __name__ == "__main__":
    code_dir = "/private/home/vivc/code/composable-llm/"
    for exp in range(1, 5):
        print(f"Running experiment {exp}")
        gzip_estimate(exp, code_dir)
