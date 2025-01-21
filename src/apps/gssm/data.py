"""
Scripts to create datasets from configuration files

This file save to a file a dataset generated on the fly by the nanollama.data.gssm DataLoader.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import json
import logging
import os
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import h5py
import yaml

from nanollama.data.gssm import GSSMConfig, OnlineDataLoader, init_dataloader_state
from nanollama.utils import initialize_nested_object

logger = logging.getLogger("nanollama")


@dataclass
class DatasetConfig:
    path: str = ""
    n_data: int = 0

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        assert self.path, "Path to save the dataset must be specified."
        assert self.n_data, "n_data must be specified."

        self.path = os.path.expandvars(self.path)


@dataclass
class DataGenerationConfig:
    seq_len: int = 0
    seed: int = 0
    chunk_size: int = 0
    gssm: GSSMConfig = field(default_factory=GSSMConfig)
    sets: list[DatasetConfig] = field(default_factory=list)

    def __post_init__(self):
        # Mimic the nanollama.data.gssm.DataConfig
        self.asynchronous = False
        self.buffer_size = None
        self.batch_size = 0

        # Check validity of arguments and fill in missing values.
        assert self.seq_len, "seq_len must be specified."
        assert self.sets, "At least one dataset configuration must be specified."


def create_dataset(config: DataGenerationConfig) -> None:
    """
    Create a dataset according to the configuration.

    Parameters
    ----------
    config:
        Configuration of the data loader.
        The `batch_size` arguments is used as the total number of data to generate.
    """
    state = init_dataloader_state(config)
    chunk_size = config.chunk_size
    seq_len = config.seq_len

    with OnlineDataLoader(config, state) as dataloader:
        # iterate over the datasets to create
        for set_config in config.sets:
            n_data = set_config.n_data
            logger.info(f"Creating dataset with {n_data:,} points.")

            # we create the dataset by chunks
            if chunk_size:
                batch_size = chunk_size
            else:
                batch_size = n_data
                logger.info("Chunk size not specified. Saving dataset without chunking.")
            dataloader.batch_size = batch_size
            nb_chunks = n_data // batch_size

            # saving in hdf5 format
            path = Path(set_config.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as f:
                dset = f.create_dataset("data", shape=(n_data, seq_len), dtype=int, chunks=(chunk_size, seq_len))

                for i in range(nb_chunks):
                    logger.info(f"Creating chunk {i + 1}/{nb_chunks}")
                    begin, end = i * chunk_size, (i + 1) * chunk_size
                    dset[begin:end] = next(dataloader.generator)

                dataloader.batch_size = n_data % chunk_size
                if dataloader.batch_size:
                    logger.info("Creating residual chunk")
                    dset[end:] = next(dataloader.generator)

            logger.info(f"Dataset saved to {path}")


def main() -> None:
    """
    Create a dataset from configuration file

    Read argument from a config file specified by the `config` cli argument.
    E.g.,
    ```bash
    python -m src.nanollama.data.gssm config=apps/gssm/configs/debug.yaml
    ```

    Non-specified arguments will be filled with the default values of the Config classes.
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument("--task-id", type=int, default=1, help="Task id in the job array.")
    parser.add_argument("--nb-tasks", type=int, default=1, help="Number of tasks in the job array.")
    path = parser.parse_args().config
    task_id = parser.parse_args().task_id
    nb_tasks = parser.parse_args().nb_tasks

    with open(path) as f:
        file_configs = yaml.safe_load(f)

    all_seeds = file_configs["seed"]
    all_nodes = file_configs["gssm"]["nodes"]
    original_paths = [set_config["path"] for set_config in file_configs["sets"]]

    for i, (nodes, seed) in enumerate(product(all_nodes, all_seeds)):
        if i % nb_tasks != task_id - 1:
            continue

        file_configs["gssm"]["nodes"] = nodes
        file_configs["seed"] = seed
        for set_config, abc_path in zip(file_configs["sets"], original_paths):
            set_config["path"] = abc_path.replace("$GRIDID", str(i))
        logger.info(f"Creating datasets for environment {seed=}, {nodes=}")
        config = initialize_nested_object(DataGenerationConfig, file_configs, inplace=False)
        create_dataset(config)

    if task_id == 1:
        map_datasetid_gssm(path)


# ------------------------------------------------------------------------------
# Mappings from dataset to configuration
# ------------------------------------------------------------------------------


def map_datasetid_gssm(data_path: str) -> None:
    """
    Retrieve gssm configuration linked to datasets generated from the main function.
    """
    save_path = Path(data_path).parent / "map_grid_id_gssm_config.jsonl"
    id_path = Path(data_path).parent / "map_grid_id_gssm_id.jsonl"
    node_path = Path(data_path).parent / "map_gssm_id_gssm_config.jsonl"

    with open(data_path) as f:
        file_configs = yaml.safe_load(f)

    for path in [save_path, id_path, node_path]:
        with open(path, "w") as f:
            pass

    all_seeds = file_configs["seed"]
    all_nodes = file_configs["gssm"]["nodes"]
    testset_path: str = file_configs.pop("sets")[1]["path"]
    file_configs.pop("chunk_size")

    data_config = {}

    for i, (nodes, seed) in enumerate(product(all_nodes, all_seeds)):
        # specify gssm config
        gssm_config = {}
        gssm_config["gssm"] = file_configs["gssm"]
        gssm_config["gssm"]["nodes"] = nodes
        gssm_config["seed"] = seed
        gssm_config["batch_size"] = "FAKE"
        gssm_config["seq_len"] = "FAKE"

        # retrieve where the generated testset is stored.
        data_config |= {"path": testset_path.replace("$GRIDID", str(i))}

        with open(save_path, "a") as f:
            print(json.dumps({"data": data_config, "gssm": gssm_config}), file=f, flush=True)

        with open(id_path, "a") as f:
            print(json.dumps({"node_id": all_nodes.index(nodes), "seed": seed, "grid_id": i}), file=f, flush=True)

    for nodes in all_nodes:
        with open(node_path, "a") as f:
            print(json.dumps({"node_id": all_nodes.index(nodes), "nodes": nodes}, indent=4), file=f, flush=True)


if __name__ == "__main__":
    main()
