"""
Scripts to create datasets from configuration files

This file save to a file a dataset generated on the fly by the nanollama.data.gssm DataLoaderManager.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nanollama.data.gssm import DataLoaderManager, GSSMConfig, init_dataloader_state

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    path: str = ""
    n_data: int = 0

    def __manual_post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__manual_post_init__"):
                module.__manual_post_init__()
        assert self.path, "Path to save the dataset must be specified."
        assert self.n_data, "n_data must be specified."


@dataclass
class DataGenerationConfig:
    seq_len: int = 0
    seed: int = 0
    gssm: GSSMConfig = field(default_factory=GSSMConfig)
    sets: list[DatasetConfig] = field(default_factory=list)

    def __post_init__(self):
        # Mimic the nanollama.data.gssm.DataLoaderConfig
        self.asynchronous = False
        self.buffer_size = None
        self.batch_size = 0

    def __manual_post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # manual post initialization of all modules
        for module in self.sets + [self.gssm]:
            if hasattr(module, "__manual_post_init__"):
                module.__manual_post_init__()

        assert self.seq_len, "seq_len must be specified."
        assert self.sets, "At least one dataset configuration must be specified."


def create_dataset(config: DataGenerationConfig):
    """
    Create a dataset according to the configuration.

    Parameters
    ----------
    config:
        Configuration of the data loader.
        The `batch_size` arguments is used as the total number of data to generate.
    """
    # Initialize the data loader state
    state = init_dataloader_state(config)

    # Create a DataLoaderManager instance
    with DataLoaderManager(config, state) as dataloader:
        for set_config in config.sets:
            # create dataset with `n_data` point
            n_data = set_config.n_data
            dataloader.batch_size = n_data
            dataset, _ = dataloader.get_batch()

            # Save the dataset to a file
            path = Path(set_config.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dataset, path)
            logger.info(f"Dataset with {n_data} points saved to {path}")


def main():
    """
    Create a dataset from configuration file

    Read argument from a config file specified by the `config` cli argument.
    E.g.,
    ```bash
    python -m src.nanollama.data.gssm config=apps/gssm/configs/debug.yaml
    ```

    Non-specified arguments will be filled with the default values of the Config classes.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    cli_args = OmegaConf.from_cli()
    file_configs = OmegaConf.load(cli_args.pop("config", None))

    default_config = OmegaConf.structured(DataGenerationConfig())

    for env, file_config in file_configs.items():
        logger.info(f"Creating datasets for environment {env}")
        config = OmegaConf.merge(default_config, file_config, cli_args)
        config = OmegaConf.to_object(config)
        config.__manual_post_init__()

        create_dataset(config)


if __name__ == "__main__":
    main()
