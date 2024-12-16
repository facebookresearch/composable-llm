"""
Path Configuration

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

from configparser import ConfigParser
from pathlib import Path

# Placeholders
CHECKPOINT_DIR = Path()

# Priority to user config
ROOT_DIR = Path(__file__).parents[2].resolve()
config: dict[str, dict] = ConfigParser()
config.read(ROOT_DIR / "default_config.ini")
config.read(ROOT_DIR / "user_config.ini")

# Priority to absolute over relative paths
for name in config["Relative Paths"]:
    globals()[name.upper()] = ROOT_DIR / config["Relative Paths"][name]

for name in config["Absolute Paths"]:
    globals()[name.upper()] = Path(config["Absolute Paths"][name])
