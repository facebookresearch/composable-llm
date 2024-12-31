"""
Initialization of the cluster module

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from .cluster import ClusterConfig, ClusterManager
from .utils import get_hostname, get_rank, is_master_process
