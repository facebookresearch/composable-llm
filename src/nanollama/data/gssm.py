"""
Graph Structured Sequential Model (GSSM)

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from collections.abc import Generator
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import numpy as np
from numpy.random import SeedSequence, default_rng

from ..distributed import get_rank
from .loader import DataLoader

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# GSSM - Graph Structure
# ------------------------------------------------------------------------------


@dataclass
class NodeConfig:
    name: str = ""
    state_dim: int = 0
    alpha: float = 0
    parents: list[str] = field(default_factory=list)
    mode: str = "default"
    root: bool = False

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        assert self.name, f"Node name must be specified. {self}"
        assert self.state_dim, f"`state_dim` must be specified. {self}"
        assert self.alpha, f"`alpha` must be specified. {self}"
        assert self.mode.lower() in ["default", "slow", "dead", "context"], f"Invalid mode: {self.mode}"


class Node:
    """
    Node in a graph-structured sequential model (GSSM).

    Parameters
    ----------
    state_dim:
        Number of state values the node can take
    alphas:
        Dirichlet concentration parameter
    parents:
        List of parent nodes
    mode:
        Mode of the transition kernel. Can be 'default', 'slow', 'dead', or 'context'.
    rng:
        Random number generator
    root:
        Whether the node is the observed node

    Attributes
    ----------
    state:
        Current state of the node.
    time:
        Current time step.
    kernels:
        TransitionKernel used for sampling.
    """

    def __init__(
        self,
        state_dim: int,
        alpha: float,
        parents: list["Node"],
        mode: str,
        rng: np.random.Generator,
        root: bool,
    ):
        self.state_dim = state_dim
        self.alpha = alpha
        self.parents = parents
        self.mode = mode
        self.rng = rng
        self.root = root

        self.kernels = self.sample_transitions(alpha, mode)

        self.state = None
        self.time = None

    def sample_transitions(self, alpha: float, mode: str = "default") -> list[np.ndarray[float]]:
        """Initialize transition kernels"""

        # in the `context` mode, the transition matrix change each time
        if mode == "context":
            return []

        # observed node does not have connection to itself
        if self.root:
            fan_ins = []
        else:
            fan_ins = [self.state_dim]
        fan_ins += [pnode.state_dim for pnode in self.parents]

        return [
            self.sample_transition(fan_in=fan_in, alpha=alpha, mode=mode)
            for fan_in in fan_ins
        ]

    def sample_transition(self, fan_in: int, alpha: float, mode: str) -> np.ndarray[float]:
        """Sample transition kernel"""

        # transition
        alphas = np.full(self.state_dim, alpha)
        transition = self.rng.dirichlet(alphas, size=fan_in)

        # handle special modes
        mode = mode.lower()
        # in the `slow` mode, argmax p(state[t+1] | state[t]=z) = z
        if mode == "slow":
            index = np.arange(fan_in)
            argmax = transition.argmax(axis=1)
            max_val = transition[index, argmax]
            transition[index, argmax] = transition[index, index]
            transition[index, index] = max_val

        # in the `dead` mode, argmax p(state[t+1] | ...) = 0
        elif mode == "dead":
            index = np.arange(fan_in)
            argmax = transition.argmax(axis=1)
            max_val = transition[index, argmax]
            transition[index, argmax] = transition[:, 0]
            transition[:, 0] = max_val

        return transition

    def initialize(self, bsz: int) -> None:
        """
        Initialize the state of the node.

        Parameters
        ----------
        bsz:
            Batch size
        """
        for parent in self.parents:
            if parent.time != 0:
                parent.initialize(bsz)
        self.state = np.zeros(bsz, dtype=int)
        self.time = 0

        # in-context learning mode
        if self.mode == "context":
            self.kernels = self.sample_transitions(alpha=self.alpha)

    def evolve(self) -> None:
        """
        Sample the next state given the current state and parent states.

        Parameters
        ----------
        self_state
            Current state of the node
        parent_states
            Current states of the parent nodes
        """
        for parent in self.parents:
            if parent.time != self.time + 1:
                assert parent.time == self.time, "Parent node time is not correct."
                parent.evolve()

        if self.root:
            all_states = []
        else:
            all_states = [self.state]

        all_states += [parent.state for parent in self.parents]
        proba = sum([kernel[state] for kernel, state in zip(self.kernels, all_states)])

        # Vectorized sampling
        random_values = self.rng.random(self.state.shape)
        p_cumulative = proba.cumsum(axis=-1)
        p_cumulative /= p_cumulative[..., -1:]

        self.state = (random_values[:, None] < p_cumulative).argmax(axis=1)
        self.time += 1

    def __repr__(self):
        return "Node(" + ", ".join(
            [
                f"state_dim={self.state_dim}",
                f"state={self.state}",
                f"time={self.time}",
                f"nb_parents={len(self.parents)}",
                f"root={self.root})",
            ]
        )


# ------------------------------------------------------------------------------
# GSSM - Configuration and Building
# ------------------------------------------------------------------------------


@dataclass
class GSSMConfig:
    nodes: list[NodeConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.nodes:
            raise ValueError("At least one node must be specified.")


def build_gssm(config: GSSMConfig, rng: np.random.Generator) -> Node:
    """
    Build a graph from a configuration.

    Parameters
    ----------
    config:
        Configuration of the GSSM.
    rng:
        Random number generator.

    Returns
    -------
    nodes:
        Dictionary of nodes.
    """
    logger.info(f"Building graph from {config}")
    nodes_to_initialize = config.nodes
    nodes: dict[str, Node] = {}

    # observe all the other nodes
    while nodes_to_initialize:
        node_config = nodes_to_initialize.pop(0)

        # check if all parents are initialized
        parents_name = node_config.parents
        minimum = True
        for parent_name in parents_name:
            if parent_name not in nodes:
                minimum = False
                break

        if not minimum:
            logger.debug(f"Parents of node {node_config} are not initialized yet")
            nodes_to_initialize.append(node_config)
            continue

        parents = []
        for parent_name in parents_name:
            parents.append(nodes[parent_name])

        logger.info(f"Initializing node {node_config.name}")
        nodes[node_config.name] = Node(
            state_dim=node_config.state_dim,
            alpha=float(node_config.alpha),
            parents=parents,
            mode=node_config.mode,
            rng=rng,
            root=node_config.root,
        )

        logger.info("Graph is built")
        if node_config.root:
            return nodes[node_config.name]

    raise ValueError("No root node found")


# ------------------------------------------------------------------------------
# DataLoader Manager - State and Configuration
# ------------------------------------------------------------------------------


@dataclass
class DataConfig:
    seq_len: int = 0
    batch_size: int = 0
    seed: int = 0
    gssm: GSSMConfig = field(default_factory=GSSMConfig)
    asynchronous: bool = True  # asynchronous data loading
    buffer_size: int = 4  # number of batches to bufferize asynchronously for data loading

    def __post_init__(self):
        assert self.batch_size, "batch_size should be set"
        assert self.seq_len, "seq_len should be specified"


@dataclass
class DataLoaderState:
    rng_state: dict[str, Any]
    graph_rng_state: dict[str, Any] = field(default_factory=dict)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng_state,
            "graph_rng_state": self.graph_rng_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rng_state = state_dict["rng_state"]
        self.graph_rng_state = state_dict["graph_rng_state"]

    def report_restart_info(self, rng_state: dict[str, Any]) -> None:
        """
        Report the restart information to the state.

        See Also
        --------
        OnlineDataLoader.get_restart_info
        """
        self.rng_state = rng_state


def init_dataloader_state(config: DataConfig) -> DataLoaderState:
    """
    Initialize the state of random number generators.
    """
    # generate independent seeds
    ss = SeedSequence(config.seed)
    rank = get_rank()
    seeds = ss.spawn(rank + 1)

    # recover state from seeds
    # graph should be the same for every process
    graph_rng_state = default_rng(seeds[0]).bit_generator.state
    rng_state = default_rng(seeds[-1]).bit_generator.state
    return DataLoaderState(rng_state=rng_state, graph_rng_state=graph_rng_state)


# ------------------------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------------------------


class OnlineDataLoader(DataLoader):
    """
    Context manager for the online data loader.

    Parameters
    ----------
    config:
        The configuration of the data loader.
    state:
        The state of the data loader.
    """

    def __init__(self, config: DataConfig, state: DataLoaderState):
        super().__init__(config)

        # data loader configuration
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len

        # track randomness
        self.rng_state = state.rng_state
        logger.debug(f"RNG: {state}")

        # ensure consistency of transition kernels over restart
        rng = default_rng()
        rng.bit_generator.state = state.graph_rng_state
        self.node = build_gssm(config.gssm, rng=rng)

    def batch_iterator(self) -> Generator[np.ndarray, None, None]:
        """
        Generate batches of sentences iteratively.
        """
        # ensure consistency of transition kernels over restart
        rng = default_rng()
        rng.bit_generator.state = self.rng_state

        while True:
            batch = np.empty((self.batch_size, self.seq_len), dtype=int)
            self.node.initialize(self.batch_size)

            for t in range(self.seq_len):
                assert self.node.time == t, f"Discrepancy in time: {self.node.time} and {t}."
                self.node.evolve()
                batch[:, t] = self.node.state

            self.rng_state = rng.bit_generator.state
            yield batch

    def get_restart_info(self) -> dict[str, Any]:
        """
        Get restart information.

        See Also
        --------
        DataLoaderState.report_restart_info
        """
        return self.rng_state
