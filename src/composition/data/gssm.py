"""
Graph Structured Sequential Model (GSSM)

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
from scipy.stats import dirichlet

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------
# GSSM - Transition Kernel
# -------------------------------------------------------------------------------


class TransitionKernel:
    """
    A transition kernel that generates the next state given the current state.

    Parameters
    ----------
    fan_in:
        Number of input states.
    fan_out:
        Number of output states.
    rng:
        Random number generator.
    alphas:
        Dirichlet prior parameter. Can be a float, int, list of floats, or a numpy array of shape (fan_out,).

    Attributes
    ----------
    p_transition:
        A fan_in x fan_out transition matrix.
    """

    def __init__(
        self, fan_in: int, fan_out: int, alphas: Union[int, float, list[float], np.ndarray], rng: Generator = None
    ):

        # handle various types
        if isinstance(alphas, float) or isinstance(alphas, int):
            alphas = np.full(fan_out, alphas)
        if isinstance(alphas, list):
            alphas = np.array(alphas)
        assert alphas.shape == (
            fan_out,
        ), "Alphas must be a float, int, list of floats, or a numpy array of shape (fan_out,)."

        # set random number generator
        if rng is None:
            rng = default_rng()
        self.rng = rng

        self.fan_out = fan_out
        self.p_transition = dirichlet.rvs(alphas, size=fan_in, random_state=rng)
        self._cumulative = np.cumsum(self.p_transition, axis=1)

    def __call__(self, state: Union[int, list[int], np.ndarray]) -> Union[int, np.ndarray]:
        """
        Generate the next state given the current state.

        Parameters
        ----------
        state:
            Current state. Can be an integer, list of integers, or a numpy array of integers.

        Returns
        -------
        next_state
            Next state. If state is a list, it will return a numpy array.
        """
        if isinstance(state, int):
            return self.rng.choice(self.fan_out, p=self.p_transition[state])

        # Convert state to a numpy array if it's a list
        if isinstance(state, list):
            state = np.asarray(state)

        # Vectorized sampling
        random_values = self.rng.random(state.shape)
        p_cumulative = self._cumulative[state]
        return (random_values[:, None] < p_cumulative).argmax(axis=1)


# -------------------------------------------------------------------------------
# GSSM - Graph Structure
# -------------------------------------------------------------------------------


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
    rng:
        Random number generator

    Attributes
    ----------
    state:
        Current state of the node.
    time:
        Current time step.
    size_in:
        The size of the unravel input state, including the state_dim of the node and the state_dim of each parent node.
    kernel:
        TransitionKernel used for sampling.
    """

    def __init__(
        self,
        state_dim: int,
        alphas: Union[int, float, list[float], np.ndarray],
        parents: Optional[list["Node"]] = None,
        rng: Generator = None,
    ):
        self.parents = parents if parents is not None else []

        # Calculate the fan_in based on the parents
        fan_in = state_dim  # GSSM takes the previous state as input
        for parent in self.parents:
            fan_in *= parent.state_dim

        # Useful attributes for raveling multi-dimensional states
        self.state_dim = state_dim
        self.size_in = (state_dim, *(parent.state_dim for parent in self.parents))

        self.kernel = TransitionKernel(fan_in=fan_in, fan_out=state_dim, alphas=alphas, rng=rng)

        self.state = None
        self.time = None

    def initialize(self, bsz: int):
        """
        Initialize the state of the node.

        Parameters
        ----------
        bsz:
            Batch size
        """
        for parent in self.parents:
            if parent.time is None or parent.time != 0:
                parent.initialize(bsz)
        self.state = np.zeros(bsz, dtype=int)
        self.time = 0

    def evolve(self):
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

        input_state = np.vstack((self.state, *(parent.state for parent in self.parents)))
        input_state = np.ravel_multi_index(input_state, self.size_in)
        self.state = self.kernel(input_state)
        self.time += 1

    def __repr__(self):
        return f"Node(state_dim={self.state_dim}, state={self.state}, time={self.time}, nb_parents={len(self.parents)})"


# -------------------------------------------------------------------------------
# GSSM - Configuration and Building
# -------------------------------------------------------------------------------


@dataclass
class NodeConfig:
    name: str = ""
    state_dim: int = 0
    alpha: float = 0
    parents: list[str] = field(default_factory=list)

    def __manual_post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        if not self.name:
            raise ValueError(f"Node name must be specified. {self}")
        if self.state_dim == 0:
            raise ValueError(f"`state_dim` must be specified. {self}")
        if self.alpha == 0:
            raise ValueError(f"`alphas` must be a specified. {self}")


@dataclass
class GSSMConfig:
    nodes: list[NodeConfig] = field(default_factory=list)

    def __manual_post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        if not self.nodes:
            raise ValueError("At least one node must be specified.")
        for node in self.nodes:
            node.__manual_post_init__()


def build_gssm(config: GSSMConfig, rng: Generator = None) -> dict[str, Node]:
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
    nodes = {}

    while nodes_to_initialize:
        node_config = nodes_to_initialize.pop(0)
        parents_name = node_config.parents
        parents = []
        minimum = True
        for parent_name in parents_name:
            if parent_name not in nodes:
                minimum = False
                break
            parents.append(nodes[parent_name])

        if not minimum:
            logger.info(f"Parents of node {node_config} are not initialized yet")
            nodes_to_initialize.append(node_config)
            continue

        logger.info(f"Initializing node {node_config.name}")
        nodes[node_config.name] = Node(
            state_dim=node_config.state_dim, alphas=float(node_config.alpha), parents=parents, rng=rng
        )

    logger.info("Graph is built")
    return nodes


# -------------------------------------------------------------------------------
# DataLoader Manager - State and Configuration
# -------------------------------------------------------------------------------


@dataclass
class DataConfig:
    seq_len: int = -1
    batch_size: int = -1
    graph_seed: Optional[int] = None
    seed: Optional[int] = None
    gssm: GSSMConfig = field(default_factory=GSSMConfig)

    def __manual_post_init__(self):
        # Ensure decorrelated seeds
        if self.graph_seed is None and self.seed is None:
            ss = SeedSequence()
            seeds = ss.spawn(2)
            self.graph_seed = seeds[0]
            self.seed = seeds[1]


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


def init_dataloader_state(config: DataConfig) -> DataLoaderState:
    # Recover state from seeds
    graph_rng_state = default_rng(config.graph_seed).bit_generator.state
    rng_state = default_rng(config.seed).bit_generator.state
    return DataLoaderState(rng_state=rng_state, graph_rng_state=graph_rng_state)


# -------------------------------------------------------------------------------
# DataLoader Manager - Iterator
# -------------------------------------------------------------------------------


def get_batch(nodes: dict[str, Node], batch_size: int, seq_len: int) -> np.ndarray:
    """
    Generate batches of sentences.

    Parameters
    ----------
    nodes:
        Nodes of the graph-structured sequential model (GSSM).
    batch_size:
        The size of the batch.
    seq_len:
        The length of the sequence to generate.

    Yields
    ------
    np.ndarray
        The generated batch of sentences.
    """
    batch = np.empty((batch_size, seq_len), dtype=int)
    nodes["X"].initialize(batch_size)

    for t in range(seq_len):
        assert nodes["X"].time == t, f"Discrepancy in time: {nodes["X"].time} and {t}."
        nodes["X"].evolve()
        batch[:, t] = nodes["X"].state
    return batch


class DataLoaderManager:
    """
    Context manager for the data loader.

    Parameters
    ----------
    config:
        The configuration of the data loader.
    state:
        The state of the data loader.

    Attributes
    ----------
    rng:
        Random number generator.

    Yields
    ------
    tuple[np.ndarray, dict[str, Any]]
        The generated batch of sentences and the state of the random number generator.
    """

    def __init__(self, config: DataConfig, state: DataLoaderState):
        self.config = config
        self.state = state

        # track randomness
        self.rng = default_rng()

        # ensure consistency of transition kernels over restart
        self.rng.bit_generator.state = self.state.graph_rng_state
        self.nodes = build_gssm(self.config.gssm, rng=self.rng)
        assert "X" in self.nodes, "The graph must contain a node named 'X', acting as the observed node."

        # ensure randomness consistency
        self.rng.bit_generator.state = self.state.rng_state
        logger.debug(f"RNG: {self.state}")

    def __enter__(self):
        logger.info("Entering dataloader.")

        def iterator_func():
            while True:
                batch = get_batch(nodes=self.nodes, batch_size=self.config.batch_size, seq_len=self.config.seq_len)
                self.state.rng_state = self.rng.bit_generator.state
                yield batch

        return iterator_func()

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info("Exiting dataloader.")
        logger.debug(f"RNG: {self.state}")
