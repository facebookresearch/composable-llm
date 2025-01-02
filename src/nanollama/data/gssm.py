"""
Graph Structured Sequential Model (GSSM)

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import logging
import zlib
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from queue import Empty, Full
from types import TracebackType
from typing import Any, Union

import numpy as np
import torch
from numpy.random import Generator, SeedSequence, default_rng
from scipy.stats import dirichlet

from ..cluster import get_rank

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
    alphas:
        Dirichlet prior parameter. Can be a float, int, list of floats, or a numpy array of shape (fan_out,).
    mode:
        Mode of the transition kernel. Can be 'default', 'slow', 'dead', or 'context'.
    rng:
        Random number generator.

    Attributes
    ----------
    p_transition:
        A fan_in x fan_out transition matrix.
    """

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        alphas: Union[int, float, list[float], np.ndarray],
        mode: str = "default",
        rng: Generator = None,
    ):
        # handle various types for concentration parameters
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

        self.mode = mode.lower()

        # in the `context` mode, the transition matrix change each time
        if self.mode == "context":
            self.p_transition = None
            self.alphas = alphas
            self.fan_in = fan_in
            return

        self.p_transition = dirichlet.rvs(alphas, size=fan_in, random_state=rng)

        # in the `slow` mode, argmax p(state[t+1] | state[t], parents) = x
        if self.mode == "slow":
            index = np.arange(fan_in)
            size_in = (fan_out, fan_in // fan_out)
            new_argmax = np.unravel_index(index, size_in)[0]
            argmax = self.p_transition.argmax(axis=1)
            max_val = self.p_transition[index, argmax]
            self.p_transition[index, argmax] = self.p_transition[index, new_argmax]
            self.p_transition[index, new_argmax] = max_val

        # in the `dead` mode, argmax p(state[t+1] | ...) = 0
        elif self.mode == "dead":
            index = np.arange(fan_in)
            argmax = self.p_transition.argmax(axis=1)
            max_val = self.p_transition[index, argmax]
            self.p_transition[index, argmax] = self.p_transition[:, 0]
            self.p_transition[:, 0] = max_val

        else:
            assert self.mode == "default", f"Unknown mode: {mode}."

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

        # in-context learning mode
        if self.p_transition is None:
            p_transition = dirichlet.rvs(self.alphas, size=self.fan_in, random_state=self.rng)
            p_cumulative = np.cumsum(p_transition, axis=1)[state]
        else:
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
        parents: list["Node"] = None,
        mode: str = "default",
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

        self.kernel = TransitionKernel(fan_in=fan_in, fan_out=state_dim, alphas=alphas, mode=mode, rng=rng)

        self.state = None
        self.time = None

        # set random number generator
        if rng is None:
            rng = default_rng()
        self.rng = rng

    def initialize(self, bsz: int) -> None:
        """
        Initialize the state of the node.

        Parameters
        ----------
        bsz:
            Batch size
        """
        for parent in self.parents:
            if parent.time != 0 and not isinstance(parent, ObservedNode):
                parent.initialize(bsz)
        # self.state = self.rng.integers(0, self.state_dim, bsz, dtype=int)
        self.state = np.zeros(bsz, dtype=int)
        self.time = 0

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
            if parent.time != self.time + 1 and not isinstance(parent, ObservedNode):
                assert parent.time == self.time, "Parent node time is not correct."
                parent.evolve()

        input_state = self.get_input_state()
        input_state = np.ravel_multi_index(input_state, self.size_in)
        self.state = self.kernel(input_state)
        self.time += 1

    def get_input_state(self) -> np.ndarray:
        input_state = np.vstack((self.state, *(parent.state for parent in self.parents)))
        return input_state

    def __repr__(self):
        return "Node(" + " ,".join(
            [
                f"state_dim={self.state_dim}",
                f"state={self.state}",
                f"time={self.time}",
                f"nb_parents={len(self.parents)}",
                f"mode={self.kernel.mode}",
            ]
        )


class ObservedNode(Node):
    """
    Observed node in a graph-structured sequential model (GSSM).

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
    """

    def __init__(
        self,
        state_dim: int,
        alphas: Union[int, float, list[float], np.ndarray],
        parents: list["Node"] = None,
        mode: str = "default",
        rng: Generator = None,
    ):
        self.reinit(state_dim, alphas, parents, mode, rng)

    def reinit(
        self,
        state_dim: int,
        alphas: Union[int, float, list[float], np.ndarray],
        parents: list["Node"] = None,
        mode: str = "default",
        rng: Generator = None,
    ) -> None:
        self.parents = parents if parents is not None else []

        # Calculate the fan_in based on the parents
        fan_in = 1  # Observed node do not take the previous state as input
        for parent in self.parents:
            fan_in *= parent.state_dim

        # Useful attributes for raveling multi-dimensional states
        self.state_dim = state_dim
        self.size_in = (*(parent.state_dim for parent in self.parents),)

        self.kernel = TransitionKernel(fan_in=fan_in, fan_out=state_dim, alphas=alphas, mode=mode, rng=rng)

        self.state = None
        self.time = None

        # set random number generator
        if rng is None:
            rng = default_rng()
        self.rng = rng

    def get_input_state(self) -> np.ndarray:
        input_state = np.vstack((*(parent.state for parent in self.parents),))
        return input_state

    def __repr__(self):
        return "ObservedNode(" + " ,".join(
            [
                f"state_dim={self.state_dim}",
                f"state={self.state}",
                f"time={self.time}",
                f"nb_parents={len(self.parents)}",
                f"mode={self.kernel.mode}",
            ]
        )


# -------------------------------------------------------------------------------
# GSSM - Configuration and Building
# -------------------------------------------------------------------------------


@dataclass
class NodeConfig:
    name: str = ""
    state_dim: int = 0
    alpha: float = 0
    mode: str = "default"
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
    nodes_to_initialize: list[NodeConfig] = []
    nodes: dict[str, Node] = {}

    # start by initializing the observed node without parents
    for node_config in config.nodes:
        if node_config.name == "X":
            logger.info("Initializing observed node")
            observed_config = node_config
            nodes["X"] = ObservedNode(state_dim=node_config.state_dim, alphas=1)
        else:
            nodes_to_initialize.append(node_config)

    # observe all the other nodes
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
            state_dim=node_config.state_dim,
            alphas=float(node_config.alpha),
            parents=parents,
            mode=node_config.mode,
            rng=rng,
        )

    # set the parents of the observed node
    node_config = observed_config
    parents_name = node_config.parents
    parents = []
    minimum = True
    for parent_name in parents_name:
        assert parent_name in nodes
        parents.append(nodes[parent_name])

    logger.info("Reinitializing observed node")
    nodes[node_config.name].reinit(
        state_dim=node_config.state_dim,
        alphas=float(node_config.alpha),
        parents=parents,
        mode=node_config.mode,
        rng=rng,
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
    seed: int = 0
    gssm: GSSMConfig = field(default_factory=GSSMConfig)
    asynchronous: bool = True  # asynchronous data loading
    buffer_size: int = 4  # number of batches to bufferize asynchronously for data loading


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


# -------------------------------------------------------------------------------
# DataLoader Manager - Iterator
# -------------------------------------------------------------------------------


class OnlineDataLoaderManager:
    """
    Context manager for the online data loader.

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
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.asynchronous = config.asynchronous

        # track randomness
        self.rng = default_rng()

        # ensure consistency of transition kernels over restart
        self.rng.bit_generator.state = state.graph_rng_state
        self.nodes = build_gssm(config.gssm, rng=self.rng)
        assert "X" in self.nodes, "The graph must contain a node named 'X', acting as the observed node."

        # ensure randomness consistency
        self.rng.bit_generator.state = state.rng_state
        logger.debug(f"RNG: {state}")

        # asynchronous data loader: a worker writes batches in a buffer, that a reader consumes
        if self.asynchronous:
            self.process = Process(target=self.async_create_batch)
            self.buffer = Queue(maxsize=config.buffer_size)

    def __enter__(self):
        logger.info("Entering dataloader.")
        if self.asynchronous:
            self.process.start()
        return self

    def get_batch(self) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Generate a batch of sentences.
        """
        batch = np.empty((self.batch_size, self.seq_len), dtype=int)
        self.nodes["X"].initialize(self.batch_size)

        for t in range(self.seq_len):
            assert self.nodes["X"].time == t, f"Discrepancy in time: {self.nodes["X"].time} and {t}."
            self.nodes["X"].evolve()
            batch[:, t] = self.nodes["X"].state

        restart_info = self.rng.bit_generator.state

        return batch, restart_info

    def async_create_batch(self) -> None:
        """
        Asynchronous batch generation, writting batches to the buffer.
        """
        # loop on batch creation
        while True:
            batch, restart_info = self.get_batch()
            # convert to torch tensor
            batch = torch.from_numpy(batch).long()

            # put it in the buffer
            while True:
                try:
                    self.buffer.put((batch, restart_info), timeout=0.1)
                    break
                # if the buffer is full, wait until there is space
                except Full:
                    logger.debug("Buffer is full. Waiting for data comsumption.")
            logger.debug("New batch put in the buffer.")

    def async_get_batch(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Asynchronous batch acquisition, reading batches from the buffer.
        """
        # read batch from the buffer
        while True:
            try:
                return self.buffer.get(timeout=0.1)
            # if the buffer is full, wait until it is filled
            except Empty:
                logger.debug("Buffer is empty. Waiting for data.")

    def __next__(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.asynchronous:
            return self.async_get_batch()
        else:
            batch, restart_info = self.get_batch()
            batch = torch.from_numpy(batch).long()
            return (batch, restart_info)

    def __exit__(
        self,
        exc: type[BaseException],
        value: BaseException,
        tb: TracebackType,
    ):
        logger.info("Exiting dataloader.")
        if self.asynchronous:
            self.process.kill()
            self.buffer.close()


def estimate_seq_entropy_by_compression(seq: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Estimates the entropy of a sequence (np.ndarray/torch.tensor) via zlib compression.

    Parameters:
    - seq: NumPy ndarray of shape (sequence_length,)

    Returns:
    - entropy_estimate: float
    """
    if isinstance(seq, torch.Tensor):
      seq = seq.cpu().numpy()

    seq = seq.tobytes()
    compressed_data = zlib.compress(seq, level=9)

    # Calculate entropy as compressed size / original size
    original_size = len(seq)
    return len(compressed_data) / original_size if original_size > 0 else 0


if __name__ == "__main__":
    # Example usage
    batch = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 2, 3, 4, 5, 6, 7],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 5, 1, 2, 8, 3, 6],
        ]
    )

    entropy_estimates = list(map(estimate_seq_entropy_by_compression, batch))
    print(entropy_estimates)
