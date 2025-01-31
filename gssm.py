import copy
from dataclasses import dataclass, field

import numpy as np


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
    kernel_type: str = "product"  # or fullrank
    observed: bool = False


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
    kernel_type:
        Which type of transition kernel? Can be 'product' or 'fullrank'.
    rng:
        Random number generator
    observed:
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
        name: str,
        state_dim: int,
        alpha: float,
        parents: list["Node"],
        mode: str,
        kernel_type: str,
        rng: np.random.Generator,
        observed: bool,
    ):
        self.name = name
        self.state_dim = state_dim
        self.alpha = alpha
        self.parents = parents
        self.mode = mode.lower()
        self.kernel_type = kernel_type.lower()
        self.rng = rng
        self.observed = observed

        self.consistency_checks()

        self.kernels = self.sample_transitions(alpha)

        self.state = None
        self.time = None

    def consistency_checks(self) -> None:
        """Check that the node is defined correctly"""
        assert self.name, f"Node name must be specified. {self}"
        assert self.state_dim, f"`state_dim` must be specified. {self}"
        assert self.alpha, f"`alpha` must be specified. {self}"
        assert self.mode in ["default", "slow", "dead", "context"], f"Invalid mode: {self.mode}"
        assert self.kernel_type.lower() in ["product", "fullrank"]
        assert not (self.mode == "slow" and self.observed), "Observed nodes cannot have slow transitions"
        assert not (self.mode == "dead" and self.observed), "Observed nodes cannot have dead transitions"
        assert not (self.mode == "slow" and len(self.parents) > 0), "Slow nodes cannot have parents"

    def sample_transitions(self, alpha: float) -> list[np.ndarray[float]]:
        """Sample transition kernels"""
        if self.kernel_type == "fullrank":
            return [self.sample_fullrank_transitions(alpha)]
        elif self.kernel_type == "product":
            return self.sample_product_transitions(alpha)

    def sample_fullrank_transitions(self, alpha: float) -> np.ndarray[float]:
        """Sample a full-rank transition kernel"""
        self.parents = self.parents if self.parents is not None else []

        # Calculate the fan_in based on the parents node.observed
        size_in = tuple() if self.observed else (self.state_dim,)
        for parent in self.parents:
            size_in += (parent.state_dim,)

        self.size_in = size_in
        p_transition = self.sample_transition(fan_in=np.prod(size_in).item(), alpha=alpha)
        self._cumulative = np.cumsum(p_transition, axis=1)
        return p_transition

    def sample_product_transitions(self, alpha: float) -> list[np.ndarray[float]]:
        """Initialize transition kernels"""
        # observed node does not have connection to itself
        if self.observed:
            fan_ins = []
        else:
            fan_ins = [self.state_dim]
        fan_ins += [pnode.state_dim for pnode in self.parents]

        return [self.sample_transition(fan_in=fan_in, alpha=alpha) for fan_in in fan_ins]

    def sample_transition(self, fan_in: int, alpha: float) -> np.ndarray[float]:
        """Sample transition kernel"""
        # transition
        alphas = np.full(self.state_dim, alpha)
        transition = self.rng.dirichlet(alphas, size=fan_in)

        # in the `dead` mode, argmax p(state[t+1] | ...) = 0
        if self.mode in ["dead", "slow"]:
            index = np.arange(fan_in)
            argmax = transition.argmax(axis=1)
            max_val = transition[index, argmax]
            if self.mode == "dead":
                transition[index, argmax] = transition[:, 0]
                transition[:, 0] = max_val
            else:
                transition[index, argmax] = transition[index, index]
                transition[index, index] = max_val

        np.clip(transition, a_min=1e-10, a_max=None, out=transition)  # avoid underflow
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
            if True:  # parent.time != 0:
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
        if self.time is None:
            raise RuntimeError("please initialize your node first")
        for parent in self.parents:
            if parent.time != self.time + 1:
                assert parent.time == self.time, "Parent node time is not correct."
                parent.evolve()

        if self.observed:
            all_states = []
        else:
            all_states = [self.state]

        all_states += [parent.state for parent in self.parents]

        self.state = self._get_kernel_probas(all_states)
        self.time += 1

    def _get_kernel_probas(self, all_states: list[np.ndarray[int]]) -> np.ndarray[int]:
        if self.kernel_type == "product":
            proba = np.prod([kernel[state] for kernel, state in zip(self.kernels, all_states)], axis=0)
            # Vectorized sampling
            random_values = self.rng.random(self.state.shape)
            proba.cumsum(axis=-1, out=proba)
            proba /= proba[..., -1:]

        elif self.kernel_type == "fullrank":
            input_state = np.vstack(all_states)
            input_state = np.ravel_multi_index(input_state, self.size_in)
            # Vectorized sampling
            random_values = self.rng.random(input_state.shape)
            proba = self._cumulative[input_state]

        return (random_values[:, None] < proba).argmax(axis=1)

    def __repr__(self):
        return "Node(" + ", ".join(
            [
                f"state_dim={self.state_dim}",
                f"state={self.state}",
                f"time={self.time}",
                f"nb_parents={len(self.parents)}",
                f"observed={self.observed})",
            ]
        )


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
    nodes_to_initialize = copy.deepcopy(config.nodes)
    nodes: dict[str, Node] = {}

    # initialize all nodes
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
            nodes_to_initialize.append(node_config)
            continue

        parents = []
        for parent_name in parents_name:
            parents.append(nodes[parent_name])

        nodes[node_config.name] = Node(
            name=node_config.name,
            state_dim=node_config.state_dim,
            alpha=float(node_config.alpha),
            parents=parents,
            mode=node_config.mode,
            kernel_type=node_config.kernel_type,
            rng=rng,
            observed=node_config.observed,
        )

        if node_config.observed:
            return nodes[node_config.name]

    raise ValueError("No observed node found")
