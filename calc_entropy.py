# %%
import numpy as np

# from omegaconf import OmegaConf
from nanollama.utils import initialize_nested_object
from src.nanollama.data import gssm
import numpy as np
import torch
from omegaconf import OmegaConf

# %%
gssm_config = {
    "nodes": [
        {
            "name": "Z1",
            "state_dim": 5,
            "parents": [],
            "alpha": 1e-8,
            "mode": "default",
        },
        {
            "name": "Z2",
            "state_dim": 6,
            "parents": ["Z1"],
            "alpha": 1e-8,
            "mode": "default",
        },
        {
            "name": "Z3",
            "state_dim": 7,
            "parents": ["Z2"],
            "alpha": 1e-8,
            "mode": "default",
        },
        {
            "name": "X",
            "state_dim": 8,
            "parents": ["Z1", "Z3"],
            "alpha": 1e-8,
            "mode": "default",
        },
    ]
}


# %%


class HMM:
    SYM_IN = "abcxdefghijklmnopqrstuvwxyz"
    SYM_OUT = "ABCXDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, config, random_seed=100):
        """
        makes an HMM from a graph config via the product state
        """
        self.config = initialize_nested_object(gssm.GSSMConfig, config, inplace=False)
        # self.config = OmegaConf.create(config)
        self.rng = np.random.default_rng(random_seed)
        self.nodes = gssm.build_gssm(self.config, self.rng)
        self.names = self._names_to_nodes(self.nodes)
        self.top_node = self.nodes["X"]  # WARNING assumption
        self.topo_order = self._dfs_names(self.top_node)
        self.indexs = {self.nodes[name]: i for i, name in enumerate(self.topo_order)}
        self.transitions = {node: self._format_transition(node) for node in self.nodes.values()}

    def evolve_classic(self, steps):
        for _ in range(steps):
            self.top_node.evolve()
        return {name: node.state for name, node in self.nodes.items()}

    def _node_init(self, node: gssm.Node, bsz, i=0):
        node.time = 0
        for parent in node.parents:
            if parent.time != 0 and not isinstance(parent, gssm.ObservedNode):
                self._node_init(parent, bsz, i + 1)
        node.state = np.zeros(bsz, dtype=int)
        # node.state = self.rng.integers(0, node.state_dim, size=bsz, dtype=int)

    def _init_all_nodes(self, bsz):
        self._node_init(self.top_node, bsz)

    @staticmethod
    def _names_to_nodes(nodes):
        names = {}
        for name, node in nodes.items():
            names[node] = name
        return names

    def _dfs_names(self, node):
        def __dfs_names(node, fc=True):
            if not fc and isinstance(node, gssm.ObservedNode):
                return [self.names[node]]
            return [d for p in node.parents for d in __dfs_names(p, False)] + [
                self.names[node]
            ]

        return list(dict.fromkeys(__dfs_names(node)))

    @staticmethod
    def _format_transition(node):
        parents = node.parents
        parent_state_dims = tuple([p.state_dim for p in parents])
        observed = isinstance(node, gssm.ObservedNode)
        trans = node.kernel.p_transition
        target_shape = tuple() if observed else (node.state_dim,)
        target_shape += parent_state_dims + (node.state_dim,)
        return trans.reshape(target_shape)

    def _node_sym_in(self, node):
        return HMM.SYM_IN[self.indexs[node]]

    def _node_sym_out(self, node):
        return HMM.SYM_OUT[self.indexs[node]]

    @staticmethod
    def _one_hot_state(node):
        targets = node.state
        nb_classes = node.state_dim
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

    def one_hot_product_state(self, node):
        node_order = self._dfs_names(node)
        einsum_str = (
            "B"
            + ",B".join(HMM.SYM_IN[: len(node_order)])
            + "->B"
            + HMM.SYM_IN[: len(node_order)]
        )
        product_state = np.einsum(
            einsum_str, *[self._one_hot_state(self.nodes[name]) for name in node_order]
        )
        return product_state.reshape(product_state.shape[0], -1)

    def product_state(self, tgt_node):
        return self.one_hot_product_state(tgt_node).argmax(-1)

    def individual_states(self, prod_state, node_order):
        state_dims = [self.nodes[name].state_dim for name in node_order]
        return np.unravel_index(prod_state, state_dims)

    @staticmethod
    def square_matrix(m):
        dim = int(np.prod(m.shape) ** 0.5)
        return m.reshape(dim, dim)

    def einsum_input_str(self, tgt_node):
        """
        Constructs the einsum string for the target node transition matrix when used as input
        """
        observed = self.names[tgt_node] == "X"
        einsum_str = self._node_sym_in(tgt_node) if not observed else ""
        einsum_str += "".join(self._node_sym_out(p) for p in tgt_node.parents)
        einsum_str += self._node_sym_out(tgt_node)
        return einsum_str

    def einsum_full_prod_str(self, tgt_node):
        """
        Constructs the einsum string for the target node transition matrix in the product state form
        """
        ordered_nodes = [self.nodes[name] for name in self._dfs_names(tgt_node)]
        in_str = "".join(self._node_sym_in(n) for n in ordered_nodes)
        out_str = "".join(self._node_sym_out(n) for n in ordered_nodes)
        return in_str + out_str

    def make_prod_transition(self, tgt_node):
        node_order = self._dfs_names(tgt_node)
        node_order = [self.nodes[name] for name in node_order]
        state_dims = [node.state_dim for node in node_order]
        prod_input_str = HMM.SYM_IN[: len(node_order)]

        def prod_str(node):
            return f"{prod_input_str}{self._node_sym_out(node)}"

        einsum_input_strs = {node: self.einsum_input_str(node) for node in node_order}
        einsum_prod_strs = {node: prod_str(node) for node in node_order}
        einsum_prod_strs[None] = prod_input_str
        # a collection of einsum_str -> array to use for the actual einsum
        # we build the logic only with the strings and fetch the matrices from here
        einsum_str_to_arr = {prod_input_str: np.ones(state_dims)}  # this is the root (for convenience)
        einsum_str_to_arr |= {s: self.transitions[node] for node, s in einsum_input_strs.items()}
        # step 1 : fill the abcxX matrices
        for node in node_order:
            parents = node.parents or [None]
            # this is good for Tabcx_A
            input_strs = [
                einsum_input_strs[node],
                *[einsum_prod_strs[p] for p in parents],
            ]
            output_str = einsum_prod_strs[node]
            einsum_str = ",".join(input_strs) + "->" + output_str
            input_tensors = [einsum_str_to_arr[s] for s in input_strs]
            einsum_str_to_arr[output_str] = np.einsum(einsum_str, *input_tensors)

        # step 2 : get final product transition
        input_strs = [einsum_prod_strs[node] for node in node_order]
        input_tensors = [einsum_str_to_arr[s] for s in input_strs]
        output_str = self.einsum_full_prod_str(tgt_node)
        einsum_str = ",".join(input_strs) + "->" + output_str

        return np.einsum(einsum_str, *input_tensors)

    def fwd_product_state(self, prod_transition):
        p_transition = self.square_matrix(prod_transition)
        kernel = gssm.TransitionKernel(*p_transition.shape, 1)
        kernel.p_transition = p_transition
        kernel._cumulative = np.cumsum(kernel.p_transition, axis=1)
        next_state = kernel(self.product_state(self.top_node))
        return self.individual_states(next_state, self.topo_order)

    def fwd_via_matmul(self, prod_transition):
        p_transition = self.square_matrix(prod_transition)
        state = self.one_hot_product_state(self.top_node)
        next_state = state @ p_transition
        # sampling
        samples = torch.multinomial(torch.tensor(next_state), 1).numpy()
        return self.individual_states(samples, self.topo_order)

    def get_p_emission(self, tgt_node):
        # p_emission is a projection matrix from the product state
        # state @ p_emission = state_X
        assert isinstance(tgt_node, gssm.ObservedNode)
        idx = self.indexs[tgt_node]
        shape = [1] * (len(self.topo_order) + 1)
        dim = tgt_node.state_dim
        shape[idx] = dim
        shape[-1] = dim
        p_emission = np.eye(dim).reshape(shape)
        for i, name in enumerate(self.topo_order):
            shape[i] = self.nodes[name].state_dim
        p_emission = np.broadcast_to(p_emission, shape)
        p_emission = p_emission.reshape(-1, dim)
        return p_emission

    def current_one_hot_product_state(self):
        return self.one_hot_product_state(self.top_node)

# %%
hmm = HMM(gssm_config)
hmm._init_all_nodes(4)
prod_transition = hmm.make_prod_transition(hmm.top_node)
p_emission = hmm.get_p_emission(hmm.top_node)
# %%
data_prod = hmm.fwd_product_state(prod_transition)
data_prod = {name: data_prod[i] for i, name in enumerate(hmm.topo_order)}
data_prod_mm = hmm.fwd_via_matmul(prod_transition)
data_prod_mm = {name: data_prod_mm[i] for i, name in enumerate(hmm.topo_order)}
data_classic = hmm.evolve_classic(1)
# %%
import matplotlib.pyplot as plt


for name in data_classic:
    plt.title(name)
    plt.hist(data_prod[name], label="prod", alpha=0.5)
    plt.hist(data_prod_mm[name], label="prod_mm", alpha=0.5)
    plt.hist(data_classic[name], label="classic", alpha=0.5)
    plt.legend()
    plt.show()

# %%



# %%
def forward_algorithm(
    observations: torch.tensor,
    log_A: torch.tensor,
    log_B: torch.tensor,
    log_pi: torch.tensor,
):
    """
    Perform the forward-backward algorithm to compute the forward and backward probabilities.
    S = hidden state vocab
    O = observation state vocab

    Args:
        observations (torch.tensor): List of observed sequences [seq_len,B] (batch last)
        A (torch.tensor): Transition matrix. [S, S]
        B (torch.tensor): Emission matrix. [S, O]
        pi (torch.tensor): Initial state probabilities. [S]
    Returns:
        forward_probs (torch.tensor): Forward probabilities [S, T, B]
    """
    DEVICE = "cpu"
    num_states = log_A.shape[0]
    T, B = observations.shape

    def lognorm(log_x):
        return log_x - torch.logsumexp(log_x, dim=0, keepdim=True)

    forward_probs = torch.zeros((num_states, T, B), device=DEVICE)

    forward_probs[:, 0, :] = lognorm(log_pi[:, None] + log_B[:, observations[0]])

    log_A = log_A[:, :, None]

    for t in range(1, T):
        forward_probs[:, t, :] = log_B[:, observations[t]] + torch.logsumexp(
            forward_probs[:, None, t - 1, :] + log_A, dim=0
        )
        forward_probs[:, t, :] = lognorm(forward_probs[:, t, :])

    return torch.exp(forward_probs)


def test_manual():
    A = torch.log(torch.tensor([[0, 1], [1, 0]]))
    B = torch.log(torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]))
    pi = torch.log(torch.tensor([0.5, 0.5]))
    observations = torch.tensor([[1], [0], [1], [0]])
    print(B.shape)
    p_zt_given_x_smaller_t = forward_algorithm(observations, A, B, pi)
    print(p_zt_given_x_smaller_t)


def hmm_forward_probs(config):
    hmm = HMM(config)
    batch_size = 2
    hmm._init_all_nodes(batch_size)
    prior = torch.tensor(hmm.current_one_hot_product_state()[0])
    prod_transition = hmm.make_prod_transition(hmm.top_node)
    prod_transition = torch.tensor(HMM.square_matrix(prod_transition))
    p_emission = torch.tensor(hmm.get_p_emission(hmm.top_node))
    seq_len = 5
    observations = torch.zeros((seq_len, batch_size), dtype=int)
    for i in range(seq_len):
        data_classic = hmm.evolve_classic(1)
        observations[i] = torch.tensor(data_classic["X"])
    forward_probs = forward_algorithm(
        observations, torch.log(prod_transition), torch.log(p_emission), torch.log(prior)
    )
    return forward_probs


forward_probs = hmm_forward_probs(gssm_config)
print(forward_probs.shape)
# %%
# ---------------------------------------------------------------------
# Vivien's code to compute the equivalent big HMM transition matrix
# Nik: use this later to simplify above.
# ---------------------------------------------------------------------


def get_transition_matrix(nodes: dict[str, gssm.Node]) -> np.ndarray:
    """
    Here is the logic I would use to compute the transition matrix.
    Not sure all my broadcasts and reshapes are correct though.

    In essence, it used the fact that for (Ai)_{i in N} N discrete variables,
    F(A1, A2, ..., AN) can be represented as a N-D tensor
    And a formula, e.g., with N=5,
        F(A1, A2, ..., A5) = F1(A1, A2, A3) * F2(A4, A5)
    can be computed by writting F1(A1, A2, A3) as a 3D tensor, broadcasting it to a 5-D tensor based on
        F1'(A1, A2, A3, A4, A5) = F1(A1, A2, A3)
    and similarly writting F2 as a 2D tensor being broadcast to a 5D with
        F2'(A1, A2, A3, A4, A5) = F2(A4, A5)
    and multiplying F1' and F2' element-wise.

    In our case, the discrete elements are the states of the nodes at time t and t-1.
    """

    size = []
    keys = {}
    for i, node in enumerate(nodes.values()):
        size.append(node.state_dim)
        keys[node] = i

    def _format_transition(node):
        observed = isinstance(node, gssm.ObservedNode)
        order = ([] if observed else [node]) + node.parents
        shape = [n.state_dim for n in order] + [node.state_dim]
        key_order = np.argsort([keys[n] for n in order] + [np.inf])
        trans = node.kernel.p_transition.reshape(shape)
        return trans.transpose(key_order)

    proba = np.ones((*size, *size))
    for name, node in nodes.items():
        input_shape = np.ones(len(nodes), dtype=int)
        output_shape = np.ones(len(nodes), dtype=int)

        output_shape[keys[node]] = node.state_dim
        if not isinstance(node, gssm.ObservedNode):
            input_shape[keys[node]] = node.state_dim

        for pnode in node.parents:
            input_shape[keys[pnode]] = pnode.state_dim
        print(name, input_shape, output_shape)

        # bug here, we should first reshape p_transition according to (node.state_dim, *(pnode.state_dim for pnode in node.parents))
        # then we should transpose it to have the same order as in proba.
        # EDIT: i think _format_transition accomplishes that, please check
        proba *= _format_transition(node).reshape((*input_shape, *output_shape))

    return proba


prod_transition_viv = get_transition_matrix(
    {name: hmm.nodes[name] for name in hmm.topo_order}
)
(prod_transition_viv == prod_transition).all()  # not quite right for now, needs correct time dependence
# %%
