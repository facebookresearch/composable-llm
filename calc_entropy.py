# %%
import numpy as np
from omegaconf import OmegaConf

from nanollama.utils import initialize_nested_object
from src.nanollama.data import gssm

# %%
gssm_config = {
    "nodes": [
        {
            "name": "Z1",
            "state_dim": 8,
            "parents": [],
            "alpha": 1e-8,
            "mode": "default",
        },
        {
            "name": "Z2",
            "state_dim": 8,
            "parents": ["Z1"],
            "alpha": 1e-8,
            "mode": "default",
        },
        {
            "name": "Z3",
            "state_dim": 8,
            "parents": ["Z2"],
            "alpha": 1e-8,
            "mode": "default",
        },
        {
            "name": "X",
            "state_dim": 8,
            "parents": ["Z3"],
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
        self.config = OmegaConf.create(config)
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
        node.state = np.random.randint(0, 5, size=bsz, dtype=int)  # (np.arange(bsz, dtype=int) + i)[::-1]

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
            return [d for p in node.parents for d in __dfs_names(p, False)] + [self.names[node]]

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
    def one_hot_state(node):
        targets = node.state
        nb_classes = node.state_dim
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

    def one_hot_product_state(self, node):
        node_order = self._dfs_names(node)
        einsum_str = "B" + ",B".join(HMM.SYM_IN[: len(node_order)]) + "->B" + HMM.SYM_IN[: len(node_order)]
        product_state = np.einsum(einsum_str, *[self.one_hot_state(self.nodes[name]) for name in node_order])
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
        # x,aA,b,c->xabcA
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


# %%
hmm = HMM(gssm_config)
hmm._init_all_nodes(10000)
prod_transition = hmm.make_prod_transition(hmm.top_node)
data_prod = hmm.fwd_product_state(prod_transition)
data_prod = {name: data_prod[i] for i, name in enumerate(hmm.topo_order)}
data_classic = hmm.evolve_classic(1)
# %%

# %%
import matplotlib.pyplot as plt

for name in data_classic:
    plt.title(name)
    plt.hist(data_prod[name], label="prod", alpha=0.5)
    plt.hist(data_classic[name], label="classic", alpha=0.5)
    plt.legend()
    plt.show()


# %%


# %%
def manual(hmm: HMM):
    Ta_A = hmm.transitions[hmm.nodes["Z1"]]
    TbA_B = hmm.transitions[hmm.nodes["Z2"]]
    TcB_C = hmm.transitions[hmm.nodes["Z3"]]
    TABC_X = hmm.transitions[hmm.nodes["X"]]

    one_A = np.ones(hmm.nodes["Z1"].state_dim)
    one_B = np.ones(hmm.nodes["Z2"].state_dim)
    one_C = np.ones(hmm.nodes["Z3"].state_dim)
    one_X = np.ones(hmm.nodes["X"].state_dim)

    Tabcx_1 = np.einsum("a,b,c,x->abcx", one_A, one_B, one_C, one_X)
    Tabcx_A = np.einsum("aA,abcx->abcxA", Ta_A, Tabcx_1)
    Tabcx_B = np.einsum("bAB,abcxA->abcxB", TbA_B, Tabcx_A)
    Tabcx_C = np.einsum("cBC,abcxB->abcxC", TcB_C, Tabcx_B)
    Tabcx_X = np.einsum("ABCX,abcxA,abcxB,abcxC->abcxX", TABC_X, Tabcx_A, Tabcx_B, Tabcx_C)
    # output product_state
    Tabcx_ABCX = np.einsum("abcxX,abcxA,abcxB,abcxC->abcxABCX", Tabcx_X, Tabcx_A, Tabcx_B, Tabcx_C)

    print(
        hmm.individual_states(
            (hmm.one_hot_product_state(hmm.nodes["X"]) @ hmm.square_matrix(Tabcx_ABCX)).argmax(-1),
            hmm.topo_order,
        )
    )
    return Tabcx_ABCX


reference_prod_transition = manual()


# %%

# ---------------------------------------------------------------------
# Vivien's code to compute the equivalent big HMM transition matrix
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
        keys[id(node)] = i

    proba = np.ones((*size, *size))
    for name, node in nodes.items():
        input_shape = np.ones(len(nodes), dtype=int)
        output_shape = np.ones(len(nodes), dtype=int)

        output_shape[keys[id(node)]] = node.state_dim
        if name != "X":
            input_shape[keys[id(node)]] = node.state_dim

        for pnode in node.parents:
            input_shape[keys[id(pnode)]] = pnode.state_dim

        transition = node.kernel.p_transition.reshape((*input_shape, *output_shape))
        proba *= transition

    return proba


nodes = gssm.build_gssm(initialize_nested_object(gssm.GSSMConfig, gssm_config), None)
proba = get_transition_matrix(nodes)
