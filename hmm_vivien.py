# %%
from src.nanollama.data import gssm
from src.apps.gssm.hidden_markov_model import HMM
import numpy as np
# ---------------------------------------------------------------------
# Vivien's code to compute the equivalent big HMM transition matrix
# Nik: use this later to simplify above, atm buggy
# ---------------------------------------------------------------------
def _dfs(node):
    def __dfs(node, fc=True):
        if not fc and node.observed:
            return [(node.name, node)]
        return [d for p in node.parents for d in __dfs(p, False)] + [(node.name, node)]

    return list(dict.fromkeys(__dfs(node)))


def __get_transition_matrix(top_node) -> np.ndarray:
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
    node_order = _dfs(top_node)
    
    for i, (name, node) in enumerate(node_order):
        size.append(node.state_dim)
        keys[node] = i

    @staticmethod
    def _join_parent_kernels(node):
        n_in = len(node.kernels)
        # p(x|a), p(x|b) -> p(x|ab)
        # print("individual:\n", [x.sum(-1) for x in node.kernels])
        einsum_str = ",".join([f"{HMM.SYM_IN[i]}X" for i in range(n_in)])
        einsum_str += "->" + "".join([HMM.SYM_IN[i] for i in range(n_in)]) + "X"
        trans = np.einsum(einsum_str, *node.kernels)
        trans[trans.sum(-1) == 0] = 1 / node.state_dim
        trans = trans / trans.sum(-1, keepdims=True)
        # print("prod_trans: \n", trans)
        return trans

    @staticmethod
    def _format_transition(node):
        parents = node.parents
        parent_state_dims = tuple([p.state_dim for p in parents])
        trans = _join_parent_kernels(node)
        # target_shape = tuple() if node.observed else (node.state_dim,)
        # FIXME
        target_shape = (node.state_dim,)
        target_shape += parent_state_dims + (node.state_dim,)
        return trans.reshape(target_shape)

    proba = np.ones((*size, *size))
    for name, node in node_order:
        input_shape = np.ones(len(node_order), dtype=int)
        output_shape = np.ones(len(node_order), dtype=int)

        output_shape[keys[node]] = node.state_dim
        # if not node.observed:
        #     input_shape[keys[node]] = node.state_dim
        input_shape[keys[node]] = node.state_dim

        for pnode in node.parents:
            input_shape[keys[pnode]] = pnode.state_dim
        print(name, input_shape, output_shape)

        # bug here, we should first reshape p_transition according to (node.state_dim, *(pnode.state_dim for pnode in node.parents))
        # then we should transpose it to have the same order as in proba.
        # EDIT: i think _format_transition accomplishes that, please check
        proba *= _format_transition(node).reshape((*input_shape, *output_shape))

    return proba

# %%
gssm_config = {
    "nodes": [
        {
            "name": "Z1",
            "state_dim": 2,
            "parents": [],
            "alpha": .1,
            "mode": "default",
            "observed": False,
        },
        {
            "name": "X",
            "state_dim": 3,
            "parents": ["Z1"],
            "alpha": .05,
            "mode": "default",
            "observed": True,
        },
    ]
}
# %%
hmm = HMM(gssm_config)
# %%
hmm.square_matrix(__get_transition_matrix(hmm.top_node))
# %%
hmm.make_prod_transition(hmm.top_node)
# %%
