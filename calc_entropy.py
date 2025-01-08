# %%
from src.nanollama.data import gssm
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from src.apps.gssm.train import TrainingConfig
from collections import defaultdict

# %%
# file_config = OmegaConf.load("src/apps/gssm/configs/low_entropy.yaml")
# default_config = OmegaConf.structured(TrainingConfig())
# config = OmegaConf.merge(default_config, file_config)
# config = OmegaConf.to_object(config)
# config.__manual_post_init__()

# %%
gssm_config = OmegaConf.create(
    {
        "nodes": [
            {
                "name": "Z1",
                "state_dim": 2,
                "parents": ["X"],
                "alpha": 0.00001,
                "mode": "default",
            },
            {
                "name": "Z2",
                "state_dim": 3,
                "parents": ["Z1"],
                "alpha": 0.00001,
                "mode": "default",
            },
            {
                "name": "Z3",
                "state_dim": 5,
                "parents": ["Z2"],
                "alpha": 0.00001,
                "mode": "default",
            },
            {
                "name": "X",
                "state_dim": 4,
                "parents": ["Z1", "Z3"],
                "alpha": 0.00001,
                "mode": "default",
            },
        ]
    }
)

nodes = gssm.build_gssm(gssm_config, np.random.default_rng(100))

# %%
names = {}
indexs = {}
transitions = {}
for i, (name, node) in enumerate(nodes.items()):
    parents = node.parents
    parent_state_dims = tuple([p.state_dim for p in parents])
    observed = name == "X"
    trans = node.kernel.p_transition
    target_shape = tuple() if observed else (node.state_dim,)
    target_shape += parent_state_dims + (node.state_dim,)
    transitions[node] = trans.reshape(target_shape)
    names[node] = name
    indexs[node] = i
    print(name, target_shape, node.kernel.p_transition.shape)

# %%
SYM_IN = "abcdefghijklmnopqrstuvwxyz"
SYM_OUT = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def node_sym_in(node):
    return SYM_IN[indexs[node]]


def node_sym_out(node):
    return SYM_OUT[indexs[node]]


def dfs_names(node):
    def _dfs_names(node, fc=True):
        if not fc and isinstance(node, gssm.ObservedNode):
            return [names[node]]
        return [names[node]] + [d for p in node.parents for d in _dfs_names(p, False)]

    # unique
    ns = _dfs_names(node)
    return list(set(ns))


def einsum_input_str(tgt_node):
    """
    Constructs the einsum string for the target node transition matrix when used as input
    """
    observed = names[tgt_node] == "X"
    einsum_str = node_sym_in(tgt_node) if not observed else ""
    einsum_str += "".join(node_sym_in(p) for p in tgt_node.parents)
    einsum_str += node_sym_out(tgt_node)
    return einsum_str


def einsum_prod_str(tgt_node):
    """
    Constructs the einsum string for the target node transition matrix in the product state form
    """
    ordered_nodes = [nodes[name] for name in dfs_names(tgt_node)]
    in_str = "".join(node_sym_in(n) for n in ordered_nodes)
    # skip first if observed
    if names[tgt_node] == "X":
        in_str = in_str[1:]
    out_str = "".join(node_sym_out(n) for n in ordered_nodes)
    return in_str + out_str


# %%
dfs_names(nodes["X"])
# %%
einsum_input_str(nodes["X"]), einsum_input_str(nodes["Z1"]), einsum_input_str(
    nodes["Z2"]
)
# %%
einsum_prod_str(nodes["X"]), einsum_prod_str(nodes["Z1"]), einsum_prod_str(nodes["Z2"])


# %%
def get_product_transition(tgt_node, is_root_call=True):
    """
    Constructs the total transition matrix for the full product state
    """
    src_sds = tuple([p.state_dim for p in tgt_node.parents])
    if len(src_sds) == 0:
        return transitions[tgt_node]
    if isinstance(tgt_node, gssm.ObservedNode) and not is_root_call:
        return transitions[
            tgt_node
        ]  # in the root call we want to calculate the product state of the observed node, but not after.
    else:
        parent_transs = [get_product_transition(p, False) for p in tgt_node.parents]
        # now comes the einsum
        # transitions[tgt_node] is of shape [(self.state_dim, *parent_state_dims) x self.state_dim]
        # parents einsums
        parent_einsums = [
            (
                einsum_prod_str(p)
                if not isinstance(p, gssm.ObservedNode)
                else einsum_input_str(p) # if X is a parent, don't modify that (it is not producted yet)
            )
            for p in tgt_node.parents
        ]
        inputs = [einsum_input_str(tgt_node), *parent_einsums]
        einsum_str = ",".join(inputs) + "->" + einsum_prod_str(tgt_node)
        print(names[tgt_node], einsum_str)
        prod = np.einsum(einsum_str, transitions[tgt_node], *parent_transs)
        if isinstance(tgt_node, gssm.ObservedNode):
            prod = np.broadcast_to(prod[None], (tgt_node.state_dim, *prod.shape))
        return prod
        # T = np.einsum("toT,oO->toTO", transitions[tgt_node], *parent_transs)
        # T21_21 = np.einsum("toT,oO->toTO", T21_2,T1_1)
        # TX21_X21_alt = np.einsum("otX,oO,toT->toXTO", T12_X, T1_1, T21_2)
        # s (if not observed) (p1 p2 .. Os) , (p1 p1p1 p1p2 .. Op1) , (p2 p2p1 p2p2 ... Op2), ... -> p1 p2 .. Op1 Op2 .. Os
        # 012,13->0123
        # return T


get_product_transition(nodes["X"]).shape
# %%


# %%
transitions
# %%
# T1_1; aA -> T1_1; aA
# T21_2; baB -> T21_21; baBA
# T32_3; cbC -> T321_321; cbaCBA


# %%
T1_1 = transitions[nodes["Z1"]]
T21_2 = transitions[nodes["Z2"]]
T12_X = transitions[nodes["X"]]
T21_21 = np.einsum("toT,oO->toTO", T21_2, T1_1)
TX21_X21: np.ndarray = np.einsum("otX,toTO->toXTO", T12_X, T21_21)
# repeat to make xtoXTO shape
TX21_X21 = np.broadcast_to(TX21_X21[None], (nodes["X"].state_dim, *TX21_X21.shape))
TX21_X21_alt: np.ndarray = np.einsum("toT,oO,otX->toXTO", T21_2, T1_1, T12_X)
TX21_X21_alt = np.broadcast_to(
    TX21_X21_alt[None], (nodes["X"].state_dim, *TX21_X21_alt.shape)
)
# %%
(TX21_X21 == TX21_X21_alt).all()
# %%
np.einsum("i->ii", np.ones(2))

# %%
nodes["X"].name


# %%
def init(node, bsz):
    for parent in node.parents:
        if parent.time != 0 and not isinstance(parent, gssm.ObservedNode):
            init(parent, bsz)
    node.state = np.zeros(bsz, dtype=int)
    node.time = 0


init(nodes["X"], 10)
nodes["Z1"].state, nodes["X"].state
# %%
for i in range(4):
    nodes["X"].evolve()
    print(nodes["Z1"].state, nodes["X"].state)
