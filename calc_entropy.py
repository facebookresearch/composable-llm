# %%
from src.nanollama.data import gssm
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from src.apps.gssm.train import TrainingConfig
from collections import defaultdict
from einops import rearrange

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
                "state_dim": 5,
                "parents": [],
                "alpha":0.00001,
                "mode": "default",
            },
            {
                "name": "Z2",
                "state_dim": 6,
                "parents": ["Z1"],
                "alpha":0.00001,
                "mode": "default",
            },
            # {
            #     "name": "Z3",
            #     "state_dim": 10,
            #     "parents": ["Z2"],
            #     "alpha":0.00001,
            #     "mode": "default",
            # },
            {
                "name": "X",
                "state_dim": 4,
                "parents": ["Z1", "Z2"],
                "alpha":0.00001,
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
    return list(dict.fromkeys(ns))


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
# the recipe for the next two functions
# T1_1 = transitions[nodes["Z1"]]
# T21_2 = transitions[nodes["Z2"]]
# T12_X = transitions[nodes["X"]]

# building T1_1
# trivial, exists already

# building T21_21
# via get_combined_transition
# T21_2 = np.einsum("baB,Aa->bAB", T21_2, T1_1)

# via get_product_transition
# T21_21 = np.einsum("baB,aA->baBA", T21_2, T1_1)

#building TX12_X12

# via get_combined_transition
# T12_X = np.einsum("abX,Aa->AbX", T12_X, T1_1)
# T12_X = np.einsum("abX,Bab->aBX", T12_X, T21_2)

# via get_product_transition
# T12_X12 = np.einsum("abX,baBA->abXAB", T12_X, T21_21)
# TX12_X12 = np.broadcast_to(T12_X12[None], (nodes["X"].state_dim, *T12_X12.shape))

# %%
def get_combined_transition(tgt_node, is_root_call=True):
    """
    example graph
    Z0 -> Z1
    v     v
    X0    X1


    known:
    P(X0|Z0) = P(X1|Z1), P(Z1|Z0)
    want:
    P(Z1|Z0,X0) (already known)
    P(X1|Z0,X0) = P(X1|Z0) = sum_Z1 P(X1|Z1) P(Z1|Z0) <- this is the output of this function
    """
    src_sds = tuple([p.state_dim for p in tgt_node.parents])
    if len(src_sds) == 0:
        return transitions[tgt_node]
    if isinstance(tgt_node, gssm.ObservedNode) and not is_root_call:
        return transitions[
            tgt_node
        ]  # in the root call we want to calculate the product state of the observed node, but not after.
    else:
      #abX
      def make_output_str(inp:str, i:int, observed:bool):
        inp = list(inp)
        idx = i + (not observed)
        inp[idx] = inp[idx].upper()
        return "".join(inp)

      def einsum_comb_str(node):
        s = list(einsum_input_str(node))
        s[0], s[-1] = s[-1], s[0]
        return "".join(s)

      node_input_str = einsum_input_str(tgt_node)
      node_comb_trans = transitions[tgt_node]
      for i_parent, parent in enumerate(tgt_node.parents):
        parent_comb_trans = get_combined_transition(parent,False)
        parent_comb_str = einsum_comb_str(parent)
        node_output_str = make_output_str(node_input_str, i_parent, isinstance(tgt_node, gssm.ObservedNode))
        einsum_str = f"{node_input_str},{parent_comb_str}->{node_output_str}"
        print("comb", names[parent], einsum_str)
        node_comb_trans = np.einsum(einsum_str, node_comb_trans, parent_comb_trans)
      return node_comb_trans

get_combined_transition(nodes["X"]).shape

# %%

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
        print("prod", names[tgt_node], einsum_str)
        prod = np.einsum(einsum_str, get_combined_transition(tgt_node), *parent_transs)
        if isinstance(tgt_node, gssm.ObservedNode):
            prod = np.broadcast_to(prod[None], (tgt_node.state_dim, *prod.shape))
        return prod


prod_transition = get_product_transition(nodes["X"])

# %%
# T1_1; aA -> T1_1; aA
# T21_2; baB -> T21_21; baBA
# T32_3; cbC -> T321_321; cbaCBA


# %%
# T1_1 = transitions[nodes["Z1"]]
# T21_2 = transitions[nodes["Z2"]]
# T12_X = transitions[nodes["X"]]
# T21_21 = np.einsum("toT,oO->toTO", T21_2, T1_1)
# TX21_X21: np.ndarray = np.einsum("otX,toTO->toXTO", T12_X, T21_21)
# # repeat to make xtoXTO shape
# TX21_X21 = np.broadcast_to(TX21_X21[None], (nodes["X"].state_dim, *TX21_X21.shape))
# TX21_X21_alt: np.ndarray = np.einsum("toT,oO,otX->toXTO", T21_2, T1_1, T12_X)
# TX21_X21_alt = np.broadcast_to(
#     TX21_X21_alt[None], (nodes["X"].state_dim, *TX21_X21_alt.shape)
# )
# # %%
# (TX21_X21 == TX21_X21_alt).all()
# # %%
# np.einsum("i->ii", np.ones(2))

# # %%
# nodes["X"].name


# %%
def init(node, bsz, i=0):
    node.time = 0
    for parent in node.parents:
        if parent.time != 0 and not isinstance(parent, gssm.ObservedNode):
            init(parent, bsz, i+1)
    node.state = np.arange(bsz, dtype=int) + i


init(nodes["X"], 4)
{name: node.state for name,node in nodes.items()}
# %%
for i in range(1):
    nodes["X"].evolve()
    print({name: node.state for name,node in nodes.items()})

# %%
nodes["X"].kernel.p_transition.shape, nodes["X"].state.shape
# %%
def one_hot_state(node):
    targets = node.state
    nb_classes = node.state_dim
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def product_state_oh(tgt_node):
  node_order = dfs_names(tgt_node)
  einsum_str = "B" + ",B".join(SYM_IN[:len(node_order)]) + "->B" + SYM_IN[:len(node_order)]
  print(einsum_str)
  product_state = np.einsum(einsum_str, *[one_hot_state(nodes[name]) for name in node_order])
  return product_state.reshape(product_state.shape[0], -1)

def product_state(tgt_node):
  prod_state = product_state_oh(tgt_node)
  prod_state = prod_state.argmax(-1)
  return prod_state

def individual_states(prod_state, node_order):
  state_dims = [nodes[name].state_dim for name in node_order]
  state_idxs = []
  for dim in state_dims[::-1]:
    state_idxs.append(prod_state % dim)
    prod_state = prod_state // dim
  return state_idxs[::-1]

state = product_state(nodes["X"])
print(state)
individual_states(state, dfs_names(nodes["X"]))
# %%
dfs_names(nodes["X"])
# %%

# %%
print({name: node.state for name,node in nodes.items()})
prod_dim=int(np.prod(prod_transition.shape)**.5)
kernel = gssm.TransitionKernel(prod_dim, prod_dim, 1)
kernel.p_transition = prod_transition.reshape(prod_dim, prod_dim)
kernel._cumulative = np.cumsum(kernel.p_transition, axis=1)
next_state = kernel(product_state(nodes["X"]))
individual_states(next_state, dfs_names(nodes["X"]))
# %%
prod_transition.shape

# %%

# %%
def evolve_prod(T, state):
  # 1. reshape the transition matrix into square
  prod_dim = int(np.prod(T.shape)**.5)
  T = T.reshape(prod_dim, prod_dim)
  # 2. make product state
  # 3. evolve product state
  next_state = state @ T
  # 4. split up into individuals to check
  return next_state

state = product_state_oh(nodes["X"])
next_state = evolve_prod(prod_transition, state)
# %%
next_state.shape

# %%
nodes["Z1"].kernel.p_transition
# %%
nodes["X"].kernel.p_transition
# %%
prod_transition[0]
# %%
product_state(nodes["X"])
# %%
(product_state_oh(nodes["X"]) @ prod_transition.reshape(12,12)).argmax(-1)
# %%
nodes["X"].state
# %%
def fwd(node: gssm.Node, input_state: np.ndarray = None):
  if input_state is None:
    input_state = node.get_input_state()
    input_state = np.ravel_multi_index(input_state, node.size_in)
  state = input_state
  # Convert state to a numpy array if it's a list
  if isinstance(state, list):
      state = np.asarray(state)

  # Vectorized sampling
  random_values = np.random.default_rng().random(state.shape)

  # in-context learning mode
  _cumulative = np.cumsum(node.kernel.p_transition, axis=1)
  p_cumulative = _cumulative[state]
  return (random_values[:, None] < p_cumulative).argmax(axis=1)

z2_state = nodes["Z2"].get_input_state()
z2_state[1] = fwd(nodes["Z1"])
z2_state = np.ravel_multi_index(z2_state, nodes["Z2"].size_in)
fwd(nodes["Z2"], input_state=z2_state)
# fwd(nodes["Z1"], input_state=fwd(nodes["Z1"]))
# fwd(nodes["X"])

# %%
print({name: node.state for name,node in nodes.items()})
# %%
def square(m):
   dim = int(np.prod(m.shape)**.5)
   return m.reshape(dim,dim)
# %%
T1_1 = transitions[nodes["Z1"]]
T21_2 = transitions[nodes["Z2"]]
T21_2 = np.einsum("baB,Aa->bAB", T21_2, T1_1)
T21_21 = np.einsum("baB,aA->baBA", T21_2, T1_1)
T12_X = transitions[nodes["X"]]
# correct:
# T21_X = np.einsum("baX,BAba->BAX", T21_X,T21_21)
#also good:
# T21_X = np.einsum("abX->baX",T12_X)
# T21_X = np.einsum("baX,Aa->bAX", T21_X, T1_1)
# T21_X = np.einsum("baX,Bab->BaX", T21_X, T21_2)
# T12_X12 = np.einsum("baX,baBA->abXAB", T21_X, T21_21)

T12_X = np.einsum("abX,Aa->AbX", T12_X, T1_1)
T12_X = np.einsum("abX,Bab->aBX", T12_X, T21_2)
T12_X12 = np.einsum("abX,baBA->abXAB", T12_X, T21_21)

TX12_X12 = np.broadcast_to(T12_X12[None], (nodes["X"].state_dim, *T12_X12.shape))

# T21_X = np.einsum("baX,Aa,baBA->BAX", T21_X, T1_1, T21_21)
# TX21_X21 = 0
# TX1_X1 = np.einsum("aB,aA->aBA", T1_X, T1_1)
# TX1_X1 = np.broadcast_to(TX1_X1[None], (nodes["X"].state_dim, *TX1_X1.shape))
# %%
(product_state_oh(nodes["Z2"]) @ T21_X.reshape(30, 4)).argmax(-1)
# %%
individual_states((product_state_oh(nodes["X"]) @ square(TX12_X12)).argmax(-1), dfs_names(nodes["X"]))
# %%
(product_state_oh(nodes["Z2"]) @ T21_2.reshape(30, 6)).argmax(-1), dfs_names(nodes["Z2"])
# %%
individual_states((product_state_oh(nodes["Z2"]) @ square(T21_21)).argmax(-1), dfs_names(nodes["Z2"]))

# %%
# {'X': array([1, 3, 2, 3]), 'Z1': array([0, 0, 2, 3]), 'Z2': array([0, 3, 0, 5])}
