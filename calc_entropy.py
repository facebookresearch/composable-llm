# %%
from src.nanollama.data import gssm
import numpy as np
from omegaconf import OmegaConf

# %%
gssm_config = OmegaConf.create(
    {
        "nodes": [
            {
                "name": "Z1",
                "state_dim": 5,
                "parents": [],
                "alpha":1e-8,
                "mode": "default",
            },
            {
                "name": "Z2",
                "state_dim": 5,
                "parents": ["Z1"],
                "alpha":1e-8,
                "mode": "default",
            },
            {
                "name": "Z3",
                "state_dim": 6,
                "parents": ["Z2"],
                "alpha":1e-8,
                "mode": "default",
            },
            {
                "name": "X",
                "state_dim": 8,
                "parents": ["Z1", "Z2", "Z3"],
                "alpha":1e-8,
                "mode": "default",
            },
        ]
    }
)

nodes = gssm.build_gssm(gssm_config, np.random.default_rng(100))

# %%
names = {}
for name, node in nodes.items():
  names[node] = name

def dfs_names(node):
    def _dfs_names(node, fc=True):
        if not fc and isinstance(node, gssm.ObservedNode):
            return [names[node]]
        return [d for p in node.parents for d in _dfs_names(p, False)] + [names[node]]

    # unique
    ns = _dfs_names(node)
    return list(dict.fromkeys(ns))

indexs = {}
transitions = {}
for i, name in enumerate(dfs_names(nodes["X"])):
    node = nodes[name]
    parents = node.parents
    parent_state_dims = tuple([p.state_dim for p in parents])
    observed = name == "X"
    trans = node.kernel.p_transition
    target_shape = tuple() if observed else (node.state_dim,)
    target_shape += parent_state_dims + (node.state_dim,)
    transitions[node] = trans.reshape(target_shape)
    indexs[node] = i
    print(name, target_shape, node.kernel.p_transition.shape)

# %%
SYM_IN = "abcxdefghijklmnopqrstuvwxyz"
SYM_OUT = "ABCXDEFGHIJKLMNOPQRSTUVWXYZ"

# %%

def node_sym_in(node):
    return SYM_IN[indexs[node]]


def node_sym_out(node):
    return SYM_OUT[indexs[node]]

# %%
def init(node, bsz, i=0):
    node.time = 0
    for parent in node.parents:
        if parent.time != 0 and not isinstance(parent, gssm.ObservedNode):
            init(parent, bsz, i+1)
    node.state = (np.arange(bsz, dtype=int) + i)[::-1]


init(nodes["X"], 4)
{name: node.state for name,node in nodes.items()}

# %%
def one_hot_state(node):
    targets = node.state
    nb_classes = node.state_dim
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def product_state_oh(tgt_node):
  node_order = dfs_names(tgt_node)
  einsum_str = "B" + ",B".join(SYM_IN[:len(node_order)]) + "->B" + SYM_IN[:len(node_order)]
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

def test():
  state = product_state(nodes["X"])
  print(state)
  individual_states(state, dfs_names(nodes["X"]))

# %%
def square(m):
   dim = int(np.prod(m.shape)**.5)
   return m.reshape(dim,dim)

# %%
def einsum_input_str(tgt_node):
    """
    Constructs the einsum string for the target node transition matrix when used as input
    """
    observed = names[tgt_node] == "X"
    einsum_str = node_sym_in(tgt_node) if not observed else ""
    einsum_str += "".join(node_sym_out(p) for p in tgt_node.parents)
    einsum_str += node_sym_out(tgt_node)
    return einsum_str

def einsum_prod_str(tgt_node):
    """
    Constructs the einsum string for the target node transition matrix in the product state form
    """
    ordered_nodes = [nodes[name] for name in dfs_names(tgt_node)]
    in_str = "".join(node_sym_in(n) for n in ordered_nodes)
    out_str = "".join(node_sym_out(n) for n in ordered_nodes)
    return in_str + out_str

# %%
def manual():
  Ta_A = transitions[nodes["Z1"]]
  TbA_B = transitions[nodes["Z2"]]
  TcB_C = transitions[nodes["Z3"]]
  TABC_X = transitions[nodes["X"]]

  one_A = np.ones(nodes["Z1"].state_dim)
  one_B = np.ones(nodes["Z2"].state_dim)
  one_C = np.ones(nodes["Z3"].state_dim)
  one_X = np.ones(nodes["X"].state_dim)

  Tabcx_1 = np.einsum("a,b,c,x->abcx", one_A,one_B,one_C,one_X)
  Tabcx_A = np.einsum("aA,abcx->abcxA",Ta_A,Tabcx_1)
  Tabcx_B = np.einsum("bAB,abcxA->abcxB",TbA_B,Tabcx_A)
  Tabcx_C = np.einsum("cBC,abcxB->abcxC",TcB_C,Tabcx_B)
  Tabcx_X = np.einsum("ABCX,abcxA,abcxB,abcxC->abcxX",TABC_X,Tabcx_A,Tabcx_B,Tabcx_C)
  # output product_state
  Tabcx_ABCX = np.einsum("abcxX,abcxA,abcxB,abcxC->abcxABCX",Tabcx_X,Tabcx_A,Tabcx_B,Tabcx_C)
  def flatten_input_dim(m):
    return m.reshape(-1, m.shape[-1]) 
  print(individual_states((product_state_oh(nodes["X"]) @ flatten_input_dim(Tabcx_A)).argmax(-1), dfs_names(nodes["X"])))
  print(individual_states((product_state_oh(nodes["X"]) @ flatten_input_dim(Tabcx_B)).argmax(-1), dfs_names(nodes["X"])))
  print(individual_states((product_state_oh(nodes["X"]) @ flatten_input_dim(Tabcx_C)).argmax(-1), dfs_names(nodes["X"])))
  print(individual_states((product_state_oh(nodes["X"]) @ square(Tabcx_ABCX)).argmax(-1), dfs_names(nodes["X"])))
  return Tabcx_ABCX

reference_prod_transition = manual()

# %%
def make_prod_transition(tgt_node):
  # x,aA,b,c->xabcA
  node_order = dfs_names(tgt_node)
  state_dims = [nodes[name].state_dim for name in node_order]
  prod_input_str = SYM_IN[:len(node_order)]
  def prod_str(node):
     return f"{prod_input_str}{node_sym_out(node)}"

  einsum_input_strs = {nodes[name] : einsum_input_str(nodes[name]) for name in node_order}
  einsum_prod_strs = {nodes[name] : prod_str(nodes[name]) for name in node_order}
  einsum_prod_strs["root"] = prod_input_str
  print(einsum_input_strs)
  print(einsum_prod_strs)
  # a collection of einsum_str -> array to use for the actual einsum
  # we build the logic only with the strings and fetch the matrices from here
  einsum_str_to_arr = {prod_input_str : np.ones(state_dims)} # this is the root (for convenience)
  einsum_str_to_arr |= {
     s : transitions[node] for node,s in einsum_input_strs.items()
  }
  print({name : x.shape for name,x in einsum_str_to_arr.items()})
  # step 1 : fill the abcxX matrices
  for name in node_order:
    node = nodes[name]
    parents = node.parents or ["root"]
    # this is good for Tabcx_A
    input_strs = [einsum_input_strs[node], *[einsum_prod_strs[p] for p in parents]]
    output_str = einsum_prod_strs[node]
    einsum_str = ",".join(input_strs) + "->" + output_str
    input_tensors = [einsum_str_to_arr[s] for s in input_strs]
    einsum_str_to_arr[output_str]=np.einsum(einsum_str, *input_tensors)
  
  print({name : x.shape for name,x in einsum_str_to_arr.items()})
  # step 2 : get final product transition
  input_strs = [einsum_prod_strs[nodes[name]] for name in node_order]
  input_tensors = [einsum_str_to_arr[s] for s in input_strs]
  output_str = einsum_prod_str(tgt_node)
  einsum_str = ",".join(input_strs) + "->" + output_str

  return np.einsum(einsum_str, *input_tensors)
    

prod_transition = make_prod_transition(nodes["X"])
# %%
assert (prod_transition == reference_prod_transition).all()
individual_states((product_state_oh(nodes["X"]) @ square(prod_transition)).argmax(-1), dfs_names(nodes["X"]))


# %%
def fwd_product_state_example(prod_transition):
  prod_dim=int(np.prod(prod_transition.shape)**.5)
  kernel = gssm.TransitionKernel(prod_dim, prod_dim, 1)
  kernel.p_transition = square(prod_transition)
  kernel._cumulative = np.cumsum(kernel.p_transition, axis=1)
  next_state = kernel(product_state(nodes["X"]))
  print(individual_states(next_state, dfs_names(nodes["X"])))

fwd_product_state_example(prod_transition)

# %%
# this is the ground truth (one actual evolve of the nodes)
# nodes have to be reinitialized after this
for i in range(1):
    nodes["X"].evolve()
    print({name: node.state for name,node in nodes.items()})
# %%
