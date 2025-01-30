# %%
from apps.gssm.hidden_markov_model import *
import matplotlib.pyplot as plt
import tqdm
from collections import defaultdict

# %%
plt.hist(np.random.dirichlet([6e-3]*128, 100000).max(axis=1), bins=500, density=True, alpha=.5, range=(0,1), label="128")
plt.hist(np.random.dirichlet([2.5e-2]*32, 100000).max(axis=1), bins=500, density=True, alpha=.5, range=(0,1), label="32")
plt.hist(np.random.dirichlet([1.55e-1]*6, 100000).max(axis=1), bins=500, density=True, alpha=.5, range=(0,1), label="6")
plt.hist(np.random.dirichlet([2.5e-1]*4, 100000).max(axis=1), bins=500, density=True, alpha=.5, range=(0,1), label="4")
plt.hist(np.random.dirichlet([7.9e-1]*2, 100000).max(axis=1), bins=500, density=True, alpha=.5, range=(0,1), label="2")
plt.legend()
plt.show()
# %%

def make_data(hmm: HMM, batch_size, seq_len):
    hmm._init_all_nodes(batch_size)
    observations = {n:np.zeros((seq_len, batch_size), dtype=int) for n,_ in hmm.topo_order}
    for i in range(seq_len):
      for name, node in hmm.topo_order:
        observations[name][i] = np.array(node.state)
      hmm.evolve_classic(1)
    return observations

def get_entropy(cfg, seed, n_data=1000, seq_len=256):
  hmm = HMM(cfg, random_seed=seed)
  test_data = make_data(hmm, n_data, seq_len)["X"].T
  hmm_estimate = hmm.entropy_of_observations(test_data.T, fast=True, small_mem=False)
  hmm_mean = (hmm_estimate / (seq_len - 1)).mean().item()
  return hmm_mean

# %%

def make_4node_config(alpha_z, alpha_x):
  cfg = {
      "nodes": [
          { "name": "Z1", "state_dim": 6, "parents": [], "alpha": alpha_z, "mode": "default", "observed": False},
          { "name": "Z2", "state_dim": 6, "parents": [], "alpha": alpha_z, "mode": "default", "observed": False},
          { "name": "Z3", "state_dim": 6, "parents": [], "alpha": alpha_z, "mode": "default", "observed": False},
          { "name": "Z4", "state_dim": 6, "parents": [], "alpha": alpha_z, "mode": "default", "observed": False},
          { "name": "X", "state_dim": 128, "parents": ["Z1","Z2","Z3","Z4"], "alpha": alpha_x, "mode": "default", "kernel_type": "fullrank", "observed": True, },
      ]
  }
  return cfg

def make_2node_config(alpha_z, alpha_x):
  cfg = {
      "nodes": [
          { "name": "Z1", "state_dim": 36, "parents": [], "alpha": alpha_z, "mode": "default", "observed": False},
          { "name": "Z2", "state_dim": 36, "parents": [], "alpha": alpha_z, "mode": "default", "observed": False},
          { "name": "X", "state_dim": 128, "parents": ["Z1","Z2"], "alpha": alpha_x, "mode": "default", "kernel_type": "fullrank", "observed": True, },
      ]
  }
  return cfg

def make_1node_config(alpha_z, alpha_x):
  cfg = {
      "nodes": [
          { "name": "Z1", "state_dim": 1296, "parents": [], "alpha": alpha_z, "mode": "default", "observed": False},
          { "name": "X", "state_dim": 128, "parents": ["Z1"], "alpha": alpha_x, "mode": "default", "kernel_type": "fullrank", "observed": True},
      ]
  }
  return cfg

#this is for experiment 12: compositionality
alpha_z_scale_1node = .02
alpha_z_scale_2node = .4
n_data = 10

ent1s = defaultdict(list)
ent2s = defaultdict(list)
ent4s = defaultdict(list)
for alpha_x in [1e-2, 1e-1]:
  for alpha_z in [1e-2, 2e-2, 5e-2]:
    for seed in tqdm.trange(10):
      cfg1 = make_1node_config(alpha_z*alpha_z_scale_1node, alpha_x)
      cfg2 = make_2node_config(alpha_z*alpha_z_scale_2node, alpha_x)
      cfg4 = make_4node_config(alpha_z, alpha_x)
      ent1s[(alpha_x,alpha_z)].append(get_entropy(cfg1, seed, n_data=n_data))
      ent2s[(alpha_x,alpha_z)].append(get_entropy(cfg2, seed, n_data=n_data))
      ent4s[(alpha_x,alpha_z)].append(get_entropy(cfg4, seed, n_data=n_data))

for key in ent2s.keys():
  ent2 = ent2s[key]
  ent4 = ent4s[key]
  plt.scatter(ent2, ent4, label=key)
plt.plot([0,3.5],[0,3.5], color="black")
plt.xlabel("2 nodes")
plt.ylabel("4 nodes")
plt.legend()
plt.show()

for key in ent2s.keys():
  ent2 = ent2s[key]
  ent1 = ent1s[key]
  plt.scatter(ent2, ent1, label=key)
plt.plot([0,3.5],[0,3.5], color="black")
plt.xlabel("2 nodes")
plt.ylabel("1 nodes")
plt.legend()
plt.show()
# %%

#experiment 13: computational hardness
def get_cfg(state_dim, alpha):
    cfg = {
        "nodes": [
            { "name": "Z1", "state_dim": state_dim, "parents": [], "alpha": alpha, "mode": "default", "observed": False},
            { "name": "X", "state_dim": 128, "parents": ["Z1"], "alpha": 2e-3, "mode": "default", "kernel_type": "fullrank", "observed": True},
        ]
    }
    return cfg

n_data = 10

entropyss = defaultdict(list)
for alpha_z in [5e-3, 1e-2, 2e-2]:
  for sd in [2048, 1024, 512, 256]:
    for seed in tqdm.trange(10):
      alpha_z_ = alpha_z / sd * 256
      cfg = get_cfg(sd, alpha_z_)
      entropyss[(sd, alpha_z_)].append(get_entropy(cfg, seed, n_data=n_data))
# %%

for (sd, alpha_z), entropys in entropyss.items():
    plt.hist(entropys, bins=50, range=(1.5, np.log(32)), label=f"{sd}, {alpha_z}", color=f"C{sd//256}", alpha=.5)
plt.legend()

# %%

# experiment 14: conditional independences

def get_cfg_sparse(state_dim, alpha):
    cfg = {
        "nodes": [
            { "name": "Z1", "state_dim": state_dim, "parents": [], "alpha": alpha, "mode": "default", "observed": False},
            { "name": "Z2", "state_dim": state_dim, "parents": [], "alpha": alpha, "mode": "default", "observed": False},
            { "name": "Z3", "state_dim": state_dim, "parents": ["Z1"], "alpha": alpha, "mode": "default", "kernel_type": "fullrank", "observed": False},
            { "name": "Z4", "state_dim": state_dim, "parents": ["Z2"], "alpha": alpha, "mode": "default", "kernel_type": "fullrank", "observed": False},
            { "name": "X", "state_dim": 128, "parents": ["Z3", "Z4"], "alpha": 6e-3, "mode": "default", "kernel_type": "fullrank", "observed": True},
        ]
    }
    return cfg

def get_cfg_dense(state_dim, alpha):
    cfg = {
        "nodes": [
            { "name": "Z1", "state_dim": state_dim, "parents": [], "alpha": alpha, "mode": "default", "observed": False},
            { "name": "Z2", "state_dim": state_dim, "parents": [], "alpha": alpha, "mode": "default", "observed": False},
            { "name": "Z3", "state_dim": state_dim, "parents": ["Z1", "Z2"], "alpha": alpha, "mode": "default", "kernel_type": "fullrank", "observed": False},
            { "name": "Z4", "state_dim": state_dim, "parents": ["Z1", "Z2"], "alpha": alpha, "mode": "default", "kernel_type": "fullrank", "observed": False},
            { "name": "X", "state_dim": 128, "parents": ["Z3", "Z4"], "alpha": 6e-3, "mode": "default", "kernel_type": "fullrank", "observed": True},
        ]
    }
    return cfg

n_data = 20

entropyss = defaultdict(list)
for alpha_z in [5e-2, 1e-1, 2e-1]:
  for seed in tqdm.trange(30):
    cfg = get_cfg_sparse(6, alpha_z)
    entropyss[(alpha_z, "sparse")].append(get_entropy(cfg, seed, n_data=n_data))
    cfg = get_cfg_dense(6, alpha_z)
    entropyss[(alpha_z, "dense")].append(get_entropy(cfg, seed, n_data=n_data))
    entropyss

# %%
for (alpha_z, mode), entropys in entropyss.items():
    plt.hist(entropys, bins=50, range=(.5, np.log(128)), label=f"{mode}, {alpha_z}", alpha=.5)#, color=f"C{int(alpha_z/5e-2)}")
plt.legend()

# %%
# experiment 15: 

def get_cfg_1controller(state_dim_control, alpha_control, state_dim, alpha, alpha_X=6e-3):
    cfg = {
        "nodes": [
            { "name": "Z12", "state_dim": state_dim_control, "parents": [], "alpha": alpha_control, "mode": "slow", "observed": False},
            { "name": "Z3", "state_dim": state_dim, "parents": ["Z12"], "alpha": alpha, "mode": "default", "kernel_type": "product", "observed": False},
            { "name": "Z4", "state_dim": state_dim, "parents": ["Z12"], "alpha": alpha, "mode": "default", "kernel_type": "product", "observed": False},
            { "name": "X", "state_dim": 128, "parents": ["Z3", "Z4"], "alpha": alpha_X, "mode": "default", "kernel_type": "product", "observed": True},
        ]
    }
    return cfg

def get_cfg_2controller(state_dim_control, alpha_control, state_dim, alpha, alpha_X=6e-3):
    cfg = {
        "nodes": [
            { "name": "Z1", "state_dim": state_dim_control, "parents": [], "alpha": alpha_control, "mode": "slow", "observed": False},
            { "name": "Z2", "state_dim": state_dim_control, "parents": [], "alpha": alpha_control, "mode": "slow", "observed": False},
            { "name": "Z3", "state_dim": state_dim, "parents": ["Z1", "Z2"], "alpha": alpha, "mode": "default", "kernel_type": "product", "observed": False},
            { "name": "Z4", "state_dim": state_dim, "parents": ["Z1", "Z2"], "alpha": alpha, "mode": "default", "kernel_type": "product", "observed": False},
            { "name": "X", "state_dim": 128, "parents": ["Z3", "Z4"], "alpha": alpha_X, "mode": "default", "kernel_type": "product", "observed": True},
        ]
    }
    return cfg

# %%
n_data = 20
seq_len = 100

entropyss = defaultdict(list)
for alpha_z in [5e-2, 1e-1, 2e-1]:
  for seed in tqdm.trange(5):
    cfg = get_cfg_1controller(6, 1.55e-1, 6, alpha_z)
    entropyss[(alpha_z, "1control")].append(get_entropy(cfg, seed, n_data=n_data, seq_len=seq_len))
    cfg = get_cfg_2controller(6, 1.55e-1, 6, alpha_z)
    entropyss[(alpha_z, "2control")].append(get_entropy(cfg, seed, n_data=n_data, seq_len=seq_len))
    entropyss

# %%
for (alpha_z, mode), entropys in entropyss.items():
    plt.hist(entropys, bins=50, range=(.5, np.log(128)), label=f"{mode}, {alpha_z}", alpha=.5)#, color=f"C{int(alpha_z/5e-2)}")
plt.axvline(np.log(128))
plt.legend()
# %%
# %%
# experiment 16: slow and dead nodes

def get_cfg(state_dim, alpha, alpha_X=6e-3, mode="default"):
    cfg = {
        "nodes": [
            { "name": "Z1", "state_dim": state_dim, "parents": [], "alpha": alpha, "mode": mode, "observed": False},
            { "name": "Z2", "state_dim": state_dim, "parents": [], "alpha": alpha, "mode": mode, "observed": False},
            { "name": "X", "state_dim": 128, "parents": ["Z1", "Z2"], "alpha": alpha_X, "mode": "default", "kernel_type": "fullrank", "observed": True},
        ]
    }
    return cfg

# %%
n_data = 20
seq_len = 64
state_dim = 32

# entropyss = defaultdict(list)
# for alpha_z in [2e-2, 5e-2, 8e-2]:
#   for seed in tqdm.trange(10):
#     for mode, mult in zip(["slow", "dead", "default"], (1,1,.65)):
#       cfg = get_cfg(state_dim, alpha_z*mult, mode=mode)
#       entropyss[mode].append(get_entropy(cfg, seed, n_data=n_data, seq_len=seq_len))

entropyss = defaultdict(list)
for alpha_z in [2e-2, 3e-2, 5e-2, 6e-2, 7e-2, 8e-2]:
  for seed in tqdm.trange(5):
    for mode, mult in zip(["slow", "dead", "default"], (1,.95,.65)):
      cfg = get_cfg(state_dim, alpha_z*mult, alpha_X=6e-3, mode=mode)
      entropyss[mode].append(get_entropy(cfg, seed, n_data=n_data, seq_len=seq_len))

max_ = np.log(128)
plt.scatter(entropyss["default"], entropyss["slow"], label="slow vs default")
plt.scatter(entropyss["default"], entropyss["dead"], label="dead vs default")
plt.plot([0,max_], [0, max_])
plt.legend()
plt.show()
# %%
for mode, entropys in entropyss.items():
    plt.hist(entropys, bins=50, range=(.5, max_), label=f"{mode}", alpha=.5)#, color=f"C{int(alpha_z/5e-2)}")
plt.axvline(max_)
plt.legend()
plt.show()
# %%
