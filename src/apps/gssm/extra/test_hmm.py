# %%
from apps.gssm.hidden_markov_model import *

gssm_config = {
    "nodes": [
        {
            "name": "Z1",
            "state_dim": 4,
            "parents": [],
            "alpha": .1,
            "mode": "slow",
            "observed": False,
        },
        {
            "name": "Z2",
            "state_dim": 4,
            "parents": ["Z1"],
            "alpha": 1,
            "mode": "default",
            "kernel_type": "fullrank",
            "observed": False,
        },
        {
            "name": "Z3",
            "state_dim": 4,
            "parents": ["Z1","Z2"],
            "alpha": 1,
            "mode": "default",
            "observed": False,
        },
        {
            "name": "Z4",
            "state_dim": 4,
            "parents": ["Z1","Z2"],
            "alpha": 1,
            "mode": "default",
            "observed": False,
        },
        {
            "name": "X",
            "state_dim": 32,
            "parents": ["Z1","Z3","Z4"],
            "alpha": .1,
            "mode": "default",
            "kernel_type": "fullrank",
            "observed": True,
        },
    ]
}

gssm_config_ICL = {
    "nodes": [
        {
            "name": "Z1",
            "state_dim": 4,
            "parents": [],
            "alpha": .1,
            "mode": "default",
            "observed": False,
        },
        {
            "name": "Z2",
            "state_dim": 4,
            "parents": ["Z1"],
            "alpha": .1,
            "mode": "default",
            "kernel_type": "fullrank",
            "observed": False,
        },
        {
            "name": "Z3",
            "state_dim": 4,
            "parents": ["Z2"],
            "alpha": 1,
            "mode": "context",
            "observed": False,
        },
        {
            "name": "X",
            "state_dim": 8,
            "parents": ["Z1", "Z3"],
            "alpha": .05,
            "mode": "context",
            "kernel_type": "fullrank",
            "observed": True,
        },
    ]
}

small_gssm_config = {
    "nodes": [
        {
            "name": "Z1",
            "state_dim": 4,
            "parents": [],
            "alpha": .1,
            "mode": "slow",
            "observed": False,
        },
        {
            "name": "Z2",
            "state_dim": 4,
            "parents": ["Z1"],
            "alpha": .1,
            "mode": "default",
            "observed": False,
        },
        {
            "name": "X",
            "state_dim": 32,
            "parents": ["Z1","Z2"],
            "alpha": .1,
            "mode": "default",
            "kernel_type": "fullrank",
            "observed": True,
        },
    ]
}

# %%
def make_data(hmm: HMM, batch_size, seq_len):
    hmm._init_all_nodes(batch_size)
    observations = np.zeros((seq_len, batch_size), dtype=int)
    for i in range(seq_len):
        observations[i] = np.array(hmm.top_node.state)
        hmm.evolve_classic(1)
    return observations

def make_data2(hmm: HMM, batch_size, seq_len):
    hmm._init_all_nodes(batch_size)
    observations = np.zeros((seq_len, batch_size), dtype=int)
    prod_trans = hmm.make_prod_transition()
    for i in range(seq_len):
        observations[i] = np.array(hmm.top_node.state)
        data = hmm.fwd_via_matmul(prod_trans)
        for (st, (_,node)) in zip(data, hmm.topo_order):
          node.state = st
    return observations

def test_entropys():
  hmm = HMM(gssm_config, random_seed=np.random.randint(29042))
  data1 = make_data(hmm, batch_size=20, seq_len=20)
  entropys1 = hmm.entropy_of_observations(data1, small_mem=False, fast=False) # the old but gold
  entropys2 = hmm.entropy_of_observations(data1, small_mem=True, fast=False)
  entropys3 = hmm.entropy_of_observations(data1, small_mem=False, fast=True)
  entropys4 = hmm.entropy_of_observations(data1, small_mem=True, fast=True)
  print(((entropys1 - entropys2)).abs().mean())
  print(((entropys1 - entropys3)).abs().mean())
  print(((entropys1 - entropys4)).abs().mean())

def test_generation(bsz=100):
  seed = np.random.randint(29042)
  hmm = HMM(small_gssm_config, random_seed=seed)
  data1 = make_data(hmm, bsz, 50)
  data2 = make_data2(hmm, bsz, 50)
  entropys1 = hmm.entropy_of_observations(data1)
  entropys2 = hmm.entropy_of_observations(data2)
  hmm_mean1 = (entropys2 / (seq_len - 1)).mean().item()
  hmm_mean2 = (entropys2 / (seq_len - 1)).mean().item()
  print("entropy means", hmm_mean1, hmm_mean2)
  import matplotlib.pyplot as plt
  for i in range(5, 10):
    plt.hist(data1[i], alpha=.5, range=(0, hmm.top_node.state_dim), label="true evolve")
    plt.hist(data2[i], alpha=.5, range=(0, hmm.top_node.state_dim), label="hmm evolve")
    plt.legend()
    plt.show()

test_entropys()
test_generation()

# %%
emb_dim = 128
nb_heads = 4
seq_len = 50
nb_layers = 4
n_train = 40000
n_test = 1000

hmm = HMM(gssm_config, random_seed=2489)
train_data = make_data(hmm, n_train, seq_len).T
test_data = make_data(hmm, n_test, seq_len).T
hmm_estimate = hmm.entropy_of_observations(test_data.T, fast=True, small_mem=False)
hmm_mean = (hmm_estimate / (seq_len - 1)).mean().item()
train_data.shape, test_data.shape, hmm_estimate.shape
hmm_mean
# %%
from src.nanollama.model.transfomer import Transformer, TransformerConfig, TransformerBlockConfig
block_cfg = TransformerBlockConfig(seq_len=seq_len, emb_dim=emb_dim, nb_heads=nb_heads)
t_cfg = TransformerConfig(block_cfg, vocab_size=hmm.top_node.state_dim, emb_dim=emb_dim, nb_layers=nb_layers)
model = Transformer(t_cfg)

import tqdm
def loss_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = preds.size(-1)
    return torch.nn.functional.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")

device = "cuda"

batch_size = 512
epochs = 500

train_data = torch.as_tensor(train_data).to(device)
test_data = torch.as_tensor(test_data).to(device)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

# training loop
for epoch in (pbar:=tqdm.trange(epochs)):
    for batch_i in range(0, len(train_data), batch_size):
        batch = train_data[batch_i : batch_i + batch_size]
        # forward pass
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_func(logits[:,:-1], batch[:,1:]).mean()
        # backward pass
        loss.backward()
        optimizer.step()
    # eval
    with torch.inference_mode():
      loss_test = loss_func(model(test_data)[:,:-1], test_data[:,1:])
      # update progress bar
      pbar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.3f}/{loss_test.mean().item():.4f}, hmm: {hmm_mean:.4f}")

# %%
# they are not duplicates
test = [tuple(arr.tolist()) for arr in test_data]
train = [tuple(arr.tolist()) for arr in train_data]

# Check if there is any intersection
len(set(train).intersection(set(test)))
# %%

    # def test_forward_probs(config):
    #     hmm = HMM(config)
    #     batch_size = 2
    #     seq_len = 2
    #     hmm._init_all_nodes(batch_size)
    #     observations = np.zeros((seq_len, batch_size), dtype=int)
    #     for i in range(seq_len):
    #         observations[i] = np.array(hmm.top_node.state)
    #         hmm.evolve_classic(1)
    #     #sanity check
    #     print(hmm.forward_probs(observations)[0].exp().sum(0))

    # test_forward_probs(gssm_config)

    # def test_entropy(config, seq_len, batch_size, seed=3892493):
    #     observations = np.zeros((seq_len, batch_size), dtype=int)
    #     n_seeds = 10
    #     entropys = []
    #     for i_batch in range(n_seeds):
    #       mini_batch = batch_size//n_seeds
    #       batch_slice = slice(i_batch*mini_batch, (i_batch+1)*mini_batch)
    #       hmm = HMM(config, random_seed=seed + i_batch*1942)
    #       hmm._init_all_nodes(mini_batch)
    #       for i in range(seq_len):
    #           observations[i,batch_slice] = np.array(hmm.top_node.state)
    #           hmm.evolve_classic(1)
    #       entropys.append(hmm.entropy_of_observations(observations[:,batch_slice]).mean().item())
    #     return np.mean(entropys) / seq_len, np.median(entropys) / seq_len
    
    # for seq_len in np.logspace(0,np.log10(100),10):
    #   seq_len = int(seq_len)
    #   print(seq_len, test_entropy(gssm_config, seq_len, 200))

# %%
def test_entropy(config, seq_len, batch_size, seed=3892493):
    observations = np.zeros((seq_len, batch_size), dtype=int)
    n_seeds = 1
    entropys = []
    for i_batch in range(n_seeds):
        mini_batch = batch_size//n_seeds
        batch_slice = slice(i_batch*mini_batch, (i_batch+1)*mini_batch)
        hmm = HMM(config, random_seed=seed + i_batch*1942)
        hmm._init_all_nodes(mini_batch)
        for i in range(seq_len):
            observations[i,batch_slice] = np.array(hmm.top_node.state)
            hmm.evolve_classic(1)
        entropys.append(hmm.entropy_of_observations(observations[:,batch_slice]).mean().item())
    return np.mean(entropys) / seq_len, np.median(entropys) / seq_len

for seq_len in [4]:# np.logspace(0,np.log10(100),10):
    seq_len = int(seq_len)
    print(seq_len, test_entropy(gssm_config_ICL, seq_len, 2))

    
# for seq_len in np.logspace(0,np.log10(100),10):
#     seq_len = int(seq_len)
#     print(seq_len, test_entropy(gssm_config, seq_len, 200))



###########################

# %%
import numpy as np
from src.apps.gssm.hidden_markov_model import HMM
import os
import tqdm
import matplotlib.pyplot as plt
import zlib

def make_data(hmm, bsz, seq_len):
    hmm._init_all_nodes(bsz)
    data = np.zeros((seq_len, bsz), int)

    for i in range(seq_len):
        data[i] = hmm.top_node.state
        hmm.evolve_classic(1)
    return data.T
    
def get_entropy_estimate_from_hmm(hmm : HMM, data, n_estimates=5, verbose = False):
    B, T = data.shape
    assert B//n_estimates == B/n_estimates
    bsz = B//n_estimates
    entropys = []
    stds = []
    for lo,hi in zip(range(0, B, bsz), range(bsz, B+bsz, bsz)):
      all = hmm.entropy_of_observations(data[lo:hi].T, verbose = verbose)
      entropys.append(all.mean())
      stds.append(all.std())
    return np.array(entropys), np.array(stds)



# %%
## Trivial case of random sequences
bszs = [100, 1000, 2000] 
seq_len = 256
n_estimates = 5
alpha = 10000

k = 16

random_config = {"nodes": [
          {
              "name": "Z1",
              "state_dim": k,
              "parents": [],
              "alpha": alpha,
              "mode": "default",
              "observed": False,
          },
          {
              "name": "X",
              "state_dim": k,
              "parents": ["Z1"],
              "alpha": alpha,
              "mode": "default",
              "observed": True,
          },
      ]}

mean_means = []
mean_stds = []
for bsz in tqdm.tqdm(bszs):
  config = random_config
  data = np.random.randint(0, k, size=(bsz*n_estimates, seq_len-1))
  data = np.concatenate((np.zeros((bsz*n_estimates,1), dtype = int), data), axis = 1)
  hmm = HMM(config, random_seed = 1)
  estimate_means, estimate_stds = get_entropy_estimate_from_hmm(hmm, data, n_estimates=n_estimates)
  mean_means += [estimate_means.mean()]
  mean_stds += [estimate_means.std()]
  

def plot(means,  stds, title_prefix):
  means = np.array(means)
  stds = np.array(stds)
  plt.fill_between(bszs, means - stds, means + stds, alpha=0.2)
  plt.plot(bszs, means, 'g', label = "hmm")
  plt.xscale('log')
  plt.xlabel('# data')
  plt.ylabel('hmm estimate')
  plt.title(f'{title_prefix}, experiment = random, alpha={alpha}, T={seq_len}, n_estimates={n_estimates}')
  plt.show()
print("Results")
print(f"Expected value {np.log(k)*seq_len}")
print(f"HMM estimates {mean_means}")
plot(mean_means, mean_stds, "Entropy")

# %%
## Plot entropy estimates with stds for various batch sizes and various experiments, compare with gzip, record time



def get_various_configs(alpha, difficulty):
  if difficulty == "easy":
    return {
        "nodes": [
            {
                "name": "Z1",
                "state_dim": 4,
                "parents": [],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "X",
                "state_dim": 4,
                "parents": ["Z1"],
                "alpha": alpha,
                "mode": "default",
                "observed": True,
            },
        ]
    }
  elif difficulty == "medium":
    return {
        "nodes": [
            {
                "name": "Z1",
                "state_dim": 4,
                "parents": [],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "Z2",
                "state_dim": 4,
                "parents": ["Z1"],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "X",
                "state_dim": 16,
                "parents": ["Z1", "Z2"],
                "alpha": alpha,
                "mode": "default",
                "observed": True,
            },
        ]
    }
  elif difficulty == "medium_2":
    return {
        "nodes": [
            {
                "name": "Z1",
                "state_dim": 16,
                "parents": [],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "X",
                "state_dim": 16,
                "parents": ["Z1"],
                "alpha": alpha,
                "mode": "default",
                "observed": True,
            },
        ]
    }
  elif difficulty == "hard":
    return {
        "nodes": [
            {
                "name": "Z1",
                "state_dim": 4,
                "parents": [],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "Z2",
                "state_dim": 4,
                "parents": ["Z1"],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "Z3",
                "state_dim": 4,
                "parents": ["Z1", "Z2"],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "Z4",
                "state_dim": 4,
                "parents": ["Z1", "Z2", "Z3"],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "X",
                "state_dim": 256,
                "parents": ["Z1", "Z2", "Z3", "Z4"],
                "alpha": alpha,
                "mode": "default",
                "observed": True,
            },
        ]
    }
  elif difficulty == "hard2":
    return {
        "nodes": [
            {
                "name": "Z1",
                "state_dim": 256,
                "parents": [],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "X",
                "state_dim": 256,
                "parents": ["Z1"],
                "alpha": alpha,
                "mode": "default",
                "observed": True,
            },
        ]
    }
  elif difficulty == "ICL_easy":
    return {
        "nodes": [
            {
                "name": "Z1",
                "state_dim": 2,
                "parents": [],
                "alpha": alpha,
                "mode": "context",
                "observed": False,
            },
            {
                "name": "Z2",
                "state_dim": 2,
                "parents": [],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "X",
                "state_dim": 4,
                "parents": ["Z1", "Z2"],
                "alpha": alpha,
                "mode": "default",
                "observed": True,
            },
        ]
    }
  elif difficulty == "ICL_medium":
    return {
        "nodes": [
            {
                "name": "Z1",
                "state_dim": 4,
                "parents": [],
                "alpha": alpha,
                "mode": "context",
                "observed": False,
            },
            {
                "name": "Z2",
                "state_dim": 4,
                "parents": [],
                "alpha": alpha,
                "mode": "context",
                "observed": False,
            },
            {
                "name": "Z1",
                "state_dim": 4,
                "parents": [],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "Z2",
                "state_dim": 4,
                "parents": [],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "X",
                "state_dim": 64,
                "parents": ["Z1", "Z2", "Z3", "Z4"],
                "alpha": alpha,
                "mode": "default",
                "observed": True,
            },
        ]
    }
  

# # %%
# def get_entropy_estimate_from_gzip(data):
#   entropy = len(zlib.compress(data.tobytes(), level=9)) / data.size
#   return entropy

# def get_reverse_entropy_estimate_from_gzip(data):
#   entropy = len(zlib.compress(data.T.tobytes(), level=9)) / data.size
#   return entropy


# # %%
# config = get_various_configs(0.1, "easy")
# hmm = HMM(config, random_seed = 1)
# data = make_data(hmm, bsz=4, seq_len=1000)
# estimate_means, _ = get_entropy_estimate_from_hmm(hmm, data, n_estimates=4)
# print(f"estimate_means {estimate_means}")
# gzip_estimate = get_entropy_estimate_from_gzip(data)
# print(f"gzip_estimate {gzip_estimate}")
# gzip_reverse_estimate = get_reverse_entropy_estimate_from_gzip(data)
# print(f"gzip_reverse_estimate {gzip_reverse_estimate}")



# %%

# TODO: get it to run on cluster
# TODO: check if HMMs are correctly generated

import tqdm
seq_len = 100
n_estimates = 2
bszs = [2, 20] # [100, 1000, 2500, 5000]
difficulties = ["ICL_easy"]# ["easy", "medium"]#, "hard", "ICL_easy", "ICL_medium"]




for difficulty in difficulties:
  for alpha in [10000] : # [.001, .1, 1]:
    mean_means = []
    mean_stds = []
    gzip_estimates = [] 
    gzip_reverse_estimates = []
    for bsz in tqdm.tqdm(bszs):
      config = get_various_configs(alpha, difficulty)
      hmm = HMM(config, random_seed = 1)
      data = make_data(hmm, bsz=n_estimates*bsz, seq_len=seq_len)
      estimate_means, estimate_stds = get_entropy_estimate_from_hmm(hmm, data, n_estimates=n_estimates, verbose = True)
      mean_means += [estimate_means.mean()]
      mean_stds += [estimate_means.std()]
     
  def plot(means, stds, title_prefix):
    means = np.array(means)
    stds = np.array(stds)
    plt.fill_between(bszs, means - stds, means + stds, alpha=0.2)
    plt.plot(bszs, means, 'g', label = "hmm")
    plt.xscale('log')
    plt.xlabel('# data')
    plt.ylabel('hmm estimate')
    plt.title(f'{title_prefix}, experiment = {difficulty}, alpha={alpha}, T={seq_len}, n_estimates={n_estimates}')
    plt.show()
  plot(mean_means, mean_stds, "Entropy")
# %%

print(np.log(4)*seq_len)
# %%
