# %%
import numpy as np
from src.apps.gssm.hidden_markov_model import HMM
import os
import tqdm
import matplotlib.pyplot as plt

def get_config(alpha):
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

def make_data(hmm, bsz, seq_len):
    hmm._init_all_nodes(bsz)
    data = np.zeros((seq_len, bsz), int)

    for i in range(seq_len):
        data[i] = hmm.top_node.state
        hmm.evolve_classic(1)
    return data.T
    
def get_entropy_estimate_from_hmm(hmm : HMM, data, n_estimates=5):
    B, T = data.shape
    assert B//n_estimates == B/n_estimates
    bsz = B//n_estimates
    entropys = []
    stds = []
    for lo,hi in zip(range(0, B, bsz), range(bsz, B+bsz, bsz)):
      all = hmm.entropy_of_observations(data[lo:hi].T)
      entropys.append(all.mean())
      stds.append(all.std())
    return np.array(entropys), np.array(stds)

# %%
import tqdm
seq_len = 256
n_estimates = 5
bszs = [10, 20, 50, 100, 200, 500, 1000, 2000]

for alpha in [.01, .05, .1,]:
  mean_means = []
  mean_stds = []
  std_means = []
  std_stds = []
  alls = []
  for bsz in tqdm.tqdm(bszs):
    config = get_config(alpha)
    hmm = HMM(config, random_seed = 1)
    data = make_data(hmm, bsz=n_estimates*bsz, seq_len=seq_len)
    estimate_means, estimate_stds = get_entropy_estimate_from_hmm(hmm, data, n_estimates=n_estimates)
    mean_means += [estimate_means.mean()]
    mean_stds += [estimate_means.std()]
    std_means += [estimate_stds.mean()]
    std_stds += [estimate_stds.std()]
  
  def plot(means, stds, title_prefix):
    means = np.array(means)
    stds = np.array(stds)
    plt.fill_between(bszs, means - stds, means + stds, alpha=0.2)
    plt.plot(bszs, means)
    plt.xscale('log')
    plt.xlabel('# data')
    plt.ylabel('hmm estimate')
    plt.title(f'{title_prefix}, alpha={alpha}, T={seq_len}, n_estimates={n_estimates}')
    plt.show()
  plot(mean_means, mean_stds, "means")
  plot(std_means, std_stds, "stds")

# %%

seq_len = 256
n_estimates = 5
bszs = [10, 20, 50, 100, 200, 500, 1000, 2000]

for alpha in [.1]:
  config = get_config(alpha)
  alls = []
  for bsz in bszs:
    hmm = HMM(config, random_seed = 49*bsz+103)
    data = make_data(hmm, bsz=bsz, seq_len=seq_len)
    alls += [hmm.entropy_of_observations(data.T)]
  means = np.array([x.mean() for x in alls])
  stds = np.array([x.std() for x in alls])
  plt.fill_between(bszs, means - stds, means + stds, alpha=0.2)
  plt.plot(bszs, means)
  plt.xscale('log')
  plt.xlabel('# data')
  plt.ylabel('hmm estimate')
  plt.title(f'mean and std, alpha={alpha}, T={seq_len}, n_estimates={n_estimates}')
  plt.show()

# %%

seq_len = 256
n_estimates = 5
bszs = [2000]
alphas = [.005, .01, .05, .1, .5]

def get_entropy_estimate(hmms, datas):
  means = []
  stds = []
  for hmm, data in zip(hmms, datas):
    all = hmm.entropy_of_observations(data.T)
    means.append(all.mean())
    stds.append(all.std())
  return np.array(means), np.array(stds)


for bsz in bszs:
  mean_means = []
  mean_stds = []
  std_means = []
  std_stds = []
  alls = []
  for alpha in alphas:
    config = get_config(alpha)
    hmms = [HMM(config, random_seed = 19 * i + 429) for i in range(n_estimates)]
    datas = [make_data(hmms[i], bsz=bsz, seq_len=seq_len) for i in range(n_estimates)]
    estimate_means, estimate_stds = get_entropy_estimate(hmms, datas)
    mean_means += [estimate_means.mean()]
    mean_stds += [estimate_means.std()]
    std_means += [estimate_stds.mean()]
    std_stds += [estimate_stds.std()]
  
  def plot(means, stds, title_prefix):
    means = np.array(means)
    stds = np.array(stds)
    plt.fill_between(alphas, means - stds, means + stds, alpha=0.2)
    plt.plot(alphas, means)
    plt.xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('hmm estimate')
    plt.title(f'{title_prefix}, T={seq_len}, n_estimates={n_estimates}')
    plt.show()
  plot(mean_means, mean_stds, "means")
  plot(std_means, std_stds, "stds")

# %%
