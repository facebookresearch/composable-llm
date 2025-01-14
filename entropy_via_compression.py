# %%
import numpy as np
from hidden_markov_model import HMM
import os
import tqdm

# %%
gssm_config = {
    "nodes": [
        {
            "name": "Z1",
            "state_dim": 5,
            "parents": [],
            "alpha": .1,
            "mode": "default",
        },
        {
            "name": "Z2",
            "state_dim": 6,
            "parents": ["Z1"],
            "alpha": .1,
            "mode": "default",
        },
        {
            "name": "Z3",
            "state_dim": 7,
            "parents": ["Z2"],
            "alpha": .1,
            "mode": "default",
        },
        {
            "name": "X",
            "state_dim": 8,
            "parents": ["Z1", "Z3"],
            "alpha": .1,
            "mode": "default",
        },
    ]
}

# %%
def make_data(config, bsz, seq_len):
  hmm = HMM(config)
  hmm._init_all_nodes(bsz)
  data = np.zeros((seq_len, bsz), int)
  data[0] = hmm.top_node.state
  for i in range(1, seq_len):
    data[i] = hmm.evolve_classic(1)["X"]
  return " ".join(map(str, data.T.flatten()))

# %% 
data = []
n_total = 20000
step_size = 100
seq_len = 100
for n in range(n_total // step_size):
  data.append(make_data(gssm_config, bsz=step_size, seq_len=seq_len))
# %%
compressed_sizes = {}
for n in tqdm.trange(1,len(data)+1):
  total_len = len(data[0])*n
  with open("data_classic.txt", "w") as f:
    f.write(" ".join(data[:n]))
  # TODO adjust here
  os.system("../bzz/bzz -e1G data_classic.txt data_classic.bzz")
  compressed_size = os.path.getsize("data_classic.bzz")
  compressed_sizes[total_len] = compressed_size

# %%
import matplotlib.pyplot as plt
plt.plot(*zip(*compressed_sizes.items()))
plt.xlabel("length of str to be compressed")
plt.ylabel("compressed size")
# %%

# %%
