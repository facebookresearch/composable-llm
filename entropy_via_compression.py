# %%
import numpy as np
from hidden_markov_model import HMM
import os
import tqdm
import matplotlib.pyplot as plt


def get_config(alpha):
    return {
        "nodes": [
            {
                "name": "Z1",
                "state_dim": 5,
                "parents": [],
                "alpha": alpha,
                "mode": "default",
            },
            {
                "name": "Z2",
                "state_dim": 4,
                "parents": ["Z1"],
                "alpha": alpha,
                "mode": "default",
            },
            {
                "name": "Z3",
                "state_dim": 3,
                "parents": ["Z2"],
                "alpha": alpha,
                "mode": "default",
            },
            {
                "name": "X",
                "state_dim": 2,
                "parents": ["Z1", "Z3"],
                "alpha": alpha,
                "mode": "default",
            },
        ]
    }


# %%
def make_data(hmm, bsz, seq_len):
    hmm._init_all_nodes(bsz)
    data = np.zeros((seq_len, bsz), int)

    for i in range(1, seq_len):
        data[i] = hmm.top_node.state
        hmm.evolve_classic(1)
    return data


def serialize_data(data, split=" "):
    # expect [T, B]
    return split.join(map(str, data.T.flatten())) + split


data = []
n_per_seed = 1000
n_seeds = 1
n_total = n_seeds * n_per_seed
print(f"n_total: {n_total} with {n_seeds} seeds")
n_plot_steps = 50
seq_len = 100
min_new_data = 1
uncompressed_file_name = "data_classic.txt"
compressed_file_name = "data_classic.bzz"


def get_compressed_size():
    os.system(f"../bzz/bzz -e1G {uncompressed_file_name} {compressed_file_name}")
    return os.path.getsize(compressed_file_name)


for alpha in [0.005, 0.01, 0.05, 0.1, 1, 10]:
# for alpha in [.1]:
    gssm_config = get_config(alpha)
    data = np.zeros((seq_len, n_total), dtype=int)
    entropies = []
    for i in range(n_seeds):
        hmm = HMM(gssm_config, random_seed=i * 301 + 19032093)
        data_i = make_data(hmm=hmm, bsz=n_per_seed, seq_len=seq_len)
        data[:, i * n_per_seed : (i + 1) * n_per_seed] = data_i
        entropies.append(hmm.entropy_of_observations(data_i).mean().item())

    data_increments = np.logspace(1, np.log10(n_total), n_plot_steps)

    uncompressed_sizes = {}
    compressed_sizes = {}
    if os.path.exists(uncompressed_file_name):
        os.remove(uncompressed_file_name)
    for n_data in tqdm.tqdm(data_increments):
        n_data = int(n_data)
        data_str = serialize_data(data[:n_data])
        str_len = len(data_str)
        with open(uncompressed_file_name, "w") as f:
            f.write(data_str)
        size = get_compressed_size()
        uncomp_size = os.path.getsize(uncompressed_file_name)
        compressed_sizes[str_len] = size
        uncompressed_sizes[str_len] = uncomp_size

    cs = np.array([compressed_sizes[l] for l in compressed_sizes], dtype=float)
    cdiffs = cs[1:] - cs[:-1]
    us = np.array([uncompressed_sizes[l] for l in uncompressed_sizes], dtype=float)
    udiffs = us[1:] - us[:-1]
    ratios = cdiffs / udiffs

    plt.plot(us[1:], ratios, label=f"alpha: {alpha}, entropy {entropies[0]/seq_len:.3f}")
    # plt.axhline(list(compressed_sizes.values())[-1], alpha=.5, color="r")

plt.xscale("log")
# plt.yscale("log")
plt.xlabel("length of str to be compressed")
plt.ylabel("compressed size / uncompressed size")
plt.legend()

plt.show()

# %%
uncompressed_sizes
# %%
entropies
# %%
