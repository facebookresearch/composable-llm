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
                "state_dim": 2,
                "parents": [],
                "alpha": alpha,
                "mode": "default",
                "observed": False,
            },
            # {
            #     "name": "Z2",
            #     "state_dim": 3,
            #     "parents": ["Z1"],
            #     "alpha": alpha,
            #     "mode": "default",
            #     "observed": False,
            # },
            # {
            #     "name": "Z3",
            #     "state_dim": 4,
            #     "parents": ["Z2"],
            #     "alpha": alpha,
            #     "mode": "default",
            #     "observed": False,
            # },
            {
                "name": "X",
                "state_dim": 8,
                "parents": ["Z1"],
                "alpha": alpha,
                "mode": "default",
                "observed": True,
            },
        ]
    }


# %%
def make_data(hmm, bsz, seq_len):
    hmm._init_all_nodes(bsz)
    data = np.zeros((seq_len, bsz), int)

    for i in range(seq_len):
        data[i] = hmm.top_node.state
        hmm.evolve_classic(1)
    return data.T


def serialize_data(data, split=""):
    # expect [B, T]
    return split.join(map(str, data.flatten())) + split


def get_compressed_uncompressed_size(data):
    uncompressed_file_name = "data_classic.txt"
    compressed_file_name = "data_classic.bzz"
    data_str = serialize_data(data)
    with open(uncompressed_file_name, "w") as f:
        f.write(data_str)
    os.system(f"../bzz/bzz -e1G {uncompressed_file_name} {compressed_file_name}")
    return os.path.getsize(compressed_file_name), os.path.getsize(
        uncompressed_file_name
    )


def get_entropy_estimate_from_compression(train, val):
    c_train, _ = get_compressed_uncompressed_size(train)
    B, T = val.shape
    entropys = []
    for bi in range(B):
        new_data = np.concatenate((train, val[[bi]]), axis=0)
        c_new, _ = get_compressed_uncompressed_size(new_data)
        entropys.append((c_new - c_train) * np.log(2))
    return np.array(entropys)


def get_entropy_estimate_from_hmm(hmm, data):
    entropys = hmm.entropy_of_observations(data.T)
    return entropys.numpy()


data = []
n_seeds = 1
n_train = 10000
n_val = 200
n_plot_steps = 10
seq_len = 20

print(
    f"checking with {n_seeds} seeds {n_train} n_train and {n_val} n_val, seq_len {seq_len}"
)


# 1. make multiple data :done:
#   multiple sequences with different realizations of the transition matrices
# 2. calculate an entropy based on compression (C(xX)-C(X))*log(2) (x is the particular realization, X is the longer data)
#   first calculate only on the X that have been made in the same batch (same seed), taking one out every time
#   compare this to a hmm entropy estimate on the same elements.

# for alpha in [0.005, 0.01, 0.05, 0.1, 1, 10]:
for alpha in [100000]:
    gssm_config = get_config(alpha)
    train_data = []
    val_data = []
    hmms = []
    assert n_seeds == 1
    for i_seed in range(n_seeds):
        hmms += [HMM(gssm_config, random_seed=i_seed * 301 + 192942)]
        train_data += [make_data(hmm=hmms[-1], bsz=n_train, seq_len=seq_len)]
        val_data += [make_data(hmm=hmms[-1], bsz=n_val, seq_len=seq_len)]

    # check compression rate as a function of train data
    for i_seed in range(n_seeds):
        data_increments = np.logspace(np.log10(2), np.log10(n_train), n_plot_steps)

        H_compr_mean = {}
        H_compr_std = {}
        hmm_estimates = get_entropy_estimate_from_hmm(
            hmms[i_seed], val_data[i_seed]
        )
        h_mean, h_std = np.mean(hmm_estimates), np.std(hmm_estimates)

        for n_data in tqdm.tqdm(data_increments):
            n_data = int(n_data)
            train = train_data[i_seed][:n_data]
            c_estimates = get_entropy_estimate_from_compression(
                train, val_data[i_seed]
            )
            c_mean, c_std = np.mean(c_estimates), np.std(c_estimates)
            H_compr_mean[n_data] = c_mean
            H_compr_std[n_data] = c_std




    plt.plot(H_compr_mean.keys(), H_compr_mean.values(), label=f"compr, alpha: {alpha}")
    # uncertainties
    plt.fill_between(
        H_compr_mean.keys(),
        [x - y for x, y in zip(H_compr_mean.values(), H_compr_std.values())],
        [x + y for x, y in zip(H_compr_mean.values(), H_compr_std.values())],
        alpha=.3
    )
    plt.axhline(h_mean, color="red")
    plt.axhspan(h_mean - h_std, h_mean + h_std, alpha=.3, color="red")
    

plt.xscale("log")
# plt.yscale("log")
plt.xlabel("number of training sequences")
plt.ylabel("entropy estimates")
plt.legend()

plt.show()

# %%
# sanity check, h_mean should be full entropy for every element in the seq except the first (0 entropy there)
(seq_len-1) * np.log(hmms[0].top_node.state_dim), h_mean
# %%
