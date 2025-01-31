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
                "state_dim": 2,
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


def pack_bytes(data, bit_width:int) -> bytes:
    """Convert list if integers to packed byte string.
    Each integer should only consume the specified number of bits
    """

    packed_bytes = bytearray()
    buffer = 0
    bits_buffered = 0
    for sample in data:
        bit_mask = 0x01
        for i in range(bit_width):
            bit = (sample & bit_mask) >> i
            bit_mask <<= 1
            buffer |= (bit << bits_buffered)
            bits_buffered += 1

            if bits_buffered == 8:
                packed_bytes.append(buffer)
                buffer = 0
                bits_buffered = 0
    if bits_buffered != 0:
        packed_bytes.append(buffer)

    return packed_bytes

def serialize_data(data : np.ndarray, length):
    # expect [B, T]
    return pack_bytes(data.flatten(), length)
    return np.packbits(data.flatten().astype(np.int8))
    return split.join(map(str, data.flatten())) + split


def get_compressed_uncompressed_size(data):
    uncompressed_file_name = "data_classic.txt"
    compressed_file_name = "data_classic.bzz"
    length = int(np.ceil(np.log2(train.max()+1)))
    data_str = serialize_data(data, length)
    with open(uncompressed_file_name, "wb") as f:
        f.write(data_str)
    os.system(f"../bzz/bzz -e1G {uncompressed_file_name} {compressed_file_name}")
    return 8/length * os.path.getsize(compressed_file_name), 8/length * os.path.getsize(
        uncompressed_file_name
    )


def get_entropy_estimate_from_compression(train, val, n_train, n_estimates=5):
    choose_train = np.random.permutation(train.shape[0])[:n_train]
    # train_ = train[choose_train]
    train_ = train[:n_train]

    B, T = val.shape
    assert B//n_estimates == B/n_estimates
    bsz = B//n_estimates
    entropys = []

    for lo,hi in zip(range(0, B, bsz), range(bsz, B+bsz, bsz)):
        c_train, u_train = get_compressed_uncompressed_size(train_)
        new_data = np.concatenate((train_, val[lo:hi]), axis=0)
        c_new, u_new = get_compressed_uncompressed_size(new_data)
        entropys.append((c_new - c_train) * np.log(2) / bsz)
    return np.array(entropys) / 1.0329 # this is an estimate for the overhead of the compressor


def get_entropy_estimate_from_hmm(hmm : HMM, data, n_estimates=5):
    B, T = data.shape
    assert B//n_estimates == B/n_estimates
    bsz = B//n_estimates
    entropys = []
    for lo,hi in zip(range(0, B, bsz), range(bsz, B+bsz, bsz)):
      entropys.append(hmm.entropy_of_observations(data[lo:hi].T).mean())
    return np.array(entropys)


data = []
n_seeds = 1
n_train = int(1e7)
n_val = 10000
n_plot_steps = 10
seq_len = 50

print(
    f"checking with {n_seeds} seeds {n_train} n_train and {n_val} n_val, seq_len {seq_len}"
)


# 1. make multiple data :done:
#   multiple sequences with different realizations of the transition matrices
# 2. calculate an entropy based on compression (C(xX)-C(X))*log(2) (x is the particular realization, X is the longer data)
#   first calculate only on the X that have been made in the same batch (same seed), taking one out every time
#   compare this to a hmm entropy estimate on the same elements.

# for alpha in [0.005, 0.01, 0.05, 0.1, 1, 10]:
for alpha in [.1]:
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

        H_diffs_mean = {}
        H_diffs_std = {}
        H_compr_mean = {}
        H_compr_std = {}
        hmm_estimates = get_entropy_estimate_from_hmm(
            hmms[i_seed], val_data[i_seed]
        )
        h_mean, h_std = np.mean(hmm_estimates), np.std(hmm_estimates)

        for n_data in tqdm.tqdm(data_increments):
            n_data = int(n_data)
            # train = train_data[i_seed][:n_data]
            train = train_data[i_seed]
            c_estimates = get_entropy_estimate_from_compression(
                train, val_data[i_seed], n_data
            )
            c_mean, c_std = np.mean(c_estimates), np.std(c_estimates)
            H_diffs_mean[n_data] = (c_estimates - hmm_estimates).mean()
            H_diffs_std[n_data] = (c_estimates - hmm_estimates).std()
            H_compr_mean[n_data] = c_mean
            H_compr_std[n_data] = c_std


    # plt.plot(H_diffs_mean.keys(), H_diffs_mean.values(), label=f"diff, alpha: {alpha}")
    # # uncertainties
    # plt.fill_between(
    #     H_diffs_mean.keys(),
    #     [x - y for x, y in zip(H_diffs_mean.values(), H_diffs_std.values())],
    #     [x + y for x, y in zip(H_diffs_mean.values(), H_diffs_std.values())],
    #     alpha=.3
    # )
    plt.plot(H_compr_mean.keys(), H_compr_mean.values(), label=f"compr, alpha: {alpha}")
    # uncertainties
    plt.fill_between(
        H_compr_mean.keys(),
        [x - y for x, y in zip(H_compr_mean.values(), H_compr_std.values())],
        [x + y for x, y in zip(H_compr_mean.values(), H_compr_std.values())],
        alpha=.3
    )
    plt.axhline(h_mean, color="red", label=f"hmm estimate")
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

data = np.random.choice(hmms[0].top_node.state_dim, size=(100000,20))

x, y = get_compressed_uncompressed_size(data)
print(x/y)
# %%

# %%
