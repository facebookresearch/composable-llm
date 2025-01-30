# %%
import zlib

import matplotlib.pyplot as plt
import numpy as np
from src.nanollama.utils import initialize_nested_dataclass

from src.nanollama.data import gssm


def estimate_seq_entropy_by_compression(seq):
    seq = seq.tobytes()
    compressed_data = zlib.compress(seq, level=9)

    # Calculate entropy as compressed size / original size
    original_size = len(seq)
    entropy = len(compressed_data) / original_size if original_size > 0 else 0
    return entropy

# %%


def get_entropy_from_config(gssm_config, bsz=20000, seq_len=128):
    config = initialize_nested_dataclass(gssm.GSSMConfig, gssm_config)

    nodes = gssm.build_gssm(config, np.random.default_rng())
    seqs = np.empty((bsz, seq_len), dtype=int)

    nodes["X"].initialize(bsz)
    for i in range(seq_len):
        nodes["X"].evolve()
        seqs[:, i] = nodes["X"].state

    comp_rates = [estimate_seq_entropy_by_compression(seq) for seq in seqs]
    entropy_estimate = (np.mean(comp_rates).item(), np.std(comp_rates).item())
    return entropy_estimate


gssm_config_base = {
    "nodes": [
        {"name": "Z1", "state_dim": 2, "parents": [], "alpha": 0.3},
        {"name": "Z2", "state_dim": 2, "parents": ["Z1"], "alpha": 0.01},
        {"name": "Z4", "state_dim": 2, "parents": ["Z2", "Z1"], "alpha": 0.1},
        {
            "name": "X",
            "state_dim": 16,
            "parents": ["Z1", "Z4"],
            "alpha": 0.1,
        },
    ]
}


def gssm_config_template(alpha):
    return {
        "nodes": [
            {"name": "Z1", "state_dim": 2, "parents": [], "alpha": 0.1},
            {"name": "Z2", "state_dim": 2, "parents": ["Z1"], "alpha": 0.01},
            {"name": "Z4", "state_dim": 2, "parents": ["Z2", "Z1"], "alpha": 0.1},
            {
                "name": "X",
                "state_dim": 16,
                "parents": ["Z1", "Z4"],
                "alpha": float(alpha),
            },
        ]
    }


base_H, base_H_std = get_entropy_from_config(gssm_config_base)


def objective(alpha):
    cfg = gssm_config_template(alpha)
    new_H, new_H_std = get_entropy_from_config(cfg)
    return (new_H - base_H) ** 2


# minimize
# result = minimize_scalar(objective, bounds=(0.0001, 10), method="bounded")
# result

# %%
hs = []
alphas = np.logspace(-3, 1, 50)
for alpha in alphas:
    cfg = gssm_config_template(alpha)
    new_H, new_H_std = get_entropy_from_config(cfg)
    hs.append(new_H)

# %%
plt.plot(alphas, [abs(h - base_H) for h in hs])
# truth
plt.axvline(gssm_config_base["nodes"][-1]["alpha"])
plt.xscale("log")

# %%
from pgmpy.models import DynamicBayesianNetwork as DBN
dbn = DBN()
dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1))])

# %%
[(n, nbrdict) for n, nbrdict in dbn.adjacency()]

# %%
































