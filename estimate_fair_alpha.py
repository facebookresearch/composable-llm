# %%
from src.nanollama.data import gssm
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from src.nanollama.data.gssm import estimate_seq_entropy_by_compression
from scipy.optimize import minimize_scalar

# %%

def get_entropy_from_config(gssm_config, bsz=128, seq_len=1024):
  default_config = OmegaConf.structured(gssm.GSSMConfig())
  config = OmegaConf.merge(default_config, gssm_config)
  config = OmegaConf.to_object(config)
  config.__manual_post_init__()

  nodes = gssm.build_gssm(config, np.random.default_rng())
  seqs = np.empty((bsz, seq_len), dtype=int)

  nodes["X"].initialize(bsz)
  for i in range(seq_len):
    nodes["X"].evolve()
    seqs[:,i] = nodes["X"].state

  comp_rates = [estimate_seq_entropy_by_compression(seq) for seq in seqs]
  entropy_estimate = (np.mean(comp_rates).item(), np.std(comp_rates).item())
  return entropy_estimate


gssm_config_base = OmegaConf.create({
    "nodes": [
        {"name": "Z1", "state_dim": 2, "parents": [], "alpha": .1},
        {"name": "Z2", "state_dim": 3, "parents": ["Z1"], "alpha": .1},
        # {"name": "Z4", "state_dim": 2, "parents": ["Z3", "Z1"], "alpha": 1},
        {
            "name": "X",
            "state_dim": 10,
            "parents": ["Z1", "Z2"],
            "alpha": .1,
        },
    ]
})

def gssm_config_template(alpha):
  return OmegaConf.create({
      "nodes": [
          {"name": "Z1", "state_dim": 2, "parents": [], "alpha": .1},
          {"name": "Z4", "state_dim": 3, "parents": ["Z1"], "alpha": .1},
          {
              "name": "X",
              "state_dim": 10,
              "parents": ["Z1", "Z4"],
              "alpha": float(alpha),
          },
      ]
  })

base_H, base_H_std = get_entropy_from_config(gssm_config_base)
def objective(alpha):
  cfg = gssm_config_template(alpha)
  new_H, new_H_std = get_entropy_from_config(cfg)
  return ((new_H - base_H))**2

# minimize
result = minimize_scalar(objective, bounds=(0.0001, 10), method="bounded")
result

# %%
hs = []
alphas = np.logspace(-3, 1, 30)
for alpha in alphas:
  cfg = gssm_config_template(alpha)
  new_H, new_H_std = get_entropy_from_config(cfg)
  hs.append(new_H)

plt.plot(alphas, [np.abs(h - base_H) for h in hs])
#truth
plt.axvline(gssm_config_base["nodes"][-1]["alpha"])
plt.xscale('log')
# %%
