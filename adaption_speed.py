# %%
from apps.gssm.hidden_markov_model import *
import matplotlib.pyplot as plt
import tqdm
from src.nanollama.model.transfomer import Transformer, TransformerConfig, TransformerBlockConfig

def make_data(hmm: HMM, batch_size, seq_len, change_state=lambda *args: None):
    hmm._init_all_nodes(batch_size)
    observations = {n:np.zeros((seq_len, batch_size), dtype=int) for n,_ in hmm.topo_order}
    for i in range(seq_len):
      for name, node in hmm.topo_order:
        observations[name][i] = np.array(node.state)
      change_state(hmm.topo_order, i)
      hmm.evolve_classic(1)
    return observations

# %%
def get_cfg(state_dim_control, alpha_control, state_dim, alpha, alpha_X=6e-3):
    cfg = {
        "nodes": [
            # one or two controller nodes 
            { "name": "Z12", "state_dim": state_dim_control, "parents": [], "alpha": alpha_control, "mode": "slow", "observed": False},
            # { "name": "Z1", "state_dim": state_dim_control, "parents": [], "alpha": alpha_control, "mode": "slow", "observed": False},
            # { "name": "Z2", "state_dim": state_dim_control, "parents": [], "alpha": alpha_control, "mode": "slow", "observed": False},

            # one or two hidden nodes?
            { "name": "Z3", "state_dim": state_dim, "parents": ["Z12"], "alpha": alpha, "mode": "default", "kernel_type": "fullrank", "observed": False},
            # { "name": "Z4", "state_dim": state_dim, "parents": ["Z12"], "alpha": alpha, "mode": "default", "kernel_type": "fullrank", "observed": False},

            # who are X's parents?
            { "name": "X", "state_dim": 128, "parents": ["Z3"], "alpha": alpha_X, "mode": "default", "kernel_type": "fullrank", "observed": True},
            # { "name": "X", "state_dim": 128, "parents": ["Z3", "Z4"], "alpha": alpha_X, "mode": "default", "kernel_type": "fullrank", "observed": True},
        ]
    }
    return cfg

n_data = 4
seq_len = 64
state_dim = 32
alpha_controller = 8e-3
state_dim_controller = 64
# alpha_z = 1.55e-1
# alpha_x = 1e-2
alpha_z = 1e-5
alpha_x = 1e-5
cfg = get_cfg(state_dim_controller, alpha_controller, state_dim, alpha_z, alpha_x)

def change_state(topo_order, seq_idx): # this doesn't really work, because the HMM calculation goes overboard
  if ((seq_idx + 1) % (seq_len//4)) == 0:
     for name, node in topo_order:
        if name in ["Z12","Z1"]:
          node.state = node.rng.integers(0, node.state_dim, size=node.state.shape)

# seed = np.random.randint(1924289)
seed = 745706
hmm = HMM(cfg, random_seed=seed)
data = make_data(hmm, n_data, seq_len)
hmm_estimate = hmm.entropy_of_observations(data["X"], fast=True, small_mem=False, final_entry_only=False)

# %%
for i,seq in enumerate(hmm_estimate.T):
  scale = 10
  offset = scale*i
  plt.plot(seq.diff() + offset, color=f"C{i}")
  if "Z1" in data:
    z_changeds = (np.diff(data["Z1"][:,i]) != 0) | (np.diff(data["Z2"][:,i]) != 0)
  else:
    z_changeds = (np.diff(data["Z12"][:,i]) != 0)
  [plt.plot([j,j],[offset,offset+scale], color=f"C{i}", alpha=.5) for j,did in enumerate(z_changeds) if did]
  plt.axhline(offset, color=f"C{i}")
plt.show()

# %%
emb_dim = 32
nb_heads = 2
nb_layers = 2
n_train = 1500
n_test = 100

train_data_all = make_data(hmm, n_train, seq_len)
test_data_all = make_data(hmm, n_test, seq_len)
hmm_estimate = hmm.entropy_of_observations(test_data_all["X"], fast=True, small_mem=False)
hmm_mean = (hmm_estimate / (seq_len - 1)).mean().item()

# %%

block_cfg = TransformerBlockConfig(seq_len=seq_len, emb_dim=emb_dim, nb_heads=nb_heads)
t_cfg = TransformerConfig(block_cfg, vocab_size=hmm.top_node.state_dim, emb_dim=emb_dim, nb_layers=nb_layers)
model = Transformer(t_cfg)

import tqdm
def loss_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = preds.size(-1)
    return torch.nn.functional.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")

device = "cuda"

batch_size = 128
epochs = 250

train_data = torch.as_tensor(train_data_all["X"].T).to(device)
test_data = torch.as_tensor(test_data_all["X"].T).to(device)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

# training loop
for epoch in (pbar:=tqdm.trange(epochs)):
    model.train()
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
      model.eval()
      loss_test = loss_func(model(test_data)[:,:-1], test_data[:,1:]).mean().item()
      # update progress bar
      pbar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.3f}/{loss_test:.4f}, hmm: {hmm_mean:.4f}, kl:{loss_test-hmm_mean:.4f}")

# %%
model.eval()
which_seqs = slice(0,100)
with torch.inference_mode():
  x = test_data[which_seqs]
  out = model(x).cpu()[:,:-1]
  tgt = test_data[which_seqs].cpu()[:,1:]
  entropys = hmm.entropy_of_observations(x.cpu().T, fast=True, small_mem=False, final_entry_only=False).T.diff(dim=-1)
  nll = loss_func(out, tgt).view(entropys.shape)
  kl = nll - entropys
# %%
for i in range(50):
  plt.plot(nll[i], label="pred nll")
  plt.plot(entropys[i], "--", label="true nll")
  if "Z1" in test_data_all:
    z_changeds = (np.diff(test_data_all["Z1"][:,i]) != 0) | (np.diff(test_data_all["Z2"][:,i]) != 0)
  else:
    z_changeds = (np.diff(test_data_all["Z12"][:,i]) != 0)
  [plt.axvline(j, alpha=.5) for j,did in enumerate(z_changeds) if did]
  plt.yscale('log')
  plt.show()

# %%
