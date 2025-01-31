# %%
import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model.transformer import Transformer, TransformerConfig, TransformerBlockConfig
from gssm_as_hmm import HMM
# %%

alpha_z = .05
alpha_x = 1e-2

# don't make the state dims too big, otherwise you'll run out of gpu memory for entropy calc.
# prod_i(state_dim(Z_i)) <= 2048 is advisable
config = {"nodes": [
          {
              "name": "Z1",
              "state_dim": 16,
              "parents": [],
              "alpha": alpha_z,
              "mode": "slow", # or default or dead
              "kernel_type": "fullrank",
              "observed": False,
          },
          {
              "name": "Z2",
              "state_dim": 16,
              "parents": ["Z1"],
              "alpha": alpha_z,
              "mode": "default",
              "kernel_type": "fullrank",
              "observed": False,
          },
          {
              "name": "X",
              "state_dim": 128,
              "parents": ["Z1", "Z2"], # think up any graph you like
              "alpha": alpha_x,
              "mode": "default",
              "kernel_type": "fullrank",
              "observed": True,
          },
      ]}

def make_data(hmm, bsz, seq_len):
    hmm._init_all_nodes(bsz)
    data = np.zeros((seq_len, bsz), int)

    for i in range(seq_len):
        data[i] = hmm.top_node.state
        hmm.evolve_classic(1)
    return data

seq_len = 128
emb_dim = 64
nb_heads = 4
nb_layers = 4
n_train = 5000
n_test = 1000

seed = 1319
hmm = HMM(config, random_seed=seed)
train_data = make_data(hmm, n_train, seq_len).T
test_data = make_data(hmm, n_test, seq_len).T
hmm_estimate = hmm.entropy_of_observations(test_data.T, fast=True, small_mem=False, final_entry_only=False)
NLL = hmm_estimate.mean(1) / torch.arange(seq_len)

hmm_mean = (hmm_estimate[-1] / (seq_len - 1)).mean().item()
hmm_mean
# %%
block_cfg = TransformerBlockConfig(seq_len=seq_len, emb_dim=emb_dim, nb_heads=nb_heads)
t_cfg = TransformerConfig(block_cfg, vocab_size=hmm.top_node.state_dim, emb_dim=emb_dim, nb_layers=nb_layers)
model = Transformer(t_cfg)

def loss_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = preds.size(-1)
    return torch.nn.functional.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")

device = "cuda"

batch_size = 512
epochs = 100

train_data = torch.as_tensor(train_data).to(device)
test_data = torch.as_tensor(test_data).to(device)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


min_test_loss = 1e5
last_updated = 0
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
      loss_test = loss_func(model(test_data)[:,:-1], test_data[:,1:]).mean().item()
      # early stopping
      if loss_test < min_test_loss:
        min_test_loss = loss_test
        last_updated = epoch
      else:
        if epoch - last_updated > 3:
           break
      # update progress bar
      pbar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.3f}/{loss_test:.4f}, hmm: {hmm_mean:.4f}, kl:{loss_test-hmm_mean:.4f}")

# %%
with torch.inference_mode():
  loss_test = loss_func(model(test_data)[:,:-1], test_data[:,1:]).view(test_data[:,1:].shape).mean(0)
  plt.plot(loss_test.cpu(), label="loss")
  plt.plot(NLL.cpu(), label="entropy estimate")
  plt.xlabel('sequence position')
  plt.ylabel('entropy or Loss')
  plt.legend()

# %%
