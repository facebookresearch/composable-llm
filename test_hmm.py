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
            "alpha": .1,
            "mode": "default",
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
n_train = 100000
n_test = 1000
seed = np.random.randint(29042)

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


# %%
