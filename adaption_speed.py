# %%
from apps.gssm.hidden_markov_model import *
import matplotlib.pyplot as plt
import tqdm
from src.nanollama.model.transfomer import Transformer, TransformerConfig, TransformerBlockConfig

# this forward algorithm allows me to 

def forward_probs_full(hmm: HMM, observations: np.ndarray, device: str = "cuda", small_mem=False):
    T, B = observations.shape
    observations = torch.as_tensor(observations, dtype=torch.int32, device=device)
    hmm._init_all_nodes(B)
    prior = torch.tensor(hmm.one_hot_product_state_fast()[0], device=device, dtype=torch.float32).log()
    transition = torch.tensor(hmm.make_prod_transition_fast(), device=device, dtype=torch.float32).log()
    emission = torch.tensor(hmm.get_p_emission_fast(), device=device, dtype=torch.float32).log()
    log_fwd_p, log_p_seq = hmm.forward_algorithm(observations, transition, emission, prior, device=device, small_mem=small_mem)
    # log_fwd_p = p(Z_t, X_[t]=x_[t])
    # make p(X_t | X_[t-1]=x_[t-1]) from it
    # this is the same as below
    # log_fwd_p = torch.einsum("he,ih,isb->esb", emission, transition, log_fwd_p.exp()) (normed)
    S, T, B = log_fwd_p.shape
    p_zt_z_x = torch.zeros(transition.shape[1], T, B, device=log_fwd_p.device)
    p_xt_xst = torch.zeros(emission.shape[1], T, B, device=log_fwd_p.device)
    for t in range(T):
      p_zt_z_x[:,t,:] = torch.logsumexp(transition[:,:,None] + log_fwd_p[:,None,t,:], dim=0) #"ih,isb->hsb" forward_probs[:, None, t - 1, b] + log_T
      p_xt_xst[:,t,:] = torch.logsumexp(emission[:,:,None] + p_zt_z_x[:,None,t,:], dim=0) #"ih,hsb->hsb" forward_probs[:, None, t - 1, b] + log_T
    #norm
    p_xt_xst = p_xt_xst - torch.logsumexp(p_xt_xst, dim=0,keepdim=True)
    # note that p_xt_xst one is one forward now, i.e. p_xt_xst[:,t,:] = p(X_t+1| X_[t]=x_[t])
    return p_xt_xst.cpu(), log_p_seq.cpu()


def entropy_of_observations(hmm:HMM, observations: np.ndarray, device: str = "cuda"):
    log_fwd_p, log_xst = forward_probs_full(hmm, observations, device=device)
    H_t = -log_xst
    return log_fwd_p, H_t


def make_data(hmm: HMM, batch_size, seq_len):
    hmm._init_all_nodes(batch_size)
    observations = {n:np.zeros((seq_len, batch_size), dtype=int) for n,_ in hmm.topo_order}
    for i in range(seq_len):
      for name, node in hmm.topo_order:
        observations[name][i] = np.array(node.state)
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
            # { "name": "Z4", "state_dim": state_dim, "parents": ["Z12"], "alpha": alpha, "mode": "default", "kernel_type": "product", "observed": False},

            # who are X's parents?
            { "name": "X", "state_dim": 128, "parents": ["Z3"], "alpha": alpha_X, "mode": "default", "kernel_type": "fullrank", "observed": True},
            # { "name": "X", "state_dim": 128, "parents": ["Z3", "Z4"], "alpha": alpha_X, "mode": "default", "kernel_type": "product", "observed": True},
        ]
    }
    return cfg

n_data = 4
seq_len = 128

state_dim_z = 128
alpha_z = 1e-6

alpha_x = 1e-5

alpha_controller = 3e-2 # this is for the slow node
state_dim_controller = 16

def hardcode_controller_transition(hmm:HMM, controller_node_names, enable_states=slice(1,state_dim_controller)):
    controller_nodes = [node for name,node in hmm.topo_order if name in controller_node_names]
    for cnode in controller_nodes:
      assert len(cnode.parents) == 0
      cnode.kernels[0][0,:] = 0
      cnode.kernels[0][0,enable_states] = 1
      cnode.kernels[0][:,0] = 0
      cnode.kernels[0] /= cnode.kernels[0].sum(-1,keepdims=True)
    hmm.transitions = {node: hmm._format_transition(node) for _, node in hmm.topo_order}

cfg = get_cfg(state_dim_controller, alpha_controller, state_dim_z, alpha_z, alpha_x)
# seed = np.random.randint(1924289)
seed = 2498
hmm = HMM(cfg, random_seed=seed)
hardcode_controller_transition(hmm, "Z12")
data = make_data(hmm, n_data, seq_len)
# hmm_estimate = entropy_of_observations(data["X"], fast=True, small_mem=False, final_entry_only=False)
_, H_t = entropy_of_observations(hmm, data["X"])
# %%
for i,seq in enumerate(H_t.T):
  scale = 5
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
n_train = 3000
n_test = 100

# for train data enable only the first half
# hardcode_controller_transition(hmm, "Z12", enable_states=slice(1, state_dim_controller//2))
train_data_all = make_data(hmm, n_train, seq_len)
# for valid data enable only the second half
# hardcode_controller_transition(hmm, "Z12", enable_states=slice(state_dim_controller//2, state_dim_controller))
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
min_loss_test = 1e5
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
      pbar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.3f}/{loss_test:.4f}, hmm: {hmm_mean:.4f}, kl:{loss_test-hmm_mean:.4f}")
      #early stopping
      if loss_test <= min_loss_test:
        updated = epoch
        min_loss_test = loss_test
      else:
        if epoch - updated > 5:
          break

# %%
model.eval()
which_seqs = slice(0,100)
with torch.inference_mode():
  x = test_data[which_seqs]
  out = model(x).cpu()
  tgt = test_data[which_seqs].cpu()
  log_fwd_p, entropys = entropy_of_observations(hmm, x.cpu().T)
  entropys = entropys.T.diff(dim=-1) # diff makes them conditional
  nll = loss_func(out[:,:-1], tgt[:,1:]).view(entropys.shape)
  # KL(p_true(X_t | X_<t=x_<t) | p_model(X_t | X_<t=x_<t))
  kl = torch.kl_div(torch.log_softmax(out, dim=-1), log_fwd_p.permute(2,1,0), log_target=True).sum(-1)
# %%
for i in range(56,57):
  plt.plot(entropys[i] + kl[i][:-1], label="true NLL + KL")
  # plt.plot(kl[i][:-1], label="true NLL + KL")
  plt.plot(entropys[i], "--", label="true NLL")
  plt.fill_between(range(seq_len-1), entropys[i], entropys[i] + kl[i][:-1], alpha=.5, label="excess loss (KL)")
  # plt.plot(nll[i][1:], "--", label="model NLL")
  if "Z1" in test_data_all:
    z_changeds = (np.diff(test_data_all["Z1"][:,i]) != 0) | (np.diff(test_data_all["Z2"][:,i]) != 0)
  else:
    z_changeds = (np.diff(test_data_all["Z12"][:,i]) != 0)
  # [plt.axvline(j-1, alpha=.5) for j,did in enumerate(z_changeds) if did and j > 0]
  plt.legend()
  # plt.xscale('log')
  # plt.yscale('log')
  plt.ylim(0,.2)
  plt.ylabel('NLL or NLL+KL')
  plt.xlabel('sequence position')
  plt.savefig("adaption_speed_plot_wobbly.pdf")
  plt.show()

# %%
