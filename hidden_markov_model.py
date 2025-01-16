# %%
import numpy as np

# from omegaconf import OmegaConf
from nanollama.utils import initialize_nested_object
from src.nanollama.data import gssm
import numpy as np
import torch

class HMM:
    SYM_IN = "abcxdefghijklmnopqrstuvwxyz"
    SYM_OUT = "ABCXDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, config, random_seed=100):
        """
        makes an HMM from a graph config via the product state
        """
        self.config = initialize_nested_object(gssm.GSSMConfig, config, inplace=False)
        self.rng = np.random.default_rng(random_seed)
        self.top_node = gssm.build_gssm(self.config, self.rng)
        self.topo_order = self._dfs(self.top_node)
        self.indexs = {node: i for i, (_,node) in enumerate(self.topo_order)}
        self.transitions = {node: self._format_transition(node) for _,node in self.topo_order}

    def evolve_classic(self, steps):
        for _ in range(steps):
            self.top_node.evolve()
        return {name: node.state for name, node in self.topo_order}

    def _node_init(self, node: gssm.Node, bsz, i=0):
        node.time = 0
        for parent in node.parents:
            if parent.time != 0 and not parent.observed:
                self._node_init(parent, bsz, i + 1)
        node.state = np.zeros(bsz, dtype=int)
        assert (node.state == 0).all() # this is assumed in self.forward_probs
        # enable this to debug product transiiton and such
        # node.state = self.rng.integers(0,node.state_dim, size=bsz)

    def _init_all_nodes(self, bsz):
        self._node_init(self.top_node, bsz)

    def _dfs(self, node):
        def __dfs(node, fc=True):
            if not fc and node.observed:
                return [(node.name, node)]
            return [d for p in node.parents for d in __dfs(p, False)] + [
                (node.name, node)
            ]
        return list(dict.fromkeys(__dfs(node)))

    @staticmethod
    def _join_parent_kernels(node):
      n_in = len(node.kernels)
      # p(x|a), p(x|b) -> p(x|ab)
      # print("individual:\n", [x.sum(-1) for x in node.kernels])
      einsum_str = ",".join([f"{HMM.SYM_IN[i]}X" for i in range(n_in)])
      einsum_str += "->" + "".join([HMM.SYM_IN[i] for i in range(n_in)]) + "X"
      trans = np.einsum(einsum_str, *node.kernels)
      trans[trans.sum(-1) == 0] = 1/node.state_dim
      trans = trans / trans.sum(-1, keepdims=True)
      # print("prod_trans: \n", trans)
      return trans

    @staticmethod
    def _format_transition(node):
        parents = node.parents
        parent_state_dims = tuple([p.state_dim for p in parents])
        trans = HMM._join_parent_kernels(node)
        target_shape = tuple() if node.observed else (node.state_dim,)
        target_shape += parent_state_dims + (node.state_dim,)
        return trans.reshape(target_shape)

    def _node_sym_in(self, node):
        return HMM.SYM_IN[self.indexs[node]]

    def _node_sym_out(self, node):
        return HMM.SYM_OUT[self.indexs[node]]

    @staticmethod
    def _one_hot_state(node):
        targets = node.state
        nb_classes = node.state_dim
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

    def one_hot_product_state(self, node):
        node_order = self._dfs(node)
        einsum_str = (
            "B"
            + ",B".join(HMM.SYM_IN[: len(node_order)])
            + "->B"
            + HMM.SYM_IN[: len(node_order)]
        )
        product_state = np.einsum(
            einsum_str, *[self._one_hot_state(node) for _,node in node_order]
        )
        return product_state.reshape(product_state.shape[0], -1)

    def product_state(self, tgt_node):
        return self.one_hot_product_state(tgt_node).argmax(-1)

    def individual_states(self, prod_state, node_order):
        state_dims = [node.state_dim for _,node in node_order]
        return np.unravel_index(prod_state, state_dims)

    @staticmethod
    def square_matrix(m):
        dim = int(np.prod(m.shape) ** 0.5)
        return m.reshape(dim, dim)

    def einsum_input_str(self, tgt_node):
        """
        Constructs the einsum string for the target node transition matrix when used as input
        """
        einsum_str = self._node_sym_in(tgt_node) if not tgt_node.observed else ""
        einsum_str += "".join(self._node_sym_out(p) for p in tgt_node.parents)
        einsum_str += self._node_sym_out(tgt_node)
        return einsum_str

    def einsum_full_prod_str(self, tgt_node):
        """
        Constructs the einsum string for the target node transition matrix in the product state form
        """
        ordered_nodes = [node for _,node in self._dfs(tgt_node)]
        in_str = "".join(self._node_sym_in(n) for n in ordered_nodes)
        out_str = "".join(self._node_sym_out(n) for n in ordered_nodes)
        return in_str + out_str

    def make_prod_transition(self, tgt_node):
        node_order = [node for _,node in self._dfs(tgt_node)]
        state_dims = [node.state_dim for node in node_order]
        prod_input_str = HMM.SYM_IN[: len(node_order)]

        def prod_str(node):
            return f"{prod_input_str}{self._node_sym_out(node)}"

        einsum_input_strs = {node: self.einsum_input_str(node) for node in node_order}
        einsum_prod_strs = {node: prod_str(node) for node in node_order}
        einsum_prod_strs[None] = prod_input_str
        # a collection of einsum_str -> array to use for the actual einsum
        # we build the logic only with the strings and fetch the matrices from here
        einsum_str_to_arr = {prod_input_str: np.ones(state_dims)}  # this is the root (for convenience)
        einsum_str_to_arr |= {s: self.transitions[node] for node, s in einsum_input_strs.items()}
        # step 1 : fill the abcxX matrices
        for node in node_order:
            parents = node.parents or [None]
            # this is good for Tabcx_A
            input_strs = [
                einsum_input_strs[node],
                *[einsum_prod_strs[p] for p in parents],
            ]
            output_str = einsum_prod_strs[node]
            einsum_str = ",".join(input_strs) + "->" + output_str
            input_tensors = [einsum_str_to_arr[s] for s in input_strs]
            einsum_str_to_arr[output_str] = np.einsum(einsum_str, *input_tensors)

        # step 2 : get final product transition
        input_strs = [einsum_prod_strs[node] for node in node_order]
        input_tensors = [einsum_str_to_arr[s] for s in input_strs]
        output_str = self.einsum_full_prod_str(tgt_node)
        einsum_str = ",".join(input_strs) + "->" + output_str

        return self.square_matrix(np.einsum(einsum_str, *input_tensors))

    def fwd_product_state(self, prod_transition):
        raise NotImplementedError()
        p_transition = self.square_matrix(prod_transition)
        kernel = gssm.TransitionKernel(*p_transition.shape, 1)
        kernel.p_transition = p_transition
        kernel._cumulative = np.cumsum(kernel.p_transition, axis=1)
        next_state = kernel(self.product_state(self.top_node))
        return self.individual_states(next_state, self.topo_order)

    def fwd_via_matmul(self, prod_transition):
        state = self.one_hot_product_state(self.top_node)
        next_state = state @ prod_transition
        # sampling
        samples = torch.multinomial(torch.tensor(next_state), 1).numpy()
        return self.individual_states(samples, self.topo_order)

    def get_p_emission(self, tgt_node):
        # p_emission is a projection matrix from the product state
        # state @ p_emission = state_X
        assert tgt_node.observed
        idx = self.indexs[tgt_node]
        shape = [1] * (len(self.topo_order) + 1)
        dim = tgt_node.state_dim
        shape[idx] = dim
        shape[-1] = dim
        p_emission = np.eye(dim).reshape(shape)
        for i, (_,node) in enumerate(self.topo_order):
            shape[i] = node.state_dim
        p_emission = np.broadcast_to(p_emission, shape)
        p_emission = p_emission.reshape(-1, dim)
        return p_emission

    def current_one_hot_product_state(self):
        return self.one_hot_product_state(self.top_node)

    @staticmethod
    def forward_algorithm(
        obs: torch.tensor,
        log_T: torch.tensor,
        log_E: torch.tensor,
        log_pi: torch.tensor,
        device="cuda",
    ):
        """
        Perform the forward-backward algorithm to compute the forward and backward probabilities.
        S = hidden state vocab
        O = observation state vocab

        Args:
            obs (torch.tensor): List of observed sequences [seq_len,B] (batch last)
            T (torch.tensor): Transition matrix. [S, S]
            E (torch.tensor): Emission matrix. [S, O]
            pi (torch.tensor): Initial state probabilities. [S]
        Returns:
            forward_probs (torch.tensor): Forward probabilities [S, seq_len, B]
        """
        num_states = log_T.shape[0]
        T, B = obs.shape
        log_T = log_T.to(device)
        log_E = log_E.to(device)
        log_pi = log_pi.to(device)
        obs = obs.to(device)

        def lognorm(log_x):
            return torch.logsumexp(log_x, dim=0, keepdim=True)

        forward_probs = torch.zeros((num_states, T, B), device=device)
        log_p_seq = torch.zeros((1, T, B), device=device)

        forward_probs[:, 0, :] = log_pi[:, None] + log_E[:, obs[0]]
        #forward_probs[:,t,:] = p(Z_t, X_[t])
        log_p_seq[:,0,:] = lognorm(forward_probs[:, 0, :])
        # forward_probs[:,0,:] -= log_p_seq[:,0,:]
        

        log_T = log_T[:, :, None]

        for t in range(1, T):
            forward_probs[:, t, :] = log_E[:, obs[t]] + torch.logsumexp(
                forward_probs[:, None, t - 1, :] + log_T, dim=0
            )
            log_p_seq[:,t,:] = lognorm(forward_probs[:, t, :])
            # forward_probs[:, t, :] = forward_probs[:,t,:] - log_p_seq[:,t,:]

        return forward_probs.cpu(), log_p_seq.reshape(T, B).cpu()

    def forward_probs(self, observations: np.ndarray):
        T, B = observations.shape
        observations = torch.tensor(observations)
        self._init_all_nodes(B)
        prior = torch.tensor(self.current_one_hot_product_state()[0])
        transition = torch.tensor(self.make_prod_transition(self.top_node))
        emission = torch.tensor(self.get_p_emission(self.top_node))
        return self.forward_algorithm(observations, transition.log(), emission.log(), prior.log())
    
    def entropy_of_observations(self, observations: np.ndarray):
        _, log_xst = self.forward_probs(observations)
        H_t = -log_xst
        H_T = H_t[-1]
        return H_T
        


if __name__ == "__main__":

  gssm_config = {
      "nodes": [
            {
                "name": "Z1",
                "state_dim": 2,
                "parents": [],
                "alpha": .05,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "Z2",
                "state_dim": 3,
                "parents": ["Z1"],
                "alpha": .05,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "Z3",
                "state_dim": 4,
                "parents": ["Z2"],
                "alpha": .05,
                "mode": "default",
                "observed": False,
            },
            {
                "name": "X",
                "state_dim": 5,
                "parents": ["Z1", "Z3"],
                "alpha": .05,
                "mode": "default",
                "observed": True,
            },
      ]
  }

  def test_prod_transition(config):
    hmm = HMM(config, random_seed=42)
    hmm._init_all_nodes(1000)
    prod_transition = hmm.make_prod_transition(hmm.top_node)
    # data_prod = hmm.fwd_product_state(prod_transition)
    # data_prod = {name: data_prod[i] for i, name in enumerate(hmm.topo_order)}
    data_prod_mm = hmm.fwd_via_matmul(prod_transition)
    data_prod_mm = {name: data_prod_mm[i] for i, (name,_) in enumerate(hmm.topo_order)}
    data_classic = hmm.evolve_classic(1)
    import matplotlib.pyplot as plt
    for name in data_classic:
        plt.title(name)
        # plt.hist(data_prod[name], label="prod", alpha=0.5)
        plt.hist(data_prod_mm[name], label="prod_mm", alpha=0.5)
        plt.hist(data_classic[name], label="classic", alpha=0.5)
        plt.legend()
        plt.show()

  test_prod_transition(gssm_config)

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

  def test_entropy(config, seq_len, batch_size, seed=3892493):
      observations = np.zeros((seq_len, batch_size), dtype=int)
      n_seeds = 10
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
      return np.mean(entropys)
  
  for seq_len in np.logspace(0,np.log10(100),10):
    seq_len = int(seq_len)
    print(seq_len, test_entropy(gssm_config, seq_len, 200) / seq_len)

# %%
