
# %%
import sys
sys.path.append("../../../")
from functools import lru_cache
from logging import getLogger
import numpy as np
import torch

# from omegaconf import OmegaConf
from nanollama.utils import initialize_nested_object
from nanollama.data import gssm

logger = getLogger("nanollama")

# %%

class HMM:
    SYM_IN = "abcdefghijklmnopqrstuvwxyz"
    SYM_OUT = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, config=None, top_node=None, random_seed=100):
        """
        makes an HMM from a graph config via the product state
        """
        self.graph_config = config
        self.random_seed = random_seed
        if config is not None:
            assert top_node is None
            config = initialize_nested_object(gssm.GSSMConfig, config, inplace=False)
            self.rng = np.random.default_rng(random_seed)
            self.top_node = gssm.build_gssm(config, self.rng)
            self.graph_config 
        else:
            assert top_node is not None
            self.top_node = top_node
        logger.info("Retrieving topological order")
        self.topo_order = self._dfs(self.top_node)
        self.indexs = {node: i for i, (_, node) in enumerate(self.topo_order)}
        logger.info("Rewriting transition matrices")
        self.transitions = {node: self._format_transition(node) for _, node in self.topo_order}

    def evolve_classic(self, steps):
        for _ in range(steps):
            self.top_node.evolve()
        return {name: node.state for name, node in self.topo_order}

    def _node_init(self, node: gssm.Node, bsz):
        node.time = 0
        for parent in node.parents:
            self._node_init(parent, bsz)
        # node.state = self.rng.integers(node.state_dim, size=bsz, dtype=int)
        node.state = np.zeros(bsz, dtype=int)
        assert (node.state == 0).all()  # this is assumed in self.forward_probs

    def _init_all_nodes(self, bsz):
        self._node_init(self.top_node, bsz)

    def _dfs(self, node):
        def __dfs(node, fc=True):
            if not fc and node.observed:
                return [(node.name, node)]
            return [d for p in node.parents for d in __dfs(p, False)] + [(node.name, node)]

        return list(dict.fromkeys(__dfs(node)))

    @staticmethod
    def _join_parent_kernels(node: gssm.Node):
        n_in = len(node.kernels)
        # p(x|a), p(x|b) -> p(x|ab)
        # print("individual:\n", [x.sum(-1) for x in node.kernels])
        einsum_str = ",".join([f"{HMM.SYM_IN[i]}X" for i in range(n_in)])
        einsum_str += "->" + "".join([HMM.SYM_IN[i] for i in range(n_in)]) + "X"
        trans = np.einsum(einsum_str, *node.kernels)
        trans[trans.sum(-1) == 0] = 1 / node.state_dim
        trans = trans / trans.sum(-1, keepdims=True)
        # print("prod_trans: \n", trans)
        return trans

    @staticmethod
    def _format_transition(node: gssm.Node):
        logger.info(f"Rewriting transition matrix for {node}")
        parents = node.parents
        parent_state_dims = tuple([p.state_dim for p in parents])
        if node.kernel_type == "product":
          trans = HMM._join_parent_kernels(node)
        elif node.kernel_type == "fullrank":
          trans = node.kernels[0]
        else:
          raise ValueError(f"kernel_type {node.kernel_type} is not supported in HMM")
        target_shape = (1,) if node.observed else (node.state_dim,)
        target_shape += parent_state_dims + (node.state_dim,)
        trans = trans.reshape(target_shape)
        if trans.shape[0] == 1:
          trans = np.broadcast_to(trans, (node.state_dim,) + target_shape[1:])
        return trans

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

    def one_hot_product_state(self):
        node_order = self.topo_order
        einsum_str = "B" + ",B".join(HMM.SYM_IN[: len(node_order)]) + "->B" + HMM.SYM_IN[: len(node_order)]
        product_state = np.einsum(einsum_str, *[self._one_hot_state(node) for _, node in node_order])
        return product_state.reshape(product_state.shape[0], -1)

    def one_hot_product_state_fast(self):
        node_order = [(name, node) for name, node in self.topo_order if not node.observed]
        einsum_str = "B" + ",B".join(HMM.SYM_IN[: len(node_order)]) + "->B" + HMM.SYM_IN[: len(node_order)]
        product_state = np.einsum(einsum_str, *[self._one_hot_state(node) for _, node in node_order])
        return product_state.reshape(product_state.shape[0], -1)

    def product_state(self):
        return self.one_hot_product_state().argmax(-1)

    def individual_states(self, prod_state, node_order):
        state_dims = [node.state_dim for _, node in node_order]
        return np.unravel_index(prod_state, state_dims)

    @staticmethod
    def square_matrix(m):
        dim = int(np.prod(m.shape) ** 0.5)
        return m.reshape(dim, dim)

    def einsum_input_str(self, tgt_node):
        """
        Constructs the einsum string for the target node transition matrix when used as input
        """
        einsum_str = self._node_sym_in(tgt_node)
        einsum_str += "".join(self._node_sym_out(p) for p in tgt_node.parents)
        einsum_str += self._node_sym_out(tgt_node)
        return einsum_str

    def einsum_full_prod_str_fast(self):
        """
        Constructs the einsum string for the target node transition matrix in the product state form
        """
        in_str = "".join(self._node_sym_in(n) for _,n in self.topo_order if not n.observed)
        out_str = "".join(self._node_sym_out(n) for _,n in self.topo_order if not n.observed)
        return in_str + out_str

    def make_prod_transition_fast(self):
        node_order = [node for _, node in self.topo_order if not node.observed]

        einsum_input_strs = {node: self.einsum_input_str(node) for node in node_order}
        # a collection of einsum_str -> array to use for the actual einsum
        # we build the logic only with the strings and fetch the matrices from here
        einsum_str_to_arr = {s: self.transitions[node] for node, s in einsum_input_strs.items()}
        input_strs = [einsum_input_strs[node] for node in node_order]
        input_tensors = [einsum_str_to_arr[s] for s in input_strs]
        output_str = self.einsum_full_prod_str_fast()
        einsum_str = ",".join(input_strs) + "->" + output_str
        ret = self.square_matrix(np.einsum(einsum_str, *input_tensors))
        return ret

    def einsum_full_prod_str(self):
        """
        Constructs the einsum string for the target node transition matrix in the product state form
        """
        in_str = "".join(self._node_sym_in(n) for _,n in self.topo_order)
        out_str = "".join(self._node_sym_out(n) for _,n in self.topo_order)
        return in_str + out_str

    #@lru_cache(100)  # noqa: B019
    def make_prod_transition(self):
        node_order = [node for _,node in self.topo_order]

        einsum_input_strs = {node: self.einsum_input_str(node) for node in node_order}
        # a collection of einsum_str -> array to use for the actual einsum
        # we build the logic only with the strings and fetch the matrices from here
        einsum_str_to_arr = {s: self.transitions[node] for node, s in einsum_input_strs.items()}
        input_strs = [einsum_input_strs[node] for node in node_order]
        input_tensors = [einsum_str_to_arr[s] for s in input_strs]
        output_str = self.einsum_full_prod_str()
        einsum_str = ",".join(input_strs) + "->" + output_str
        ret = self.square_matrix(np.einsum(einsum_str, *input_tensors))
        return ret

    def fwd_product_state(self, prod_transition):
        state = self.product_state(self.top_node)
        proba : np.ndarray = prod_transition[state]

        random_values = self.rng.random(state.shape)
        proba.cumsum(axis=-1, out=proba)
        proba /= proba[..., -1:]

        next_state = (random_values[:,None] < proba).argmax(axis=1)
        return self.individual_states(next_state, self.topo_order)

    def fwd_via_matmul(self, prod_transition):
        state = self.one_hot_product_state()
        next_state = state @ prod_transition
        # sampling
        samples = torch.multinomial(torch.tensor(next_state), 1).numpy()[:,0]
        return self.individual_states(samples, self.topo_order)

    #@lru_cache(100)  # noqa: B019
    def get_p_emission(self):
        # p_emission is a projection matrix from the product state
        # state @ p_emission = state_X
        idx = self.indexs[self.top_node]
        shape = [1] * (len(self.topo_order) + 1)
        dim = self.top_node.state_dim
        shape[idx] = dim
        shape[-1] = dim
        p_emission = np.eye(dim).reshape(shape)
        for i, (_, node) in enumerate(self.topo_order):
            shape[i] = node.state_dim
        p_emission = np.broadcast_to(p_emission, shape)
        p_emission = p_emission.reshape(-1, dim)
        return p_emission

    def get_p_emission_fast(self):
        # p_emission is a projection matrix from the product state
        # state @ p_emission = state_X
        # p_emission signature = ACDX->ABCDX
        idx = self.indexs[self.top_node]
        reshape = [1] * (len(self.topo_order))
        broadshape = [1] * (len(self.topo_order))
        tgt_dim = self.top_node.state_dim
        reshape[-1] = tgt_dim
        broadshape[-1] = tgt_dim

        for i, (_, node) in enumerate(self.topo_order):
            broadshape[i] = node.state_dim
            if node in self.top_node.parents:
              reshape[i] = node.state_dim
        
        # print(reshape, broadshape, self.transitions[self.top_node][0].shape)
        p_emission = self.transitions[self.top_node][0].reshape(reshape)
        p_emission = np.broadcast_to(p_emission, broadshape)
        p_emission = p_emission.reshape(-1, tgt_dim)
        return p_emission

    @staticmethod
    def forward_algorithm(
        obs: torch.tensor,
        log_T: torch.tensor,
        log_E: torch.tensor,
        log_pi: torch.tensor,
        device="cuda",
        small_mem=False,
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

        forward_probs = torch.zeros((num_states, T, B), device=device, dtype=torch.float32)

        forward_probs[:, 0, :] = log_pi[:, None]# + log_E[:, obs[0]] <- don't need this because we start the emission with the prior
        #forward_probs[:,t,:] = p(Z_t, X_[t]=x_[t])
        
        if small_mem:
          import tqdm
          for t in tqdm.trange(1, T):
            for b in range(B):
              big_matrix = forward_probs[:, None, t - 1, b] + log_T
              forward_probs[:, t, b] = log_E[:, obs[t,b]] + torch.logsumexp(
                  big_matrix, dim=0
              )
        else:
          log_T = log_T[:, :, None]
          for t in range(1, T):
            big_matrix = forward_probs[:, None, t - 1, :] + log_T
            forward_probs[:, t, :] = log_E[:, obs[t]] + torch.logsumexp(
                big_matrix, dim=0
            )

        log_p_seq = torch.logsumexp(forward_probs, dim=0)

        return forward_probs, log_p_seq

    def forward_probs(self, observations: np.ndarray, device: str = "cuda", small_mem=False):
        T, B = observations.shape
        observations = torch.as_tensor(observations, dtype=torch.int32, device=device)
        self._init_all_nodes(B)
        prior = torch.tensor(self.one_hot_product_state()[0], device=device, dtype=torch.float32)
        transition = torch.tensor(self.make_prod_transition(), device=device, dtype=torch.float32)
        emission = torch.tensor(self.get_p_emission(), device=device, dtype=torch.float32)
        # print("from fwd probs", transition.shape, emission.shape, prior.shape)
        return self.forward_algorithm(observations, transition.log(), emission.log(), prior.log(), device=device, small_mem=small_mem)

    def forward_probs_fast(self, observations: np.ndarray, device: str = "cuda", small_mem=False):
        T, B = observations.shape
        observations = torch.as_tensor(observations, dtype=torch.int32, device=device)
        self._init_all_nodes(B)
        prior = torch.tensor(self.one_hot_product_state_fast()[0], device=device, dtype=torch.float32)
        transition = torch.tensor(self.make_prod_transition_fast(), device=device, dtype=torch.float32)
        emission = torch.tensor(self.get_p_emission_fast(), device=device, dtype=torch.float32)
        # print("from fwd probs fast", transition.shape, emission.shape, prior.shape)
        return self.forward_algorithm(observations, transition.log(), emission.log(), prior.log(), device=device, small_mem=small_mem)

    # def entropy_of_observations(self, observations: np.ndarray, device: str = "cuda"):
    #     _, log_xst = self.forward_probs(observations, device=device)
    #     H_t = -log_xst
    #     H_T = H_t[-1]
    #     return H_T


    def entropy_of_observations(self, observations: np.ndarray, final_entry_only : bool = True, device: str = "cuda", small_mem=False, fast=False, N2 = 300, verbose = False):
        """ 
            Computes the negloglikelihoods of the observations
            The function should be called as a method of an HMM instance that has the same configuration and random seed as the one used to generate the observations
            (this is important in the ICL case, where the fixed parts of the graph and distributions must be the same as for the generating HMM)
            The function automatically detects whether we are in the ICL or non-ICL case
            Args:
                observations : A set of observed sequences [seq_len, B]
                final_entry_only : A boolean. If true, returns the negloglikelihoods of the sequences X_[T], if false, the sequence of intermediate negloglikelihoods P(X_[1]), P(X_[2]), ...
            Returns:
            H_T: the estimated entropies [B] or [T,B] if final_entry_only == False
        """
        # Test whether we are in the ICL case
        if len([node for _, node in self.topo_order if node.mode == "context"]) != 0:
            if verbose:
                print("ICL detected")
            log_xst = self.forward_probs_ICL(observations, N2 = N2, verbose = verbose)
        else:
            if fast:
              _, log_xst = self.forward_probs_fast(observations, device=device, small_mem=small_mem)
            else:
              _, log_xst = self.forward_probs(observations, device=device, small_mem=small_mem)
        H_t = -log_xst.cpu()
        if final_entry_only:
            return H_t[-1]
        else:
            return H_t

    def forward_probs_ICL(self, observations : np.ndarray, N2: int = 300, verbose = False) -> np.ndarray:
        # Second version
        # NOTE: probably pretty costly in terms of memory and compute
        # TODO: too much back and forth between lists, ndarrays and torch tensors
        """ 
            loglikelihood estimations in the In Context Learning case
            The loglikelihood of the observations (whose mean is an approximation of the entropy of the sequences of observables) is approximated as follows:
            for each sequence of observations X_[t], N2 Hmms are randomly generated (each corresponding to a transition and an emission matrices).
            Those share the same fixed components as the self HMM instance, while the changing parts are independently generated
            Then log(p(X_[t])) ~= log(1/N2 * sum_hmm p(X_[t]|hmm))
            In line with the convention in the function forward_probs, the output is an array whose entries are the loglikelihoods of the increasing observed sequences

            Args:
                observations : A set of observed sequences [seq_len, B]
            Returns:
                log_xst: the estimated loglikelihoods log(P(X_[l]) of the increasing sequences X_[l] (for l=1,...,T) [T,B]
        """
        # WARNING: here we use distinct HMMs for each sequence of observables.
        # It probably results in better convergence, but I think it keeps us from being cuda-compatible
        T, B = observations.shape
        # a list of the loglikelihoods log(p(X_[l])) of the T increasing subsequences X_[l] of observations of the B complete sequences
        logliks_of_the_observations = []
        random_hmm = HMM(config = self.graph_config, random_seed = self.random_seed)
        for i in range(B):
            if verbose and i % 10 == 0:
                print(f"Estimaging loglikelihood of sequence {i}")
            # an array of length N2 of the loglikelihood of observation observations[:,i] conditioned by the N2 random hmms sampled
            conditional_logliks_of_the_observation = []
            for j in range(N2):
                random_hmm.rng.bit_generator.state = np.random.default_rng(self.random_seed + 10 + i*N2 + j).bit_generator.state
                random_hmm.top_node.initialize(B)
                random_hmm.transitions = {node: random_hmm._format_transition(node) for _, node in random_hmm.topo_order}
                # if verbose and  i % 10 == 0 and j % 100 == 0:
                #     print(f"Generating HMM number {j}")
                    # for node in [random_hmm.top_node] + random_hmm.top_node.parents:
                    #     print(f"Node {node.name}, mode {node.mode}, time {node.time}")
                    #     print(node.kernels)

                
                # conditional_log_xst should be of shape [T,1]
                _, conditional_log_xst = random_hmm.forward_probs(observations[:,[i]])
                # print(f"conditional_log_xst {conditional_log_xst}")
                conditional_log_xst = conditional_log_xst[:,0].cpu().numpy() # Now conditional_log_xst ~= [log(P(X_[1] | hmm)),...,log(P(X_[T]| hmm))]
                # [N2, T]
                conditional_logliks_of_the_observation.append(conditional_log_xst)
                # negloglik_of_single_observation is a scalar
            # estimate the likelihood of the sequence of the increasing subsequences of observations observations[:l,i] as the mean of the conditional likelihood P(observations[:l,i] | hmm) over the N2 hmms
            logliks_of_the_observations.append(np.logaddexp.reduce(np.array(conditional_logliks_of_the_observation), axis = 0) - np.log(N2)) 
        #     print(f"conditional_logliks_of_the_observation {np.array(conditional_logliks_of_the_observation)}")
        #     print(f"shape of conditional_logliks_of_the_observation {np.array(conditional_logliks_of_the_observation).shape}")
        # print(f"logliks_of_the_observations {np.array(logliks_of_the_observations).T}")
        # print(f"logliks_of_the_observations {np.array(logliks_of_the_observations).T.shape}")
        # [T,B]
        return np.array(logliks_of_the_observations).T
        
# %%


if __name__ == "__main__":
    gssm_config = {
        "nodes": [
            {
                "name": "Z1",
                "state_dim": 49,
                "parents": [],
                "alpha": 1e-2,
                "mode": "default",
                "kernel_type": "fullrank",
                "observed": False,
            },
            {
                "name": "Z2",
                "state_dim": 49,
                "parents": [],
                "alpha": 1e-2,
                "mode": "default",
                "kernel_type": "fullrank",
                "observed": False,
            },
            {
                "name": "X",
                "state_dim": 128,
                "parents": ["Z1","Z2"],
                "alpha": 2e-3,
                "mode": "default",
                "kernel_type": "fullrank",
                "observed": True,
            },
        ]
    }

    entropys = []
    for _ in range(10):
      hmm = HMM(gssm_config, random_seed=np.random.randint(93492))

      seq_len = 30

      def make_data(hmm: HMM, batch_size):
          hmm._init_all_nodes(batch_size)
          observations = np.zeros((seq_len, batch_size), dtype=int)
          for i in range(seq_len):
              observations[i] = np.array(hmm.top_node.state)
              hmm.evolve_classic(1)
          return observations

      data = make_data(hmm, 100)
      entropy = hmm.entropy_of_observations(data, small_mem=False, fast=True, final_entry_only=True).mean().item() / (seq_len - 1)
      entropys.append(entropy)
    print(f"{np.mean(entropys):.3f} ± {np.std(entropys):.3f}")

# %%
