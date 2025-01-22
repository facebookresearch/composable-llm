# %%
from functools import lru_cache
from logging import getLogger
import numpy as np
import torch

# from omegaconf import OmegaConf
from nanollama.utils import initialize_nested_object
from nanollama.data import gssm
# %%

logger = getLogger("nanollama")


class HMM:
    SYM_IN = "abcxdefghijklmnopqrstuvwxyz"
    SYM_OUT = "ABCXDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, config=None, top_node=None, random_seed=100):
        """
        makes an HMM from a graph config via the product state
        """
        if config is not None:
            assert top_node is None
            config = initialize_nested_object(gssm.GSSMConfig, config, inplace=False)
            self.rng = np.random.default_rng(random_seed)
            self.top_node = gssm.build_gssm(config, self.rng)
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
    def _join_parent_kernels(node):
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
    def _format_transition(node):
        logger.info(f"Rewriting transition matrix for {node}")
        parents = node.parents
        parent_state_dims = tuple([p.state_dim for p in parents])
        trans = HMM._join_parent_kernels(node)
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

    def one_hot_product_state(self, node):
        node_order = self._dfs(node)
        einsum_str = "B" + ",B".join(HMM.SYM_IN[: len(node_order)]) + "->B" + HMM.SYM_IN[: len(node_order)]
        product_state = np.einsum(einsum_str, *[self._one_hot_state(node) for _, node in node_order])
        return product_state.reshape(product_state.shape[0], -1)

    def product_state(self, tgt_node):
        return self.one_hot_product_state(tgt_node).argmax(-1)

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

    def einsum_full_prod_str(self, tgt_node):
        """
        Constructs the einsum string for the target node transition matrix in the product state form
        """
        ordered_nodes = [node for _, node in self._dfs(tgt_node)]
        in_str = "".join(self._node_sym_in(n) for n in ordered_nodes)
        out_str = "".join(self._node_sym_out(n) for n in ordered_nodes)
        return in_str + out_str

    @lru_cache(100)  # noqa: B019
    def make_prod_transition(self, tgt_node):
        node_order = [node for _, node in self._dfs(tgt_node)]

        einsum_input_strs = {node: self.einsum_input_str(node) for node in node_order}
        # a collection of einsum_str -> array to use for the actual einsum
        # we build the logic only with the strings and fetch the matrices from here
        einsum_str_to_arr = {s: self.transitions[node] for node, s in einsum_input_strs.items()}
        input_strs = [einsum_input_strs[node] for node in node_order]
        input_tensors = [einsum_str_to_arr[s] for s in input_strs]
        output_str = self.einsum_full_prod_str(tgt_node)
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
        state = self.one_hot_product_state(self.top_node)
        next_state = state @ prod_transition
        # sampling
        samples = torch.multinomial(torch.tensor(next_state), 1).numpy()[:,0]
        return self.individual_states(samples, self.topo_order)

    @lru_cache(100)  # noqa: B019
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
        for i, (_, node) in enumerate(self.topo_order):
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
        log_T = log_T.to(device).to(torch.float64)
        log_E = log_E.to(device).to(torch.float64)
        log_pi = log_pi.to(device).to(torch.float64)
        obs = obs.to(device)


        forward_probs = torch.zeros((num_states, T, B), device=device)

        forward_probs[:, 0, :] = log_pi[:, None] + log_E[:, obs[0]]
        #forward_probs[:,t,:] = p(Z_t, X_[t])
        

        log_T = log_T[:, :, None]

        for t in range(1, T):
            forward_probs[:, t, :] = log_E[:, obs[t]] + torch.logsumexp(
                forward_probs[:, None, t - 1, :] + log_T, dim=0
            )

        log_p_seq = torch.logsumexp(forward_probs, dim=0)

        return forward_probs.cpu(), log_p_seq.cpu()

    def forward_probs(self, observations: np.ndarray, device: str = "cuda"):
        T, B = observations.shape
        observations = torch.as_tensor(observations)
        self._init_all_nodes(B)
        prior = torch.tensor(self.current_one_hot_product_state()[0])
        transition = torch.tensor(self.make_prod_transition(self.top_node))
        emission = torch.tensor(self.get_p_emission(self.top_node))
        return self.forward_algorithm(observations, transition.log(), emission.log(), prior.log(), device=device)

    # def entropy_of_observations(self, observations: np.ndarray, device: str = "cuda"):
    #     _, log_xst = self.forward_probs(observations, device=device)
    #     H_t = -log_xst
    #     H_T = H_t[-1]
    #     return H_T

    def entropy_of_observations(self, observations: np.ndarray, final_entry_only : bool = True, device: str = "cuda"):
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
            log_xst = self.forward_probs_ICL(observations)
        else:
            _, log_xst = self.forward_probs(observations, device=device)
        H_t = -log_xst
        if final_entry_only:
            return H_t[-1]
        else:
            return H_t

    def forward_probs_ICL(self, observations : np.ndarray, N2: int = 500) -> np.ndarray:
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
        # [T,B]
        logliks_of_the_observations = []
        random_hmm = HMM(config = self.graph_config, random_seed = self.random_seed)
        for i in range(B):
            # a list of length N2 of the loglikelihood of observation observations[:,i] conditioned by the N2 random hmms sampled
            conditional_logliks_of_the_observation = []
            for j in range(N2):
                random_hmm.np.random.default_rng(self.random_seed + 10 + i*N2 + j)
                random_hmm.top_node.initialize()
                random_hmm.transitions = {node: random_hmm._format_transition(node) for _, node in random_hmm.topo_order}
                # conditional_log_xst should be of shape [T,1]
                _, conditional_log_xst = random_hmm.forward_probs(observations[:,[i]])
                conditional_log_xst = conditional_log_xst[:,0].item() # Now conditional_log_xst ~= [log(P(X_[1] | hmm)),...,log(P(X_[T]| hmm))]
                # [N2, T]
                conditional_logliks_of_the_observation.append(conditional_log_xst)
                # negloglik_of_single_observation is a scalar
            # estimate the likelihood of the sequence of the increasing subsequences of observations observations[:l,i] as the mean of the conditional likelihood P(observations[:l,i] | hmm) over the N2 hmms
           
            logliks_of_the_observations.append(np.logaddexp.reduce(np.array(conditional_logliks_of_the_observation), dim = 0) - np.log(N2)) 

        return np.array(logliks_of_the_observations).T
        

