from typing import Union

import numpy as np
from numpy.random import Generator, default_rng
from scipy.stats import dirichlet


class TransitionKernel:
    """
    A transition kernel that generates the next state given the current state.

    Parameters
    ----------
    fan_in:
        Number of input states.
    fan_out:
        Number of output states.
    generator:
        Random number generator.
    alphas:
        Dirichlet prior parameter. Can be a float, int, list of floats, or a numpy array of shape (fan_out,).

    Attributes
    ----------
    p_transition:
        A fan_in x fan_out transition matrix.
    """

    def __init__(
        self, fan_in: int, fan_out: int, alphas: Union[int, float, list[float], np.ndarray], generator: Generator = None
    ):

        # handle various types
        if isinstance(alphas, float) or isinstance(alphas, int):
            alphas = np.full(fan_out, alphas)
        if isinstance(alphas, list):
            alphas = np.array(alphas)
        assert alphas.shape == (
            fan_out,
        ), "Alphas must be a float, int, list of floats, or a numpy array of shape (fan_out,)."

        # set random number generator
        if generator is None:
            generator = default_rng()
        self.generator = generator

        self.fan_out = fan_out
        self.p_transition = dirichlet.rvs(alphas, size=fan_in, random_state=generator)
        self._cumulative = np.cumsum(self.p_transition, axis=1)

    def __call__(self, state: Union[int, list[int], np.ndarray]) -> Union[int, np.ndarray]:
        """
        Generate the next state given the current state.

        Parameters
        ----------
        state:
            Current state. Can be an integer, list of integers, or a numpy array of integers.

        Returns
        -------
        next_state
            Next state. If state is a list, it will return a numpy array.
        """
        if isinstance(state, int):
            return self.generator.choice(self.fan_out, p=self.p_transition[state])

        # Convert state to a numpy array if it's a list
        if isinstance(state, list):
            state = np.asarray(state)

        # Vectorized sampling
        random_values = self.generator.random(state.shape)
        p_cumulative = self._cumulative[state]
        return (random_values[:, None] < p_cumulative).argmax(axis=1)


def hmm_example():
    # Initialize parameters
    vocab_size = 256
    state_size = 16
    bsz = 10
    seq_len = 20
    alpha_z = 1e3
    generator = np.random.default_rng()
    alpha_x = 1e-2

    # Define the transition and emission probabilities using the TransitionKernel
    init_kernel = TransitionKernel(fan_in=1, fan_out=state_size, alphas=alpha_z, generator=generator)
    transition_kernel = TransitionKernel(fan_in=state_size, fan_out=state_size, alphas=alpha_z, generator=generator)

    emission_kernel = TransitionKernel(fan_in=state_size, fan_out=vocab_size, alphas=alpha_x, generator=generator)

    # Generate the sequence
    observations = np.zeros((bsz, seq_len), dtype=int)
    states = np.zeros((bsz, seq_len), dtype=int)
    states[:, 0] = init_kernel(np.zeros(bsz, dtype=int))
    observations[:, 0] = emission_kernel(states[:, 0])

    for t in range(seq_len - 1):
        states[:, t + 1] = transition_kernel(states[:, t])
        observations[:, t + 1] = emission_kernel(states[:, t + 1])

    # Print the generated states and observations
    print("Generated States:\n", states)
    print("Generated Observations:\n", observations)
