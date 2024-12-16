from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass
class DataConfig:
    seq_len: int = -1
    batch_size: int = -1
    seed: int = 42


@dataclass
class DataLoaderState:
    rng_state: dict[str, Any]


def init_dataloader_state(seed: int) -> DataLoaderState:
    rng = np.random.default_rng(seed)
    return DataLoaderState(rng_state=rng.bit_generator.state)


def get_sample_iterator(seq_len: int, rng_state: dict[str, Any]) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    """
    Generate one sentence.

    The sentence is composed of a sequence of random bits followed by a constant value, followed by the same sequence.

    Parameters
    ----------
    seq_len:
        The length of the sequence to generate.
    rng_state:
        The state of the random number generator.

    Yields
    ------
    tuple[np.ndarray, dict[str, Any]]
        The generated sentence and the state of the random number generator.
    """
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    data = np.empty((seq_len,), dtype=int)
    length = (seq_len - 2) // 2
    while True:
        # Simple linear regression
        data[:length] = rng.random((length,)) > 0
        data[length] = 2
        data[length + 1 : -1] = data[:length]
        data[-1] = 3
        yield data, rng.bit_generator.state


def get_batch_iterator(
    sample_iterator: Iterator[tuple[np.ndarray, dict[str, Any]]], batch_size: int, seq_len: int, state: DataLoaderState
) -> Iterator[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """
    Generate batches of sentences.

    Parameters
    ----------
    sample_iterator:
        The iterator that generates sentences.
    batch_size:
        The size of the batch.
    seq_len:
        The length of the sequence to generate.
    state:
        The state of the random number generator.

    Yields
    ------
    tuple[np.ndarray, np.ndarray, dict[str, Any]]
        The generated batch of sentences, the target, and the state of the random number generator.
    """
    batch = np.empty((batch_size, seq_len), dtype=int)
    i = 0
    for data, rng_state in sample_iterator:
        batch[i] = data
        i += 1
        if i == batch_size:
            state.rng_state = rng_state
            yield batch, state
            i = 0


@contextmanager
def data_loader_context(config: DataConfig, state: DataLoaderState) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    """
    Context manager for the data loader.

    Parameters
    ----------
    config:
        The configuration of the data loader.
    state:
        The state of the data loader.

    Yields
    ------
    tuple[np.ndarray, dict[str, Any]]
        The generated batch of sentences and the state of the random number generator.
    """
    sample_iterator = get_sample_iterator(config.seq_len, state.rng_state)
    try:
        yield get_batch_iterator(
            sample_iterator=sample_iterator,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            state=state,
        )
    # clearning when exiting context
    finally:
        pass
