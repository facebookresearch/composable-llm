from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ContextManager, Iterator

import numpy as np


@dataclass
class DataConfig:
    num_samples: int = 1000
    batch_size: int = 32
    seed: int = 42


@dataclass
class DataLoaderState:
    rng_state: dict[str, Any]


def init_dataloader_state(seed: int) -> DataLoaderState:
    rng = np.random.default_rng(seed)
    return DataLoaderState(rng_state=rng.bit_generator.state)


def get_sample_iterator(rng_state: dict[str, Any]) -> Iterator[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    beta = np.ones(10)
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    while True:
        # Simple linear regression
        X = rng.random(10)
        y = np.dot(X, beta) + rng.random()
        yield X, y, rng.bit_generator.state


def get_batch_iterator(
    sample_iterator: Iterator[tuple[np.ndarray, np.ndarray, dict[str, Any]]], batch_size: int, state: DataLoaderState
) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    X_batch = np.empty((batch_size, 10))
    y_batch = np.empty(batch_size)
    i = 0
    for X, y, rng_state in sample_iterator:
        X_batch[i] = X
        y_batch[i] = y
        i += 1
        if i == batch_size:
            state.rng_state = rng_state
            yield X_batch, y_batch, state
            i = 0


@contextmanager
def data_loader_context(
    config: DataConfig, state: DataLoaderState
) -> ContextManager[Iterator[tuple[np.ndarray, np.ndarray, dict[str, Any]]]]:
    sample_iterator = get_sample_iterator(state.rng_state)
    try:
        yield get_batch_iterator(
            sample_iterator=sample_iterator,
            batch_size=config.batch_size,
            state=state,
        )
    # clearning when exiting context
    finally:
        pass
