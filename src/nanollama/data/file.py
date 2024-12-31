"""
Dataloader from hdf5 file

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import h5py
import numpy as np
from numpy.random import Generator, default_rng


def get_batch(nb_epochs: int, batch_size: int, path: str, n_data: int, rng: Generator = None):
    if rng is None:
        rng = default_rng()

    batch_idx = np.array([], dtype=int)
    for _ in range(nb_epochs):
        epoch_idx = rng.permutation(n_data)
        # add residual from the previous epoch
        epoch_idx = np.append(batch_idx, epoch_idx)

        step = 0
        while True:
            begin, end = step * batch_size, (step + 1) * batch_size
            batch_idx = epoch_idx[begin:end]
            batch_idx.sort()

            # remove potential duplicate between last batch of previous epoch and first batch of the new one
            if begin == 0:
                batch_idx = np.unique(batch_idx)

            with h5py.File(path, "r") as f:
                batch = f["data"][batch_idx]  # Load rows 2000 to 2999
                step += 1
                yield batch

            if batch.shape[0] < batch_size:
                break
