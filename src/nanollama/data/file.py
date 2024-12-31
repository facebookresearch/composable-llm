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
        # add residual data from the previous epoch
        epoch_idx = np.append(batch_idx, epoch_idx)

        step = 0
        while True:
            begin, end = step * batch_size, (step + 1) * batch_size
            batch_idx = epoch_idx[begin:end]

            # do not process last batch alone if it is smaller than batch_size
            if batch_idx.shape[0] < batch_size:
                break

            # sort batch_idx to optimize I/O
            batch_idx.sort()

            # remove potential duplicate when inserting residual batch
            if begin == 0:
                batch_idx, duplicate = np.unique(batch_idx, return_counts=True)
            else:
                duplicate = None

            with h5py.File(path, "r") as f:
                batch = f["data"][batch_idx]
                step += 1

            # handle duplicate
            if duplicate and (duplicate != 1).any():
                ind = np.repeat(np.arange(len(duplicate)), duplicate)
                batch = batch[ind]

            yield batch
