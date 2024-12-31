"""
Utils functions

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from .train import TrainState


def trigger_update(state: TrainState, period: int) -> bool:
    """
    Return a boolean indicating whether the current step is a multiple of `period`.

    Parameters
    ----------
    state : TrainState
        Current training state containing optimization and accumulation steps.
    period : int
        Number of gradient updates between each check. If -1, always return False.
    """

    if period == -1:
        return False
    return (state.optim.step % period == 0) and (state.optim.acc_step == 0)
