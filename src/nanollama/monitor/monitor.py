"""
Abstract class for context managers monitoring training runs.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from dataclasses import dataclass
from logging import getLogger
from types import TracebackType

logger = getLogger(__name__)


@dataclass
class MonitorConfig:
    period: int = 1


class Monitor:
    def __init__(self, config: MonitorConfig):
        self.period = config.period
        self.step = 0

    def __enter__(self) -> "Monitor":
        """Function called when entering context."""
        return self

    def __call__(self) -> None:
        """Call update function periodically."""
        self.step += 1
        if self.step % self.period == 0:
            self.update()
            self.step = 0

    def update(self) -> None:
        """Main function ran by the Manager."""
        pass

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """Function called when exiting context"""
        pass
