"""
Abstract class for context managers monitoring training runs.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import TracebackType


@dataclass
class MonitorConfig:
    period: int = 1


class Monitor(ABC):
    def __init__(self, config: MonitorConfig):
        self.period = config.period
        self.step = 0

    @abstractmethod
    def __enter__(self) -> "Monitor":
        """Function called when entering context."""
        pass

    def __call__(self) -> None:
        """Call update function periodically."""
        self.step += 1
        if self.period <= 0:
            return
        if self.step % self.period == 0:
            self.update()

    @abstractmethod
    def update(self) -> None:
        """Main function ran by the Manager."""
        pass

    @abstractmethod
    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """Function called when exiting context"""
        pass
