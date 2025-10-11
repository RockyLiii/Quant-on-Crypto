import abc
from collections.abc import Iterator

from pendulum import DateTime


class Clock(Iterator[DateTime]):
    @property
    @abc.abstractmethod
    def now(self) -> DateTime:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def step(self) -> int:
        raise NotImplementedError
