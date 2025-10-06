import abc
from collections.abc import Iterable

import pendulum

from qoc.time_utils._datetime import DateTimeLike
from qoc.time_utils._interval import IntervalLike


class Clock(abc.ABC):
    @abc.abstractmethod
    def now(self) -> pendulum.DateTime:
        raise NotImplementedError

    @abc.abstractmethod
    def loop(
        self,
        interval: IntervalLike,
        *,
        end: DateTimeLike | None = None,
        max_duration: pendulum.Duration | None = None,
        max_iter: int | None = None,
        start: DateTimeLike | None = None,
    ) -> Iterable[pendulum.DateTime]:
        raise NotImplementedError
