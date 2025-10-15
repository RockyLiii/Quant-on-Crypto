from typing import override

import attrs
from loguru import logger
from pendulum import DateTime, Duration

from qoc.time_utils._datetime import DateTimeLike, as_datetime
from qoc.time_utils._interval import Interval, IntervalLike

from ._abc import Clock
from ._utils import get_end


@attrs.define
class ClockOffline(Clock):
    _interval: Interval
    _now: DateTime
    _end: DateTime
    _step: int = 0

    def __init__(
        self,
        interval: IntervalLike,
        *,
        end: DateTimeLike | None = None,
        max_duration: Duration | None = None,
        max_iter: int | None = None,
        start: DateTimeLike,
    ) -> None:
        interval = Interval.parse(interval)
        start = as_datetime(start)
        end: DateTime | None = get_end(
            interval=interval,
            end=end,
            max_duration=max_duration,
            max_iter=max_iter,
            start=start,
        )
        self.__attrs_init__(interval=interval, now=start, end=end)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    @override
    def now(self) -> DateTime:
        return self._now

    @property
    @override
    def step(self) -> int:
        return self._step

    @override
    def __next__(self) -> DateTime:
        self._now = self._now + self._interval.duration
        self._step += 1
        if self._end is not None and self._now > self._end:
            raise StopIteration
        # logger.debug("clock tick {}: {}", self._step, self._now)
        return self._now
