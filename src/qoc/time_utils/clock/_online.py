import time
from typing import override

import attrs
import pendulum
from loguru import logger
from pendulum import DateTime

from qoc.time_utils._datetime import DateTimeLike, as_datetime
from qoc.time_utils._interval import Interval, IntervalLike

from ._abc import Clock
from ._utils import get_end


@attrs.define
class ClockOnline(Clock):
    _interval: Interval
    _now: DateTime
    _end: DateTime
    _step: int = 0

    def __init__(
        self,
        interval: IntervalLike,
        *,
        end: DateTimeLike | None = None,
        max_duration: pendulum.Duration | None = None,
        max_iter: int | None = None,
        start: DateTimeLike | None = None,
    ) -> None:
        interval = Interval.parse(interval)
        start = as_datetime(start) if start is not None else pendulum.now(pendulum.UTC)
        end = get_end(
            interval=interval,
            end=end,
            max_duration=max_duration,
            max_iter=max_iter,
            start=start,
        )
        self.__attrs_init__(interval=interval, now=start, start=start, end=end)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    @override
    def now(self) -> pendulum.DateTime:
        return pendulum.now(pendulum.UTC)

    @property
    @override
    def step(self) -> int:
        return self._step

    @override
    def __next__(self) -> DateTime:
        tick: DateTime = self._now + self._interval.duration
        now: DateTime = pendulum.now(pendulum.UTC)
        if tick < now:
            logger.warning("clock tick is behind schedule: {} < {}", tick, now)
        else:
            duration: pendulum.Interval = tick - now
            time.sleep(duration.total_seconds())
        self._now = tick
        self._step += 1
        if self._end is not None and self._now > self._end:
            raise StopIteration
        logger.debug("clock tick {}: {}", self._step, tick)
        return tick
