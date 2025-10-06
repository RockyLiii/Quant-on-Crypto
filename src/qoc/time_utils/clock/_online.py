import itertools
import time
from collections.abc import Generator
from typing import override

import pendulum
from loguru import logger

from qoc.time_utils._datetime import DateTimeLike, as_datetime
from qoc.time_utils._interval import Interval, IntervalLike

from ._abc import Clock


class ClockOnline(Clock):
    @override
    def now(self) -> pendulum.DateTime:
        return pendulum.now(pendulum.UTC)

    @override
    def loop(
        self,
        interval: IntervalLike,
        *,
        end: DateTimeLike | None = None,
        max_duration: pendulum.Duration | None = None,
        max_iter: int | None = None,
        start: DateTimeLike | None = None,
    ) -> Generator[pendulum.DateTime]:
        for tick in self._ticks(
            interval, end=end, max_duration=max_duration, max_iter=max_iter, start=start
        ):
            now: pendulum.DateTime = pendulum.now(pendulum.UTC)
            if tick < now:
                logger.warning("clock tick is behind schedule: {} < {}", tick, now)
            else:
                duration: pendulum.Interval = tick - now
                time.sleep(duration.total_seconds())
            logger.debug("clock tick: {}", tick)
            yield tick

    def _ticks(
        self,
        interval: IntervalLike,
        *,
        end: DateTimeLike | None = None,
        max_duration: pendulum.Duration | None = None,
        max_iter: int | None = None,
        start: DateTimeLike | None = None,
    ) -> Generator[pendulum.DateTime]:
        interval: Interval = Interval.parse(interval)
        start: pendulum.DateTime = (
            pendulum.now(pendulum.UTC) if start is None else as_datetime(start)
        )
        if end is not None:
            end = as_datetime(end)
        if max_duration is not None:
            if end is None:
                end = start + max_duration
            else:
                end = min(end, start + max_duration)
        current: pendulum.DateTime = start + interval.duration
        for it in itertools.count():
            yield current
            current = current + interval.duration
            if end is not None and current >= end:
                break
            if max_iter is not None and it >= max_iter:
                break
