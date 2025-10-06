import itertools
from collections.abc import Generator
from typing import override

import attrs
import pendulum
from loguru import logger

from qoc.time_utils._datetime import DateTimeLike, as_datetime
from qoc.time_utils._interval import Interval, IntervalLike

from ._abc import Clock


@attrs.define
class ClockOffline(Clock):
    _now: pendulum.DateTime | None = None

    def now(self) -> pendulum.DateTime:
        if self._now is None:
            msg: str = f"{type(self)} has not been started yet."
            raise RuntimeError(msg)
        return self._now

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
        if start is None:
            raise ValueError("start must be provided for offline clock")  # noqa: TRY003
        for tick in self._ticks(
            interval, end=end, max_duration=max_duration, max_iter=max_iter, start=start
        ):
            self._now = tick
            logger.debug("clock tick: {}", tick)
            yield tick
        self._now = None

    def _ticks(
        self,
        interval: IntervalLike,
        *,
        end: DateTimeLike | None = None,
        max_duration: pendulum.Duration | None = None,
        max_iter: int | None = None,
        start: DateTimeLike,
    ) -> Generator[pendulum.DateTime]:
        interval: Interval = Interval.parse(interval)
        start: pendulum.DateTime = as_datetime(start)
        if end is not None:
            end = as_datetime(end)
        if max_duration is not None:
            if end is None:
                end = start + max_duration
            else:
                end = min(end, start + max_duration)
        current: pendulum.DateTime = start
        for it in itertools.count():
            yield current
            current = current + interval.duration
            if end is not None and current >= end:
                break
            if max_iter is not None and it >= max_iter:
                break
