import functools
from typing import override

import pendulum
import polars as pl

import qoc.time_utils as tu
from qoc.api._abc import AbstractApi
from qoc.api.typing import Interval, Symbol
from qoc.time_utils import DateTimeLike

from ._klines import KLines


class ApiOfflineSpot(AbstractApi):
    # TODO(liblaf): implement other endpoints

    @override
    def klines(
        self,
        symbol: Symbol,
        interval: Interval,
        *,
        startTime: DateTimeLike | None = None,
        endTime: DateTimeLike | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        interval_parsed: tu.Interval = tu.Interval.parse(interval)
        end: pendulum.DateTime = (
            tu.now() if endTime is None else tu.as_datetime(endTime)
        )
        start: pendulum.DateTime
        if startTime is None:
            start = end - 500 * interval_parsed.duration
        else:
            start = tu.as_datetime(startTime)
        return self._klines(symbol, interval, startTime=start, endTime=end, **kwargs)

    @functools.cached_property
    def _klines(self) -> KLines:
        return KLines()
