import math
from typing import Any, Protocol

import attrs
import cachetools
import pendulum
import polars as pl
from loguru import logger
from polars._typing import SchemaDefinition

import qoc.time_utils as tu
from qoc.api import utils
from qoc.api.typing import Interval, Symbol
from qoc.time_utils import DateTimeLike


class KlinesApi(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> list[list[Any]]: ...


@attrs.frozen
class CacheKey:
    symbol: Symbol
    interval: Interval
    start_time: tu.DateTimeLike
    end_time: tu.DateTimeLike

    def __str__(self) -> str:
        return f"{self.symbol}-{self.interval}: {self.start_time} -> {self.end_time}"


@attrs.define
class KlinesCache:
    """.

    References:
        1. <https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data>
    """

    func: KlinesApi
    time_unit: tu.TimeUnit
    cache: cachetools.Cache[CacheKey, pl.DataFrame] = attrs.field(
        factory=lambda: cachetools.LRUCache(maxsize=128)
    )
    limit: int = 1000

    def __call__(
        self,
        symbol: Symbol,
        interval: Interval,
        *,
        startTime: DateTimeLike | None = None,
        endTime: DateTimeLike | None = None,
        limit: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        interval_str: str = interval
        interval: tu.Interval = tu.Interval.parse(interval_str)
        end: pendulum.DateTime = (
            tu.now() if endTime is None else tu.as_datetime(endTime)
        )
        if limit is None:
            limit = self.limit // 2
        start: pendulum.DateTime = (
            tu.as_datetime(startTime)
            if startTime is not None
            else end - limit * interval.duration
        )

        chunks: list[pl.DataFrame] = []
        start_index: int = tu.datetime_to_index_floor(start, interval)
        end_index: int = tu.datetime_to_index_ceil(end, interval)
        start_chunk_index: int = (start_index // self.limit) * self.limit
        end_chunk_index: int = math.ceil(end_index / self.limit) * self.limit
        for chunk_index in range(start_chunk_index, end_chunk_index, self.limit):
            chunk_start: pendulum.DateTime = tu.index_to_datetime(chunk_index, interval)
            chunk_end: pendulum.DateTime = tu.index_to_datetime(
                chunk_index + self.limit, interval
            )
            delta: pl.DataFrame = self._klines(
                symbol, interval_str, start=chunk_start, end=chunk_end, **kwargs
            )
            if not delta.is_empty():
                chunks.append(delta)
        data: pl.DataFrame = pl.concat(chunks)
        data = data.drop("ignore")
        data = data.filter(
            (pl.col("open_time") >= start) & (pl.col("open_time") <= end)
        )
        return data

    @property
    def schema(self) -> SchemaDefinition:
        return utils.klines_schema(
            datetime_schema=pl.Datetime(
                time_unit=self.time_unit.polars_time_unit, time_zone=pendulum.UTC
            )
        )

    def _klines(
        self,
        symbol: Symbol,
        interval: Interval,
        start: pendulum.DateTime,
        end: pendulum.DateTime,
        **kwargs,
    ) -> pl.DataFrame:
        key = CacheKey(symbol, interval, start, end)
        if key in self.cache:
            logger.trace("klines cache hit: {}", key)
            return self.cache[key]
        logger.trace("klines cache miss: {}", key)
        now: pendulum.DateTime = pendulum.now(pendulum.UTC)
        raw: list[list[Any]] = self.func(
            symbol=symbol,
            interval=interval,
            startTime=math.floor(tu.as_timestamp(start, unit=self.time_unit)),
            endTime=math.ceil(tu.as_timestamp(end, unit=self.time_unit)),
            limit=self.limit,
            **kwargs,
        )
        data: pl.DataFrame = pl.from_records(raw, schema=self.schema, orient="row")
        if end < now:
            self.cache[key] = data
        return data
