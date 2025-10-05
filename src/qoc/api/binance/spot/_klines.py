import math
from typing import Any

import attrs
import binance.spot
import cachetools
import pendulum
import polars as pl
from loguru import logger
from polars._typing import SchemaDefinition

import qoc.time_utils as tu
from qoc.api.binance._utils import get_time_unit
from qoc.api.typing import Interval, Symbol
from qoc.time_utils import DateTimeLike


@attrs.frozen
class CacheKey:
    symbol: Symbol
    interval: Interval
    start_time: tu.DateTimeLike
    end_time: tu.DateTimeLike

    def __str__(self) -> str:
        return f"{self.symbol} ({self.interval}) {self.start_time} - {self.end_time}"


@attrs.define
class KLines:
    client: binance.spot.Spot
    cache: cachetools.Cache[CacheKey, pl.DataFrame] = attrs.field(
        factory=lambda: cachetools.LRUCache(maxsize=1024)
    )
    limit: int = 1000

    def __call__(
        self,
        symbol: Symbol,
        interval: Interval,
        startTime: DateTimeLike | None = None,
        endTime: DateTimeLike | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        interval_parsed: tu.Interval = tu.Interval.parse(interval)
        endTime = tu.now() if endTime is None else tu.as_datetime(endTime)
        if startTime is None:
            startTime = endTime - (self.limit / 2) * interval_parsed.duration
        else:
            startTime = tu.as_datetime(startTime)

        chunks: list[pl.DataFrame] = []
        start_time_index: int = tu.datetime_to_index_floor(startTime, interval)
        end_time_index: int = tu.datetime_to_index_ceil(endTime, interval)
        start_chunk_index: int = (start_time_index // self.limit) * self.limit
        end_chunk_index: int = math.ceil(end_time_index / self.limit) * self.limit
        for chunk_index in range(start_chunk_index, end_chunk_index, self.limit):
            chunk_start_time: pendulum.DateTime = tu.index_to_datetime(
                chunk_index, interval
            )
            chunk_end_time: pendulum.DateTime = tu.index_to_datetime(
                chunk_index + self.limit, interval
            )
            delta: pl.DataFrame = self._klines(
                symbol,
                interval,
                start_time=chunk_start_time,
                end_time=chunk_end_time,
                **kwargs,
            )
            if not delta.is_empty():
                chunks.append(delta)
        data: pl.DataFrame = pl.concat(chunks)
        data = data.filter(
            (pl.col("open_time") >= startTime) & (pl.col("open_time") <= endTime)
        )
        return data

    @property
    def schema(self) -> SchemaDefinition:
        datetime_schema = pl.Datetime(
            time_unit=self.time_unit.polars_time_unit, time_zone=pendulum.UTC
        )
        return [
            ("open_time", datetime_schema),
            ("open", pl.Float64),
            ("high", pl.Float64),
            ("low", pl.Float64),
            ("close", pl.Float64),
            ("volume", pl.Float64),
            ("close_time", datetime_schema),
            ("quote_asset_volume", pl.Float64),
            ("number_of_trades", pl.Int64),
            ("taker_buy_base_asset_volume", pl.Float64),
            ("taker_buy_quote_asset_volume", pl.Float64),
            ("ignore", pl.String),
        ]

    @property
    def time_unit(self) -> tu.TimeUnit:
        return get_time_unit(self.client)

    def _klines(
        self,
        symbol: Symbol,
        interval: Interval,
        start_time: pendulum.DateTime,
        end_time: pendulum.DateTime,
        **kwargs,
    ) -> pl.DataFrame:
        key = CacheKey(symbol, interval, start_time, end_time)
        if key in self.cache:
            logger.trace("klines cache hit: {}", key)
            return self.cache[key]
        logger.trace("klines cache miss: {}", key)
        raw: list[list[Any]] = self.client.klines(
            symbol=symbol,
            interval=interval,
            startTime=math.floor(tu.as_timestamp(start_time, unit=self.time_unit)),
            endTime=math.ceil(tu.as_timestamp(end_time, unit=self.time_unit)),
            limit=self.limit,
            **kwargs,
        )
        data: pl.DataFrame = pl.from_records(raw, schema=self.schema, orient="row")
        if data.is_empty():
            return data
        interval_parsed: tu.Interval = tu.Interval.parse(interval)
        if data["close_time"][-1] + interval_parsed.duration > end_time:
            self.cache[key] = data
        return data
