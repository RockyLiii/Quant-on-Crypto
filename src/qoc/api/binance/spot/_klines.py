from collections.abc import Generator, Hashable
from typing import Any

import attrs
import cachetools
import cachetools.keys
import pendulum
import polars as pl
from loguru import logger

import qoc.time_utils as tu
from qoc.api.typing import Interval, Symbol
from qoc.time_utils import DateTimeLike

from ._base import ApiBinanceSpotBase

API_LIMIT_MAX: int = 1000


@attrs.frozen
class CacheKey:
    symbol: Symbol
    interval: Interval
    start_time: pendulum.DateTime
    end_time: pendulum.DateTime


@attrs.define
class KlinesMixin(ApiBinanceSpotBase):
    _cache_klines: cachetools.Cache[Any, pl.DataFrame] = attrs.field(
        factory=lambda: cachetools.LRUCache(maxsize=1024)
    )

    def klines(
        self,
        symbol: str,
        interval: Interval,
        *,
        startTime: DateTimeLike | None = None,
        endTime: DateTimeLike | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        endTime = tu.as_datetime(endTime) if endTime is not None else pendulum.now()
        interval_duration: pendulum.Duration = tu.as_duration(interval)
        startTime = (
            tu.as_datetime(startTime)
            if startTime is not None
            else endTime - API_LIMIT_MAX * interval_duration
        )
        data: pl.DataFrame = pl.concat(
            self._klines(
                symbol,
                interval,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                **kwargs,
            )
            for chunk in _generate_kline_chunks(symbol, interval, startTime, endTime)
        )
        # data = data.filter(
        #     (pl.col("open_time") >= startTime) & (pl.col("open_time") <= endTime)
        # )
        return data

    def _klines(
        self,
        symbol: str,
        interval: Interval,
        *,
        start_time: pendulum.DateTime,
        end_time: pendulum.DateTime,
        **kwargs,
    ) -> pl.DataFrame:
        kwargs["limit"] = API_LIMIT_MAX
        key: Hashable = cachetools.keys.methodkey(
            self, symbol, interval, start_time, end_time, **kwargs
        )
        if key in self._cache_klines:
            logger.debug(
                "klines() cache hit: {}, {}, {} - {}",
                symbol,
                interval,
                start_time,
                end_time,
            )
            return self._cache_klines[key]
        logger.debug(
            "klines() cache miss: {}, {}, {} - {}",
            symbol,
            interval,
            start_time,
            end_time,
        )
        kwargs["startTime"] = self.time_unit.as_int_timestamp(start_time)
        kwargs["endTime"] = self.time_unit.as_int_timestamp(end_time)
        raw: list[list[Any]] = self.client.klines(symbol, interval, **kwargs)
        data: pl.DataFrame = pl.from_records(
            raw,
            [
                (
                    "open_time",
                    pl.Datetime(self.time_unit.polars_time_unit, pendulum.UTC),
                ),
                ("open", pl.Float64),
                ("high", pl.Float64),
                ("low", pl.Float64),
                ("close", pl.Float64),
                ("volume", pl.Float64),
                (
                    "close_time",
                    pl.Datetime(self.time_unit.polars_time_unit, pendulum.UTC),
                ),
                ("quote_asset_volume", pl.Float64),
                ("number_of_trades", pl.Int64),
                ("taker_buy_base_asset_volume", pl.Float64),
                ("taker_buy_quote_asset_volume", pl.Float64),
                ("ignore", pl.String),
            ],
            orient="row",
        )
        if data.height == API_LIMIT_MAX:
            logger.debug(
                "klines() fetched data height == API_LIMIT_MAX: {}, {}, {} - {}",
                symbol,
                interval,
                start_time,
                end_time,
            )
            self._cache_klines[key] = data
        else:
            logger.debug(
                "klines() fetched data height < API_LIMIT_MAX: {}, {}, {} - {}",
                symbol,
                interval,
                start_time,
                end_time,
            )
        return data


def _generate_kline_chunks(
    symbol: str,
    interval: Interval,
    start_time: pendulum.DateTime,
    end_time: pendulum.DateTime,
) -> Generator[CacheKey]:
    interval_duration: pendulum.Duration = tu.as_duration(interval)
    chunk_start: pendulum.DateTime = tu.datetime_floor(start_time, interval)
    while chunk_start < end_time:
        chunk_end: pendulum.DateTime = chunk_start + API_LIMIT_MAX * interval_duration
        yield CacheKey(symbol, interval, chunk_start, chunk_end)
        chunk_start = chunk_end
