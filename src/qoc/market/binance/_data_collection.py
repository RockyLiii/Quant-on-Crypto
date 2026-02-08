from __future__ import annotations

import asyncio
import enum
import io
import re
import urllib.parse
import zipfile
from collections.abc import Awaitable
from pathlib import Path
from typing import Self, override

import attrs
import cachetools
import httpx
import pendulum
import platformdirs
import polars as pl
from httpx_retries import RetryTransport
from liblaf import grapes
from pendulum import Date, Interval

from qoc.time_utils import DateTimeLike

from ._abc import BinanceMarketData
from ._common import KLINES_CSV_SCHEMAS, KLINES_SCHEMAS, filter_klines, parse_start_end


class _Period(enum.StrEnum):
    DAILY = "daily", "YYYY-MM-DD"
    MONTHLY = "monthly", "YYYY-MM"

    _date_fmt: str

    def __new__(cls, value: str, fmt: str) -> Self:
        self: Self = str.__new__(cls, value)
        self._value_ = value
        self._date_fmt = fmt
        return self

    @property
    def date_fmt(self) -> str:
        return self._date_fmt


def _default_cache_dir(self: BinanceDataCollection) -> Path:
    cache_dir: Path = platformdirs.user_cache_path("quant-on-crypto")
    return cache_dir / self.base_url.removeprefix("https://")


def _default_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(5.0, pool=None),
        follow_redirects=True,
        transport=RetryTransport(),
    )


from attrs import define

@define
class BinanceDataCollection(BinanceMarketData):
    base_url: str = attrs.field(kw_only=True)
    cache_dir: Path = attrs.field(
        default=attrs.Factory(_default_cache_dir, takes_self=True), kw_only=True
    )
    client: httpx.AsyncClient = attrs.field(factory=_default_client, kw_only=True)
    _not_found_cache: cachetools.Cache[str, bool] = attrs.field(
        repr=False,
        init=False,
        factory=lambda: cachetools.TTLCache(maxsize=8192, ttl=3600),
    )

    @override
    async def klines(
        self,
        symbol: str,
        interval: str,
        start: DateTimeLike | None = None,
        end: DateTimeLike | None = None,
    ) -> pl.DataFrame:
        start, end = parse_start_end(start, end, interval)
        start_end: Interval = pendulum.interval(start, end)
        tasks: list[Awaitable[pl.DataFrame | None]] = [
            self._fetch_month(symbol, interval, start_end, date)
            for date in Interval(start.start_of("month"), end.end_of("month")).range(
                "months"
            )
        ]
        chunks: list[pl.DataFrame] = [
            chunk for chunk in await asyncio.gather(*tasks) if chunk is not None
        ]
        data: pl.DataFrame = (
            pl.concat(chunks, how="vertical")
            if chunks
            else pl.DataFrame(schema=KLINES_SCHEMAS)
        )
        data = filter_klines(data, symbol, interval, start, end)
        return data

    def _cache_file_path(
        self, period: _Period, symbol: str, interval: str, date: Date
    ) -> Path:
        filename: str = f"{symbol}-{interval}-{date.format(period.date_fmt)}.parquet"
        return self.cache_dir / period.value / "klines" / symbol / interval / filename

    async def _fetch(
        self, period: _Period, symbol: str, interval: str, date: Date
    ) -> pl.DataFrame | None:
        data: pl.DataFrame | None = self._load_from_cache(
            period, symbol, interval, date
        )
        if data is not None:
            return data
        data = await self._fetch_from_upstream(period, symbol, interval, date)
        if data is not None:
            self._save_to_cache(data, period, symbol, interval, date)
        return data

    async def _fetch_month(
        self, symbol: str, interval: str, start_end: Interval[Date], date: Date
    ) -> pl.DataFrame | None:
        data: pl.DataFrame | None = await self._fetch(
            _Period.MONTHLY, symbol, interval, date
        )
        if data is not None:
            return data
        start: Date = max(start_end.start, date.start_of("month")).start_of("day")
        end: Date = min(start_end.end, date.end_of("month")).end_of("day")
        tasks: list[Awaitable[pl.DataFrame | None]] = [
            self._fetch(_Period.DAILY, symbol, interval, date)
            for date in Interval(start, end).range("days")
        ]
        chunks: list[pl.DataFrame] = [
            chunk for chunk in await asyncio.gather(*tasks) if chunk is not None
        ]
        if not chunks:
            return None
        data = pl.concat(chunks, how="vertical")
        return data

    async def _fetch_from_upstream(
        self, period: _Period, symbol: str, interval: str, date: Date
    ) -> pl.DataFrame | None:
        filename: str = f"{symbol}-{interval}-{date.format(period.date_fmt)}"
        if self._not_found_cache.get(filename, False):
            return None
        url: str = urllib.parse.urljoin(
            self.base_url, f"{period.value}/klines/{symbol}/{interval}/{filename}.zip"
        )
        response: httpx.Response = await self.client.get(url)
        if response.status_code == httpx.codes.NOT_FOUND:
            self._not_found_cache[filename] = True
            return None
        response = response.raise_for_status()
        self._not_found_cache.pop(filename, False)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            assert len(zf.namelist()) == 1
            csv_filename: str = zf.namelist()[0]
            with zf.open(csv_filename, "r") as csv_file:
                line: str = csv_file.readline().decode()
                has_header: bool = bool(re.search(r"[a-zA-Z]", line))
            with zf.open(csv_filename) as csv_file:
                data: pl.DataFrame = pl.read_csv(
                    csv_file,
                    has_header=has_header,
                    new_columns=tuple(KLINES_CSV_SCHEMAS.keys()),
                    schema=KLINES_CSV_SCHEMAS,
                )
        datetime_dtype = pl.Datetime("ms", pendulum.UTC)
        data = data.with_columns(
            pl.from_epoch("open_time", time_unit="ms").cast(datetime_dtype),
            pl.from_epoch("close_time", time_unit="ms").cast(datetime_dtype),
        )
        return data

    def _load_from_cache(
        self, period: _Period, symbol: str, interval: str, date: Date
    ) -> pl.DataFrame | None:
        cache_file_path: Path = self._cache_file_path(period, symbol, interval, date)
        if not cache_file_path.exists():
            return None
        data: pl.DataFrame = pl.read_parquet(cache_file_path)
        return data

    def _save_to_cache(
        self,
        data: pl.DataFrame,
        period: _Period,
        symbol: str,
        interval: str,
        date: Date,
    ) -> None:
        cache_file_path: Path = self._cache_file_path(period, symbol, interval, date)
        cache_file_path.parent.mkdir(parents=True, exist_ok=True)
        data.write_parquet(cache_file_path)


@grapes.attrs.define
class BinanceDataCollectionFuturesUm(BinanceDataCollection):
    """Binance Futures (UM) klines data collection with parallel downloads and disk caching.

    Note: Data may not be real-time. Use for backtesting/research only; use
    `MarketDataBinanceSdkFuturesUm` for live data.

    Fetches data in parallel and caches persistently for faster access compared to
    direct API calls, especially for large date ranges.

    Examples:
        >>> market_data = BinanceDataCollectionFuturesUm()
        >>> data = await market_data.klines("BTCUSDT", "1m", "2025-12-01", "2026-01-16")
    """

    base_url: str = attrs.field(default="https://data.binance.vision/data/futures/um/")
    cache_dir: Path = attrs.field(
        default=attrs.Factory(_default_cache_dir, takes_self=True), kw_only=True
    )


@grapes.attrs.define
class BinanceDataCollectionSpot(BinanceDataCollection):
    base_url: str = attrs.field(default="https://data.binance.vision/data/spot/")
    cache_dir: Path = attrs.field(
        default=attrs.Factory(_default_cache_dir, takes_self=True), kw_only=True
    )
