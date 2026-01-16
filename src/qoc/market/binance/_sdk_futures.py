import asyncio
import itertools
import logging
import math
from collections.abc import Awaitable
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, override

import attrs
import polars as pl
from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
from binance_sdk_derivatives_trading_usds_futures import DerivativesTradingUsdsFutures
from binance_sdk_derivatives_trading_usds_futures.rest_api import (
    DerivativesTradingUsdsFuturesRestAPI,
)
from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
    KlineCandlestickDataIntervalEnum,
)
from liblaf import grapes
from pendulum import DateTime

import qoc.time_utils as tu
from qoc.time_utils import DateTimeLike

from ._abc import BinanceMarketData
from ._common import KLINES_SCHEMAS, filter_klines, parse_start_end

logger: logging.Logger = logging.getLogger(__name__)


@grapes.attrs.define
class MarketDataBinanceSdkFuturesUm(BinanceMarketData):
    """Fetch live kline data from Binance USDS Futures API.

    Provides up-to-date klines without caching. Subject to Binance API rate limits.
    Best for real-time trading; for historical backtesting, use `BinanceDataCollectionFuturesUm`.
    """

    client: DerivativesTradingUsdsFutures = attrs.field(
        factory=lambda: DerivativesTradingUsdsFutures(
            config_rest_api=ConfigurationRestAPI(
                base_path=DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
            )
        )
    )
    executor: Executor = attrs.field(factory=ThreadPoolExecutor, kw_only=True)
    limit: int = attrs.field(default=1000, kw_only=True)

    @property
    def rest(self) -> DerivativesTradingUsdsFuturesRestAPI:
        return self.client.rest_api

    @override
    async def klines(
        self,
        symbol: str,
        interval: str,
        start: DateTimeLike | None = None,
        end: DateTimeLike | None = None,
    ) -> pl.DataFrame:
        start, end = parse_start_end(start, end, interval)
        interval_str: str = interval
        interval: tu.Interval = tu.Interval.parse(interval)
        chunk_interval: tu.Interval = tu.Interval(
            self.limit * interval.count, interval.unit
        )
        tasks: list[Awaitable[list[list[Any]]]] = []
        for chunk_start in pl.datetime_range(
            start,
            end,
            chunk_interval.to_str(case_sensitive=False),
            closed="left",
            time_unit="ms",
            time_zone="UTC",
            eager=True,
        ):
            chunk_end: DateTime = chunk_start + chunk_interval.duration
            tasks.append(self._fetch(symbol, interval_str, chunk_start, chunk_end))
        raw: list[list[Any]] = list(
            itertools.chain.from_iterable(await asyncio.gather(*tasks))
        )
        data: pl.DataFrame = pl.from_records(raw, KLINES_SCHEMAS, orient="row")
        data = filter_klines(data, symbol, interval_str, start, end)
        return data

    async def _fetch(
        self, symbol: str, interval: str, start: DateTime, end: DateTime
    ) -> list[list[Any]]:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        data: list[list[Any]] = await loop.run_in_executor(
            self.executor, self._fetch_sync, symbol, interval, start, end
        )
        return data

    def _fetch_sync(
        self, symbol: str, interval: str, start: DateTime, end: DateTime
    ) -> list[list[Any]]:
        data: list[list[Any]] = self.rest.kline_candlestick_data(
            symbol=symbol,
            interval=KlineCandlestickDataIntervalEnum(interval),
            start_time=math.floor(tu.as_timestamp(start, unit="ms")),
            end_time=math.floor(tu.as_timestamp(end, unit="ms")),
            limit=self.limit,
        ).data()  # pyright: ignore[reportAssignmentType]
        return data
