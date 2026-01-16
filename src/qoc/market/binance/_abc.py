import abc
import asyncio
from collections.abc import Iterable

import polars as pl

from qoc.time_utils import DateTimeLike


class BinanceMarketData(abc.ABC):
    @abc.abstractmethod
    async def klines(
        self,
        symbol: str,
        interval: str,
        start: DateTimeLike | None = None,
        end: DateTimeLike | None = None,
    ) -> pl.DataFrame:
        raise NotImplementedError

    async def klines_batch(
        self,
        symbols: Iterable[str],
        interval: str,
        start: DateTimeLike | None = None,
        end: DateTimeLike | None = None,
    ) -> dict[str, pl.DataFrame]:
        data: list[pl.DataFrame] = await asyncio.gather(
            *[self.klines(symbol, interval, start, end) for symbol in symbols]
        )
        return dict(zip(symbols, data, strict=True))
