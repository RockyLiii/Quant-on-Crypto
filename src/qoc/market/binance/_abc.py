import abc

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
