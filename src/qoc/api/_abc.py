import abc

import polars as pl

from qoc.api.typing import Symbol
from qoc.time_utils import DateTimeLike

from .typing import (
    Account,
    ExchangeInfo,
    Interval,
    OrderResponseFull,
    OrderSideLike,
    OrderTypeLike,
)


class Api(abc.ABC):
    # region General

    def ping(self) -> None:
        return None

    def step(self) -> bool:
        return False

    def exchange_info(
        self,
        symbol: Symbol | None = None,
        symbols: list[Symbol] | None = None,
        **kwargs,
    ) -> ExchangeInfo:
        raise NotImplementedError

    # endregion General

    # region Market Data

    def klines(
        self,
        symbol: Symbol,
        interval: Interval,
        *,
        startTime: DateTimeLike | None = None,
        endTime: DateTimeLike | None = None,
        limit: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        raise NotImplementedError

    def price(self, symbol: Symbol, interval: Interval = "1m", **kwargs) -> float:
        klines: pl.DataFrame = self.klines(symbol, interval, limit=1, **kwargs)
        return klines["close"].last()  # pyright: ignore[reportReturnType]

    # endregion Market Data

    # region Trading

    def order(
        self,
        symbol: Symbol,
        side: OrderSideLike,
        type: OrderTypeLike,  # noqa: A002
        **kwargs,
    ) -> OrderResponseFull:
        raise NotImplementedError

    @abc.abstractmethod
    def order_market(
        self,
        symbol: Symbol,
        side: OrderSideLike,
        *,
        quantity: float | None = None,
        **kwargs,
    ) -> OrderResponseFull:
        raise NotImplementedError

    # endregion Trading

    # region Account

    def account(self, *, omitZeroBalances: bool = False, **kwargs) -> Account:
        raise NotImplementedError

    # endregion Account
