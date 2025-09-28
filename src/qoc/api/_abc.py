import abc
import datetime

import polars as pl

from .typing import (
    Account,
    ExchangeInfo,
    Interval,
    OrderResponseFull,
    OrderSideLike,
    OrderTypeLike,
)


class TradingApi(abc.ABC):
    # region General

    def ping(self) -> None:
        return None

    def step(self) -> bool:
        return False

    def exchange_info(
        self,
        symbol: str | None = None,
        symbols: list[str] | None = None,
        permissions: list[str] | None = None,
    ) -> ExchangeInfo:
        raise NotImplementedError

    # endregion General

    # region Market Data

    def klines(
        self,
        symbol: str,
        interval: Interval,
        *,
        startTime: datetime.datetime | None = None,
        endTime: datetime.datetime | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        raise NotImplementedError

    # endregion Market Data

    # region Trading

    def order(
        self, symbol: str, side: OrderSideLike, type_: OrderTypeLike, **kwargs
    ) -> OrderResponseFull:
        raise NotImplementedError

    @abc.abstractmethod
    def order_market(
        self,
        symbol: str,
        side: OrderSideLike,
        *,
        quantity: float | None = None,
        quoteOrderQty: float | None = None,
        **kwargs,
    ) -> OrderResponseFull:
        raise NotImplementedError

    # endregion Trading

    # region Account

    def account(self, **kwargs) -> Account:
        raise NotImplementedError

    # endregion Account
