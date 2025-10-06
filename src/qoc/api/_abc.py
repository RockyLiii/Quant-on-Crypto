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


class AbstractApi(abc.ABC):
    # region General

    def ping(self) -> None:
        return None

    def step(self) -> bool:
        return False

    def exchange_info(
        self,
        symbol: Symbol | None = None,
        symbols: list[Symbol] | None = None,
        permissions: list[str] | None = None,
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
        **kwargs,
    ) -> pl.DataFrame:
        raise NotImplementedError

    # endregion Market Data

    # region Trading

    def order(
        self, symbol: Symbol, side: OrderSideLike, type_: OrderTypeLike, **kwargs
    ) -> OrderResponseFull:
        raise NotImplementedError

    @abc.abstractmethod
    def order_market(
        self,
        symbol: Symbol,
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
