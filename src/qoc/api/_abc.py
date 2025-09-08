import abc
import datetime

import polars as pl

from .typing import Account, Interval, OrderResponseFull, OrderSideLike


class TradingApi(abc.ABC):
    @abc.abstractmethod
    def account(self, **kwargs) -> Account:
        raise NotImplementedError

    @abc.abstractmethod
    def klines(
        self,
        symbol: str,
        interval: Interval,
        *,
        start_time: datetime.datetime | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        raise NotImplementedError

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
