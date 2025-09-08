from datetime import datetime
from typing import override

import attrs
from polars.dataframe import DataFrame

import qoc.database as db
from qoc.api._abc import TradingApi
from qoc.api.typing import Account, Interval, OrderResponseFull, OrderSideLike


@attrs.define
class ApiOffline(TradingApi):
    library: db.Library

    @override
    def account(self, **kwargs) -> Account:
        # TODO(liblaf): Implement.
        raise NotImplementedError

    @override
    def klines(
        self,
        symbol: str,
        interval: Interval,
        *,
        start_time: datetime | None = None,
        **kwargs,
    ) -> DataFrame:
        # TODO(liblaf): Implement.
        raise NotImplementedError

    @override
    def order_market(
        self,
        symbol: str,
        side: OrderSideLike,
        *,
        quantity: float | None = None,
        quoteOrderQty: float | None = None,
        timestamp: datetime | None = None,
        **kwargs,
    ) -> OrderResponseFull:
        # TODO(liblaf): Implement.
        raise NotImplementedError
