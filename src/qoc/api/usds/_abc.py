import abc
from typing import overload

import polars as pl
from pendulum import DateTime

from qoc.typing import DecimalLike, SymbolName

from .models import (
    Account,
    CommissionRate,
    ExchangeInfo,
    MarginType,
    OrderResponse,
    OrderSide,
    TickerPrice,
    TickerPriceDict,
)


class ApiUsds(abc.ABC):
    # ------------------------------ Market Data ----------------------------- #

    @abc.abstractmethod
    def exchange_info(self) -> ExchangeInfo:
        raise NotImplementedError

    @abc.abstractmethod
    def klines(
        self,
        symbol: SymbolName,
        interval: str,
        start_time: DateTime | None = None,
        end_time: DateTime | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        raise NotImplementedError

    @overload
    def ticker_price(self, symbol: SymbolName) -> TickerPrice: ...
    @overload
    def ticker_price(self, symbol: None = None) -> TickerPriceDict: ...
    @abc.abstractmethod
    def ticker_price(
        self, symbol: SymbolName | None = None
    ) -> TickerPrice | TickerPriceDict:
        raise NotImplementedError

    # --------------------------------- Trade -------------------------------- #

    @abc.abstractmethod
    def order_market(
        self, symbol: SymbolName, side: OrderSide, quantity: DecimalLike
    ) -> OrderResponse:
        raise NotImplementedError

    @abc.abstractmethod
    def change_margin_type(self, symbol: SymbolName, margin_type: MarginType) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def change_position_mode(self, *, dual_side_position: bool) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def change_leverage(self, symbol: SymbolName, leverage: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def change_multi_assets_mode(self, *, multi_assets_margin: bool) -> None:
        raise NotImplementedError

    # -------------------------------- Account ------------------------------- #

    @abc.abstractmethod
    def account(self) -> Account:
        raise NotImplementedError

    def commission_rate(self, symbol: SymbolName) -> CommissionRate:
        raise NotImplementedError
