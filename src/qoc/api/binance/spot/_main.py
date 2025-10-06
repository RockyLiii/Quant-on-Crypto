import functools
from typing import Any, override

import attrs
import binance.spot
import polars as pl
from environs import env
from loguru import logger
from typing_extensions import deprecated

import qoc.time_utils as tu
from qoc.api._abc import AbstractApi
from qoc.api.binance._utils import get_time_unit
from qoc.api.typing import (
    Account,
    ExchangeInfo,
    Interval,
    OrderResponseFull,
    OrderSide,
    OrderSideLike,
    OrderType,
    OrderTypeLike,
    Symbol,
)
from qoc.time_utils import DateTimeLike

from ._klines import KLines


@attrs.define
class ApiBinanceSpot(AbstractApi):
    client: binance.spot.Spot

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ) -> None:
        if api_key is None:
            api_key = env.str("BINANCE_API_KEY", None)
        if api_secret is None:
            api_secret = env.str("BINANCE_API_SECRET", None)
        if base_url is None:
            kwargs["base_url"] = env.str("BINANCE_BASE_URL", None)
        client = binance.spot.Spot(api_key=api_key, api_secret=api_secret, **kwargs)
        self.__attrs_init__(client=client)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def time_unit(self) -> tu.TimeUnit:
        return get_time_unit(self.client)

    # region General

    @override
    def ping(self) -> None:
        return self.client.ping()

    @override
    def step(self) -> bool:
        return False

    @override
    def exchange_info(
        self,
        symbol: Symbol | None = None,
        symbols: list[Symbol] | None = None,
        permissions: list[str] | None = None,
        **kwargs,
    ) -> ExchangeInfo:
        raw: Any = self.client.exchange_info(
            symbol=symbol,  # pyright: ignore[reportArgumentType]
            symbols=symbols,  # pyright: ignore[reportArgumentType]
            permissions=permissions,  # pyright: ignore[reportArgumentType]
        )
        return ExchangeInfo.model_validate(raw)

    # endregion General

    # region Market Data

    @override
    def klines(
        self,
        symbol: Symbol,
        interval: Interval,
        startTime: DateTimeLike | None = None,
        endTime: DateTimeLike | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        return self._klines(
            symbol=symbol,
            interval=interval,
            startTime=startTime,
            endTime=endTime,
            **kwargs,
        )

    @functools.cached_property
    def _klines(self) -> KLines:
        return KLines(client=self.client)

    # endregion Market Data

    # region Trading

    @override
    def order(
        self, symbol: str, side: OrderSideLike, type_: OrderTypeLike, **kwargs
    ) -> OrderResponseFull:
        side = OrderSide(side)
        type_ = OrderType(type_)
        logger.info(
            "New Order > symbol: {}, side: {}, type: {}, {}",
            symbol,
            side,
            type_,
            kwargs,
        )
        raw: Any = self.client.new_order(symbol=symbol, side=side, type=type_, **kwargs)
        response: OrderResponseFull = OrderResponseFull.model_validate(raw)
        logger.info(response)
        return response

    @override
    def order_market(
        self,
        symbol: str,
        side: OrderSideLike,
        *,
        quantity: float | str | None = None,
        quoteOrderQty: float | str | None = None,
        **kwargs,
    ) -> OrderResponseFull:
        # TODO(liblaf): round quantity according to LOT_SIZE filter
        if quantity is not None:
            kwargs["quantity"] = _format_float(quantity)
        if quoteOrderQty is not None:
            kwargs["quoteOrderQty"] = _format_float(quoteOrderQty)
        return self.order(symbol, side, OrderType.MARKET, **kwargs)

    # endregion Trading

    # region Account

    @override
    def account(self, **kwargs) -> Account:
        raw: dict = self.client.account(**kwargs)
        return Account.model_validate(raw)

    # endregion Account


def _format_float(obj: str | float) -> str:
    return f"{obj:f}"


@deprecated("Use `ApiBinanceSpot` instead.")
class ApiBinance(ApiBinanceSpot): ...
