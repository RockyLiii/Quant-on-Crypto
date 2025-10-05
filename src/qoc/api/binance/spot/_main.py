import functools
from typing import Any

import attrs
import binance.spot
import cachetools
import polars as pl
from environs import env
from loguru import logger

import qoc.time_utils as tu
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

from ._klines import KLines


@attrs.define
class ApiBinanceSpot:
    client: binance.spot.Spot
    _cache_klines: cachetools.Cache[tuple[Symbol, Interval], pl.DataFrame] = (
        attrs.field(factory=lambda: cachetools.LRUCache(maxsize=1024))
    )

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

    def ping(self) -> None:
        return self.client.ping()

    def step(self) -> bool:
        return False

    def exchange_info(
        self,
        symbol: str | None = None,
        symbols: list[str] | None = None,
        permissions: list[str] | None = None,
    ) -> ExchangeInfo:
        raw: Any = self.client.exchange_info(
            symbol=symbol,  # pyright: ignore[reportArgumentType]
            symbols=symbols,  # pyright: ignore[reportArgumentType]
            permissions=permissions,  # pyright: ignore[reportArgumentType]
        )
        return ExchangeInfo.model_validate(raw)

    # endregion General

    # region Market Data

    @functools.cached_property
    def klines(self) -> KLines:
        return KLines(client=self.client)

    # endregion Market Data

    # region Trading

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

    def order_market(
        self,
        symbol: str,
        side: OrderSideLike,
        *,
        quantity: float | str | None = None,
        quoteOrderQty: float | str | None = None,
        **kwargs,
    ) -> OrderResponseFull:
        if quantity is not None:
            kwargs["quantity"] = _as_str(quantity)
        if quoteOrderQty is not None:
            kwargs["quoteOrderQty"] = _as_str(quoteOrderQty)
        return self.order(symbol, side, OrderType.MARKET, **kwargs)

    # endregion Trading

    # region Account

    def account(self, **kwargs) -> Account:
        raw: dict = self.client.account(**kwargs)
        return Account.model_validate(raw)

    # endregion Account


def _as_str(obj: str | float) -> str:
    return str(obj)
