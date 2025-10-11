import functools
from typing import Any, override

import attrs
import binance.spot
import cachetools
import polars as pl
from environs import env
from loguru import logger
from typing_extensions import deprecated

import qoc.time_utils as tu
from qoc.api import utils
from qoc.api._abc import Api
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
from qoc.api.typing._exchange_info import ExchangeInfoSymbol
from qoc.time_utils import DateTimeLike

from ._klines import ApiBinanceSpotKlines


@attrs.define
class ApiBinanceSpot(Api):
    client: binance.spot.Spot
    _exchange_info_cache: cachetools.Cache[Any, ExchangeInfo] = attrs.field(
        factory=lambda: cachetools.LRUCache(maxsize=128)
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
        return utils.get_time_unit(self.client)

    # region General

    @override
    def ping(self) -> None:
        return self.client.ping()

    @override
    def step(self) -> bool:
        return False

    @override
    @cachetools.cachedmethod(lambda self: self._exchange_info_cache)
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
        *,
        startTime: DateTimeLike | None = None,
        endTime: DateTimeLike | None = None,
        limit: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        return self._klines(
            symbol=symbol,
            interval=interval,
            startTime=startTime,
            endTime=endTime,
            limit=limit,
            **kwargs,
        )

    @functools.cached_property
    def _klines(self) -> ApiBinanceSpotKlines:
        return ApiBinanceSpotKlines(client=self.client)

    # endregion Market Data

    # region Trading

    @override
    def order(
        self, symbol: str, side: OrderSideLike, type: OrderTypeLike, **kwargs
    ) -> OrderResponseFull:
        side = OrderSide(side)
        type_ = OrderType(type)
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
        **kwargs,
    ) -> OrderResponseFull:
        if quantity is not None:
            exchange_info: ExchangeInfo = self.exchange_info(symbol)
            info_symbol: ExchangeInfoSymbol = exchange_info.get_symbol(symbol)
            if info_symbol.lot_size is not None:
                quantity = info_symbol.lot_size.round(quantity)
            kwargs["quantity"] = quantity
        return self.order(symbol, side, OrderType.MARKET, **kwargs)

    # endregion Trading

    # region Account

    @override
    def account(self, **kwargs) -> Account:
        raw: dict = self.client.account(**kwargs)
        return Account.model_validate(raw)

    # endregion Account


@deprecated("Use `ApiBinanceSpot` instead.")
class ApiBinance(ApiBinanceSpot): ...
