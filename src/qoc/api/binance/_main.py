import datetime
from typing import Any, Self

import attrs
import binance.spot
import polars as pl
from environs import env
from loguru import logger

from qoc.api.typing import (
    Account,
    Interval,
    OrderResponseFull,
    OrderSide,
    OrderSideLike,
    OrderType,
    OrderTypeLike,
)

from ._exchange_info import ExchangeInfo
from ._typing import TimeUnit


@attrs.define
class ApiBinance:
    client: binance.spot.Spot = attrs.field()
    symbol_average_price: dict[str, float] = attrs.field(factory=dict)  # TODO

    @classmethod
    def create(
        cls, api_key: str | None = None, api_secret: str | None = None, **kwargs
    ) -> Self:
        if api_key is None:
            api_key = env.str("BINANCE_API_KEY", None)
        if api_secret is None:
            api_secret = env.str("BINANCE_API_SECRET", None)
        if not kwargs.get("base_url") and (
            base_url := env.str("BINANCE_BASE_URL", None)
        ):
            kwargs["base_url"] = base_url
        client = binance.spot.Spot(api_key=api_key, api_secret=api_secret, **kwargs)
        return cls(client=client)

    @property
    def timeunit(self) -> TimeUnit:
        """.

        All time and timestamp related fields in the JSON responses are in **milliseconds by default**.

        References:
            1. <https://developers.binance.com/docs/binance-spot-api-docs/testnet/rest-api/general-api-information>
        """
        return TimeUnit(
            self.client.session.headers.get("X-MBX-TIME-UNIT", TimeUnit.MILLISECOND)
        )

    # region General

    def ping(self) -> None:
        self.client.ping()

    def step(self) -> bool:
        return False

    def exchange_info(
        self,
        symbol: str | None = None,
        symbols: list[str] | None = None,
        permissions: list[str] | None = None,
    ) -> ExchangeInfo:
        data = self.client.exchange_info(
            symbol=symbol,  # pyright: ignore[reportArgumentType]
            symbols=symbols,  # pyright: ignore[reportArgumentType]
            permissions=permissions,  # pyright: ignore[reportArgumentType]
        )
        return ExchangeInfo.model_validate(data)

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
        raw: list[list[Any]] = []
        time: datetime.datetime | None = startTime
        while (time is None) or (endTime is None) or (time < endTime):
            if time is not None:
                kwargs["startTime"] = self.timeunit.from_datetime(time)
            if endTime is not None:
                kwargs["endTime"] = self.timeunit.from_datetime(endTime)
            delta: list[list[Any]] = self.client.klines(
                symbol=symbol, interval=interval, **kwargs
            )
            if len(delta) == 0:
                break
            raw.extend(delta)
            time = self.timeunit.from_int(delta[-1][6])
        data: pl.DataFrame = pl.from_records(
            raw,
            [
                (
                    "open_time",
                    pl.Datetime(
                        time_unit=self.timeunit.to_polars, time_zone=datetime.UTC
                    ),
                ),
                ("open", pl.Float64),
                ("high", pl.Float64),
                ("low", pl.Float64),
                ("close", pl.Float64),
                ("volume", pl.Float64),
                (
                    "close_time",
                    pl.Datetime(
                        time_unit=self.timeunit.to_polars, time_zone=datetime.UTC
                    ),
                ),
                ("quote_asset_volume", pl.Float64),
                ("number_of_trades", pl.Int64),
                ("taker_buy_base_asset_volume", pl.Float64),
                ("taker_buy_quote_asset_volume", pl.Float64),
                ("ignore", pl.String),
            ],
            orient="row",
        )
        return data

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
        raw: dict = self.client.new_order(
            symbol=symbol, side=side, type=type_, **kwargs
        )
        response: OrderResponseFull = OrderResponseFull.model_validate(raw)
        logger.info(response)
        return response

    def order_market(
        self,
        symbol: str,
        side: OrderSideLike,
        *,
        quantity: float | None = None,
        quoteOrderQty: float | None = None,
        **kwargs,
    ) -> OrderResponseFull:
        if quantity is not None:
            kwargs["quantity"] = f"{quantity:f}"
        if quoteOrderQty is not None:
            kwargs["quoteOrderQty"] = f"{quoteOrderQty:f}"
        return self.order(symbol, side, OrderType.MARKET, **kwargs)

    # endregion Trading

    # region Account

    def account(self, **kwargs) -> Account:
        raw: dict = self.client.account(**kwargs)
        return Account.model_validate(raw)

    # endregion Account
