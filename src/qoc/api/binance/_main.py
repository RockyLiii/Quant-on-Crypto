import datetime
from typing import Self

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
        start_time: datetime.datetime | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        if start_time:
            kwargs["startTime"] = self.timeunit.from_datetime(start_time)
        response: list[list[int | str]] = self.client.klines(
            symbol=symbol, interval=interval, **kwargs
        )
        data: pl.DataFrame = pl.from_records(
            response,
            schema=[
                (
                    "open_time",
                    pl.Datetime(time_unit=self.timeunit.to_polars),
                ),  # Kline open time
                ("open", pl.Float64),  # Open price
                ("high", pl.Float64),  # High price
                ("low", pl.Float64),  # Low price
                ("close", pl.Float64),  # Close price
                ("volume", pl.Float64),  # Volume
                (
                    "close_time",
                    pl.Datetime(time_unit=self.timeunit.to_polars),
                ),  # Kline Close time
                ("quote_volume", pl.Float64),  # Quote asset volume
                ("count", pl.Int64),  # Number of trades
                ("taker_buy_volume", pl.Float64),  # Taker buy base asset volume
                ("taker_buy_quote_volume", pl.Float64),  # Taker buy quote asset volume
                ("ignore", pl.Int64),  # Unused field, ignore.
            ],
            orient="row",
        )


        # 将 Datetime 时间戳转换为整数微秒时间戳
        time_unit_multiplier = {
            "ns": 0.001,  # 纳秒转微秒需要除以 1000
            "us": 1,      # 微秒保持不变
            "ms": 1000,   # 毫秒转微秒需要乘以 1000
            "s": 1000000, # 秒转微秒需要乘以 1000000
        }[self.timeunit.to_polars]

        # 转换时间列为整数微秒时间戳
        data = data.with_columns([
            pl.col("open_time").dt.epoch("us").alias("open_time"),
            pl.col("close_time").dt.epoch("us").alias("close_time")
        ])
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
