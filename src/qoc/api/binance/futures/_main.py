import functools

import attrs
import polars as pl
from binance_common.configuration import ConfigurationRestAPI
from binance_common.models import ApiResponse
from binance_sdk_derivatives_trading_usds_futures import DerivativesTradingUsdsFutures
from binance_sdk_derivatives_trading_usds_futures.rest_api import (
    DerivativesTradingUsdsFuturesRestAPI,
)
from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
    AccountInformationV3Response,
    NewOrderResponse,
    NewOrderSideEnum,
)
from environs import env

import qoc.time_utils as tu
from qoc.api.binance.utils import get_time_unit
from qoc.api.typing import Interval, Symbol
from qoc.api.typing._enum import OrderSideLike
from qoc.time_utils import DateTimeLike

from ._klines import FuturesKlinesCache


@attrs.define
class ApiBinanceFutures:
    client: DerivativesTradingUsdsFutures

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
            base_url = env.str("BINANCE_BASE_URL", None)
        client = DerivativesTradingUsdsFutures(
            config_rest_api=ConfigurationRestAPI(
                api_key=api_key,  # pyright: ignore[reportArgumentType]
                api_secret=api_secret,
                base_path=base_url,  # pyright: ignore[reportArgumentType]
                **kwargs,
            )
        )
        self.__attrs_init__(client=client)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def rest_api(self) -> DerivativesTradingUsdsFuturesRestAPI:
        return self.client.rest_api

    @property
    def time_unit(self) -> tu.TimeUnit:
        return get_time_unit(self.rest_api)

    def klines(
        self,
        symbol: Symbol,
        interval: Interval,
        *,
        start_time: DateTimeLike | None = None,
        end_time: DateTimeLike | None = None,
        limit: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        return self._klines(
            symbol=symbol,
            interval=interval,
            startTime=start_time,
            endTime=end_time,
            limit=limit,
            **kwargs,
        )

    @functools.cached_property
    def _klines(self) -> FuturesKlinesCache:
        return FuturesKlinesCache(client=self.rest_api)

    def new_order_market(
        self, symbol: Symbol, side: OrderSideLike, quantity: float, **kwargs
    ) -> NewOrderResponse:
        kwargs["symbol"] = symbol
        kwargs["type"] = "MARKET"
        kwargs["quantity"] = quantity
        kwargs["side"] = NewOrderSideEnum(side)
        resp: ApiResponse[NewOrderResponse] = self.rest_api.new_order(**kwargs)
        return resp.data()

    def account_information(self, **kwargs) -> AccountInformationV3Response:
        resp: ApiResponse[AccountInformationV3Response] = (
            self.rest_api.account_information_v3(**kwargs)
        )
        return resp.data()
