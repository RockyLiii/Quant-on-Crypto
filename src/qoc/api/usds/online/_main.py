import functools
from typing import Any, overload, override

import attrs
import cachetools
import pendulum
import polars as pl
from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
)
from binance_sdk_derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)
from binance_sdk_derivatives_trading_usds_futures.rest_api import (
    DerivativesTradingUsdsFuturesRestAPI,
)
from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
    AccountInformationV3Response,
    ChangeMarginTypeMarginTypeEnum,
    ExchangeInformationResponse,
    FuturesAccountConfigurationResponse,
    NewOrderResponse,
    NewOrderSideEnum,
    SymbolPriceTickerV2Response,
    SymbolPriceTickerV2Response1,
    SymbolPriceTickerV2Response2,
    UserCommissionRateResponse,
)
from environs import env
from loguru import logger

from qoc import utils
from qoc.api.binance.futures._klines import FuturesKlinesCache
from qoc.api.usds._abc import ApiUsds
from qoc.api.usds.models import (
    Account,
    AccountConfig,
    CommissionRate,
    ExchangeInfo,
    ExchangeInfoSymbol,
    MarginType,
    OrderResponse,
    OrderSide,
    SymbolConfig,
    SymbolConfigDict,
    TickerPrice,
    TickerPriceDict,
)
from qoc.typing import DecimalLike, SymbolName


@attrs.define
class ApiUsdsOnline(ApiUsds):
    wrapped: DerivativesTradingUsdsFutures

    def __init__(self, *, base_path: str | None = None) -> None:
        wrapped = DerivativesTradingUsdsFutures(
            config_rest_api=ConfigurationRestAPI(
                api_key=env.str("BINANCE_USDS_API_KEY", ""),
                api_secret=env.str("BINANCE_USDS_API_SECRET", None),
                base_path=base_path
                or utils.get_base_url(
                    "BINANCE_USDS_BASE_URL",
                    prod=DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
                    testnet=DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
                ),
                timeout=5000,  # milliseconds
                backoff=1,  # seconds
            )
        )
        self.__attrs_init__(wrapped=wrapped)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def rest(self) -> DerivativesTradingUsdsFuturesRestAPI:
        return self.wrapped.rest_api

    # ------------------------------ Market Data ----------------------------- #

    _exchange_info_cache: cachetools.Cache = attrs.field(
        repr=False, init=False, factory=lambda: cachetools.TTLCache(maxsize=1, ttl=3600)
    )

    @override
    @cachetools.cachedmethod(lambda self: self._exchange_info_cache)
    def exchange_info(self) -> ExchangeInfo:
        response: ExchangeInformationResponse = self.rest.exchange_information().data()
        return ExchangeInfo.model_validate(
            response.model_dump(mode="json", by_alias=True)
        )

    @override
    def klines(
        self,
        symbol: SymbolName,
        interval: str,  # TODO(liblaf): use enum
        start_time: pendulum.DateTime | None = None,
        end_time: pendulum.DateTime | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        # TODO(liblaf): better caching mechanism
        return self._klines_cache(
            symbol, interval, start_time=start_time, end_time=end_time, limit=limit
        )

    @functools.cached_property
    def _klines_cache(self) -> FuturesKlinesCache:
        return FuturesKlinesCache(client=self.rest)

    @overload
    def ticker_price(self, symbol: str) -> TickerPrice: ...
    @overload
    def ticker_price(self, symbol: None = None) -> TickerPriceDict: ...
    @override
    def ticker_price(self, symbol: str | None = None) -> TickerPrice | TickerPriceDict:
        # TODO(liblaf): implement caching
        response: SymbolPriceTickerV2Response = self.rest.symbol_price_ticker_v2(
            symbol
        ).data()
        if symbol is None:
            actual_instance_2: list[SymbolPriceTickerV2Response2] = (
                response.actual_instance
            )  # pyright: ignore[reportAssignmentType]
            return TickerPriceDict.model_validate(
                [
                    item.model_dump(mode="json", by_alias=True)
                    for item in actual_instance_2
                ]
            )
        actual_instance_1: SymbolPriceTickerV2Response1 = response.actual_instance  # pyright: ignore[reportAssignmentType]
        return TickerPrice.model_validate(
            actual_instance_1.model_dump(mode="json", by_alias=True)
        )

    # --------------------------------- Trade -------------------------------- #

    @override
    def order_market(
        self, symbol: SymbolName, side: OrderSide, quantity: DecimalLike
    ) -> OrderResponse:
        info: ExchangeInfoSymbol = self.exchange_info().symbols[symbol]
        if (market_lot_size := info.market_lot_size) is not None:
            quantity = market_lot_size.round(quantity)
        response: NewOrderResponse = self.rest.new_order(
            symbol=symbol,
            side=NewOrderSideEnum(side),
            type="MARKET",
            quantity=str(quantity),  # pyright: ignore[reportArgumentType]
        ).data()
        return OrderResponse.model_validate(
            response.model_dump(mode="json", by_alias=True)
        )

    def change_margin_type(self, symbol: SymbolName, margin_type: MarginType) -> None:
        symbol_config: SymbolConfig = self.symbol_config(symbol)
        if symbol_config.margin_type != margin_type:
            self.rest.change_margin_type(
                symbol, ChangeMarginTypeMarginTypeEnum(margin_type)
            )
        logger.success(
            "change margin type > symbol: {}, margin type: {}", symbol, margin_type
        )

    @override
    def change_position_mode(self, *, dual_side_position: bool) -> None:
        config: AccountConfig = self.account_config()
        if config.dual_side_position != dual_side_position:
            self.rest.change_position_mode("true" if dual_side_position else "false")
        logger.success(
            "change position mode: {}",
            "hedge mode" if dual_side_position else "one-way mode",
        )

    @override
    def change_leverage(self, symbol: str, leverage: int) -> None:
        symbol_config: SymbolConfig = self.symbol_config(symbol)
        if symbol_config.leverage != leverage:
            self.rest.change_initial_leverage(symbol, leverage)
        logger.success("change leverage > symbol: {}, leverage: {}", symbol, leverage)

    @override
    def change_multi_assets_mode(self, *, multi_assets_margin: bool) -> None:
        config: AccountConfig = self.account_config()
        if config.multi_assets_margin != multi_assets_margin:
            self.rest.change_multi_assets_mode(
                "true" if multi_assets_margin else "false"
            )
        logger.success(
            "change multi-assets mode: {}",
            "multi-assets mode" if multi_assets_margin else "single-asset mode",
        )

    # -------------------------------- Account ------------------------------- #

    @override
    def account(self) -> Account:
        response: AccountInformationV3Response = (
            self.rest.account_information_v3().data()
        )
        return Account.model_validate(response.model_dump(mode="json", by_alias=True))

    @override
    def commission_rate(self, symbol: SymbolName) -> CommissionRate:
        response: UserCommissionRateResponse = self.rest.user_commission_rate(
            symbol
        ).data()
        return CommissionRate.model_validate(
            response.model_dump(mode="json", by_alias=True)
        )

    def account_config(self) -> AccountConfig:
        response: FuturesAccountConfigurationResponse = (
            self.rest.futures_account_configuration().data()
        )
        return response

    @overload
    def symbol_config(self, symbol: SymbolName) -> SymbolConfig: ...
    @overload
    def symbol_config(self, symbol: None = None) -> SymbolConfigDict: ...
    def symbol_config(
        self, symbol: SymbolName | None = None
    ) -> SymbolConfig | SymbolConfigDict:
        response: list[dict[str, Any]] = self.rest.symbol_configuration(symbol).data()  # pyright: ignore[reportAssignmentType]
        if symbol is None:
            return SymbolConfigDict.model_validate(response)
        assert len(response) == 1
        return SymbolConfig.model_validate(response[0])
