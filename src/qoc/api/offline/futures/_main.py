import attrs
import polars as pl
from binance_sdk_derivatives_trading_usds_futures.rest_api import (
    DerivativesTradingUsdsFuturesRestAPI,
)
from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
    AccountInformationV3Response,
    AccountInformationV3ResponseAssetsInner,
    AccountInformationV3ResponsePositionsInner,
    ExchangeInformationResponse,
    ExchangeInformationResponseSymbolsInner,
    NewOrderResponse,
)

import qoc.time_utils as tu
from qoc.api.binance.futures import ApiBinanceFutures
from qoc.api.binance.utils import get_time_unit
from qoc.api.typing import Asset, Interval, Symbol
from qoc.api.typing._enum import OrderSide, OrderSideLike
from qoc.time_utils import DateTimeLike


@attrs.define
class AccountAsset:
    asset: Asset
    available_balance: float
    wallet_balance: float


@attrs.define
class AccountPosition:
    symbol: Symbol
    position_amount: float


@attrs.define
class ApiOfflineFutures:
    client: ApiBinanceFutures = attrs.field(factory=ApiBinanceFutures)
    _assets: dict[Symbol, AccountAsset] = attrs.field(factory=dict)
    _positions: dict[Symbol, AccountPosition] = attrs.field(factory=dict)

    @property
    def rest_api(self) -> DerivativesTradingUsdsFuturesRestAPI:
        return self.client.rest_api

    @property
    def time_unit(self) -> tu.TimeUnit:
        return get_time_unit(self.rest_api)

    def exchange_information(self) -> ExchangeInformationResponse:
        return self.client.exchange_information()

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
        return self.client.klines(
            symbol,
            interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            **kwargs,
        )

    def new_order_market(
        self, symbol: Symbol, side: OrderSideLike, quantity: float, **kwargs
    ) -> NewOrderResponse:
        side = OrderSide(side)
        info: ExchangeInformationResponseSymbolsInner = self._exchange_info_symbol(
            symbol
        )
        assert info.base_asset is not None
        assert info.quote_asset is not None
        price: float = self.klines(symbol, "1m", limit=1)["close"].last()  # pyright: ignore[reportAssignmentType]
        asset: AccountAsset = self._assets[info.quote_asset]
        position: AccountPosition = (
            self._positions[symbol]
            if symbol in self._positions
            else AccountPosition(symbol=symbol, position_amount=0.0)
        )
        if side == OrderSide.SELL:
            quantity = -quantity
        if asset.available_balance < quantity * price:
            msg: str = f"Insufficient balance for {quantity * price}, available: {asset.available_balance}"
            raise ValueError(msg)
        position.position_amount += quantity
        asset.available_balance -= quantity * price
        self._assets[info.quote_asset] = asset
        self._positions[symbol] = position
        resp = NewOrderResponse(
            avgPrice=str(price),
            origQty=str(abs(quantity)),
            side=str(side),
            symbol=symbol,
        )
        return resp

    def account_information(self, **kwargs) -> AccountInformationV3Response:
        resp = AccountInformationV3Response()
        resp.assets = []
        resp.positions = []
        for asset in self._assets.values():
            asset.wallet_balance = asset.available_balance
        for position in self._positions.values():
            info: ExchangeInformationResponseSymbolsInner = self._exchange_info_symbol(
                position.symbol
            )
            assert info.quote_asset is not None
            asset: AccountAsset = self._assets[info.quote_asset]
            price: float = self.klines(position.symbol, "1m", limit=1)["close"].last()  # pyright: ignore[reportAssignmentType]
            asset.wallet_balance += position.position_amount * price
            resp.positions.append(
                AccountInformationV3ResponsePositionsInner(
                    symbol=position.symbol, positionAmt=str(position.position_amount)
                )
            )
        for asset in self._assets.values():
            resp.assets.append(
                AccountInformationV3ResponseAssetsInner(
                    asset=asset.asset,
                    availableBalance=str(asset.available_balance),
                    walletBalance=str(asset.wallet_balance),
                )
            )
        return resp

    def _exchange_info_symbol(
        self, symbol: Symbol
    ) -> ExchangeInformationResponseSymbolsInner:
        ex_info: ExchangeInformationResponse = self.client.exchange_information()
        assert ex_info.symbols is not None
        for symbol_info in ex_info.symbols:
            if symbol_info.symbol == symbol:
                return symbol_info
        raise KeyError(symbol)
