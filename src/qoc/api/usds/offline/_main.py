from decimal import Decimal
from typing import overload, override

import attrs
import polars as pl
from pendulum import DateTime

import qoc.time_utils as tu
from qoc.api.usds._abc import ApiUsds
from qoc.api.usds.models import (
    Account,
    AccountAsset,
    AccountAssetsDict,
    AccountPosition,
    AccountPositionsDict,
    CommissionRate,
    ExchangeInfo,
    MarginType,
    OrderResponse,
    OrderSide,
    PositionSide,
    TickerPrice,
    TickerPriceDict,
)
from qoc.api.usds.models._exchange_info import ExchangeInfoSymbol
from qoc.api.usds.online import ApiUsdsOnline
from qoc.typing import AssetName, DecimalLike, SymbolName


@attrs.define
class Asset:
    asset: AssetName
    available_balance: float

    def dump(self, now: DateTime) -> AccountAsset:
        return AccountAsset(
            asset=self.asset,
            wallet_balance=self.available_balance,
            unrealized_profit=0.0,
            margin_balance=self.available_balance,
            available_balance=self.available_balance,
            update_time=now,  # pyright: ignore[reportArgumentType]
        )


@attrs.define
class Position:
    symbol: SymbolName
    position_amt: Decimal
    isolated_wallet: float

    def dump(self, price: float, now: DateTime) -> AccountPosition:
        isolated_margin: float = price * float(self.position_amt)
        unrealized_profit: float = isolated_margin - self.isolated_wallet
        return AccountPosition(
            symbol=self.symbol,
            position_side=PositionSide.BOTH,
            position_amt=self.position_amt,  # pyright: ignore[reportArgumentType]
            unrealized_profit=unrealized_profit,
            isolated_margin=isolated_margin,
            notional=0.0,  # TODO(liblaf): compute notional
            isolated_wallet=self.isolated_wallet,
            update_time=now,  # pyright: ignore[reportArgumentType]
        )


@attrs.define
class ApiUsdsOffline(ApiUsds):
    _online: ApiUsdsOnline = attrs.field(factory=ApiUsdsOnline)
    _assets: dict[AssetName, Asset] = attrs.field(
        factory=lambda: {"USDT": Asset(asset="USDT", available_balance=5000.0)}
    )
    _positions: dict[SymbolName, Position] = attrs.field(factory=dict)

    # ------------------------------ Market Data ----------------------------- #

    @override
    def exchange_info(self) -> ExchangeInfo:
        return self._online.exchange_info()

    @override
    def klines(
        self,
        symbol: SymbolName,
        interval: str,  # TODO(liblaf): use enum
        start_time: DateTime | None = None,
        end_time: DateTime | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        end_time = tu.now() if end_time is None else min(end_time, tu.now())
        return self._online.klines(
            symbol, interval, start_time=start_time, end_time=end_time, limit=limit
        )

    @overload
    def ticker_price(self, symbol: SymbolName) -> TickerPrice: ...
    @overload
    def ticker_price(self, symbol: None = None) -> TickerPriceDict: ...
    @override
    def ticker_price(
        self, symbol: SymbolName | None = None
    ) -> TickerPrice | TickerPriceDict:
        if symbol is None:
            raise NotImplementedError
        now: DateTime = tu.now()
        klines: pl.DataFrame = self.klines(symbol, "1m", end_time=now, limit=1)
        price: float = klines["close"].last()  # pyright: ignore[reportAssignmentType]
        return TickerPrice(symbol=symbol, price=price, time=now)  # pyright: ignore[reportArgumentType]

    # --------------------------------- Trade -------------------------------- #

    @override
    def order_market(
        self, symbol: SymbolName, side: OrderSide, quantity: DecimalLike
    ) -> OrderResponse:
        now: DateTime = tu.now()
        info: ExchangeInfoSymbol = self.exchange_info().symbols[symbol]
        if (market_lot_size := info.market_lot_size) is not None:
            quantity = market_lot_size.round(quantity)
        else:
            quantity = Decimal(quantity)
        match side:
            case OrderSide.BUY:
                pass
            case OrderSide.SELL:
                quantity = -quantity
            case _:
                raise ValueError
        price: float = self.ticker_price(symbol).price
        notional: float = price * float(quantity)
        asset: Asset = self._assets[info.quote_asset]
        position: Position = self._positions.get(
            symbol,
            Position(symbol=symbol, position_amt=Decimal(0), isolated_wallet=0.0),
        )
        commission_rate: float = self.commission_rate(symbol).taker
        commission: float = commission_rate * abs(notional)
        asset.available_balance -= notional + commission
        position.position_amt += quantity
        position.isolated_wallet += notional
        self._assets[info.quote_asset] = asset
        self._positions[symbol] = position
        return OrderResponse(
            orig_qty=quantity,
            side=side,
            position_side=PositionSide.BOTH,
            symbol=symbol,
            update_time=now,  # pyright: ignore[reportArgumentType]
        )

    @override
    def change_margin_type(self, symbol: SymbolName, margin_type: MarginType) -> None:
        if margin_type != MarginType.ISOLATED:
            raise NotImplementedError

    @override
    def change_position_mode(self, *, dual_side_position: bool) -> None:
        if dual_side_position:
            raise NotImplementedError

    @override
    def change_leverage(self, symbol: SymbolName, leverage: int) -> None:
        if leverage != 1:
            raise NotImplementedError

    @override
    def change_multi_assets_mode(self, *, multi_assets_margin: bool) -> None:
        if multi_assets_margin:
            raise NotImplementedError

    # -------------------------------- Account ------------------------------- #

    @override
    def account(self) -> Account:
        now: DateTime = tu.now()
        assets: dict[AssetName, AccountAsset] = {
            name: asset.dump(now) for name, asset in self._assets.items()
        }
        positions: dict[SymbolName, AccountPosition] = {}
        for symbol, position in self._positions.items():
            price: float = self.ticker_price(symbol).price
            dumped: AccountPosition = position.dump(price, now)
            info: ExchangeInfoSymbol = self.exchange_info().symbols[symbol]
            asset: AccountAsset = assets[info.quote_asset]
            asset.wallet_balance += dumped.isolated_wallet
            asset.unrealized_profit += dumped.unrealized_profit
            asset.margin_balance += dumped.isolated_margin
            assets[info.quote_asset] = asset
        return Account(
            assets=AccountAssetsDict(root=assets),
            positions=AccountPositionsDict(root=positions),
        )

    def commission_rate(self, symbol: SymbolName) -> CommissionRate:
        return CommissionRate(
            symbol=symbol, makerCommissionRate=0.0002, takerCommissionRate=0.0004
        )
