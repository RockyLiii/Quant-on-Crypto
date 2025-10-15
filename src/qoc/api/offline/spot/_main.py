from typing import override

import attrs
import pendulum
import polars as pl
from liblaf import grapes

import qoc.time_utils as tu
from qoc.api._abc import Api
from qoc.api.binance.spot import ApiBinanceSpot
from qoc.api.typing import (
    Account,
    Asset,
    Balance,
    CommissionRates,
    ExchangeInfo,
    ExchangeInfoSymbol,
    Interval,
    OrderResponseFill,
    OrderResponseFull,
    OrderSide,
    OrderSideLike,
    OrderType,
    Symbol,
)
from qoc.time_utils import DateTimeLike


@attrs.define
class ApiOfflineSpot(Api):
    _api: ApiBinanceSpot = attrs.field(factory=ApiBinanceSpot, kw_only=True)
    _balances: dict[Asset, Balance] = attrs.field(
        factory=lambda: {"USDT": Balance(asset="USDT", free=1000.0, locked=0.0)}
    )
    _commission_rates: CommissionRates = attrs.field(
        factory=lambda: CommissionRates(maker=0.0002, taker=0.0005, buyer=0.0, seller=0.0)
    )

    @override
    def exchange_info(
        self, symbol: Symbol | None = None, symbols: list[str] | None = None, **kwargs
    ) -> ExchangeInfo:
        return self._api.exchange_info(symbol=symbol, symbols=symbols, **kwargs)

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
        return self._api.klines(
            symbol,
            interval,
            startTime=startTime,
            endTime=endTime,
            limit=limit,
            **kwargs,
        )

    @override
    def order_market(
        self,
        symbol: Symbol,
        side: OrderSideLike,
        *,
        quantity: float | str | None = None,
        **kwargs,
    ) -> OrderResponseFull:
        side = OrderSide(side)
        if quantity is None:
            msg: str = "Quantity must be provided for market orders"
            raise ValueError(msg)
        info: ExchangeInfoSymbol = self.exchange_info(symbol=symbol).get_symbol(symbol)
        base: Balance = self._balances.get(
            info.base_asset, Balance(asset=info.base_asset, free=0.0, locked=0.0)
        )
        quote: Balance = self._balances.get(
            info.quote_asset, Balance(asset=info.quote_asset, free=0.0, locked=0.0)
        )
        if (lot_size := info.lot_size) is not None:
            quantity = lot_size.round(quantity)

            # ! temporary fix for floating point precision
            # ! We should migrate to Decimal in the future
            def round_balance(b: Balance) -> None:
                b.free = (
                    round(b.free / lot_size.step_size) + 1e-3
                ) * lot_size.step_size

            round_balance(base)
            round_balance(quote)
        quantity = float(quantity)
        price: float = self.klines(symbol, "1m")["close"].last()  # pyright: ignore[reportAssignmentType]
        cummulative_quote_qty: float = quantity * price
        commission: float
        match side:
            case OrderSide.BUY:
                commission = cummulative_quote_qty * self._commission_rates.taker
                cost: float = commission + cummulative_quote_qty
                if quote.free < cost:
                    msg = f"Insufficient balance for {quote}: {quote.free} < {cost}"
                    raise ValueError(msg)
                quote.free -= cost
                base.free += quantity
            case OrderSide.SELL:
                if base.free < quantity:
                    msg = f"Insufficient balance for {base}: {base.free} < {quantity}"
                    raise ValueError(msg)
                commission = quantity * self._commission_rates.maker
                base.free -= quantity
                quote.free += cummulative_quote_qty - commission
            case _:
                raise grapes.MatchError(side, OrderSide)
        self._balances[base.asset] = base
        self._balances[quote.asset] = quote
        now: pendulum.DateTime = tu.now()
        return OrderResponseFull(
            symbol=symbol,
            client_order_id="",
            transact_time=now,  # pyright: ignore[reportArgumentType]
            price=0.0,
            orig_qty=quantity,
            executed_qty=quantity,
            orig_quote_order_qty=0.0,
            cummulative_quote_qty=cummulative_quote_qty,
            type=OrderType.MARKET,
            side=side,
            working_time=now,  # pyright: ignore[reportArgumentType]
            fills=[
                OrderResponseFill(
                    price=price,
                    qty=quantity,
                    commission=commission,
                    commission_asset=info.quote_asset,
                    trade_id=0,
                )
            ],
        )

    @override
    def account(self, *, omitZeroBalances: bool = True, **kwargs) -> Account:
        balances: list[Balance] = list(self._balances.values())
        if omitZeroBalances:
            balances = [b for b in balances if b.free > 0.0 or b.locked > 0.0]
        return Account(commission_rates=self._commission_rates, balances=balances)
