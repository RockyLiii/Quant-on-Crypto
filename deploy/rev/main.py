import collections
from collections import deque

import attrs
import pendulum
import polars as pl
from liblaf import cherries
from loguru import logger
from pendulum import DateTime, Duration

import qoc
from qoc.api.typing import Balance, OrderResponseFull, OrderSide, Symbol


class Config(cherries.BaseConfig):
    online: bool = False


@attrs.frozen
class Order:
    quantity: float
    symbol: Symbol
    time: DateTime


@attrs.define
class StrategyRev:
    symbols: list[Symbol] = attrs.field(factory=lambda: ["BTCUSDT"])

    # -------------------------------- Config -------------------------------- #

    past_window: Duration = attrs.field(factory=lambda: pendulum.duration(hours=5))
    """过去窗口"""

    future_end: Duration = attrs.field(factory=lambda: pendulum.duration(hours=30))
    """持有期"""

    bullet_size: float = 25
    """单次下单资金 (USDT)"""

    past_threshold: float = -0.02
    """买入阈值 (跌幅)"""

    max_holdings: int = 5
    """单标最大持仓 (单)"""

    # --------------------------------- State -------------------------------- #
    orders: collections.defaultdict[str, deque[Order]] = attrs.field(
        factory=lambda: collections.defaultdict(deque)
    )

    def step(self, api: qoc.Api) -> None:
        now: DateTime = qoc.now()
        for symbol in self.symbols:
            orders: deque[Order] = self.orders[symbol]
            klines: pl.DataFrame = api.klines(
                symbol, "1m", endTime=now - self.past_window, limit=1
            )
            past_price: float = klines["close"].last()  # pyright: ignore[reportAssignmentType]
            price: float = api.price(symbol, interval="1m")
            past_growth: float = (price - past_price) / past_price

            logger.debug(
                "{} past growth: {:.4f}, now holdings: {}",
                symbol,
                past_growth,
                len(orders),
            )
            if past_growth < self.past_threshold and len(orders) < self.max_holdings:
                quantity: float = self.bullet_size / price
                response: OrderResponseFull = api.order_market(
                    symbol, OrderSide.BUY, quantity=quantity
                )
                self.orders[symbol].append(
                    Order(
                        quantity=response.executed_qty,
                        symbol=symbol,
                        time=response.transact_time,
                    )
                )
            while orders and orders[0].time + self.future_end < now:
                order: Order = orders.popleft()
                api.order_market(order.symbol, OrderSide.SELL, quantity=order.quantity)
            self.orders[symbol] = orders

    def dump(self, api: qoc.Api) -> None:
        clock: qoc.Clock = qoc.get_clock()
        cherries.log_metric("time", clock.now.timestamp(), step=clock.step)

        for symbol in self.symbols:
            orders: deque[Order] = self.orders[symbol]
            cherries.log_metric(f"holdings/{symbol}", len(orders), step=clock.step)

        total_value: float = 0.0
        balances: list[Balance] = api.account(omitZeroBalances=True).balances
        for balance in balances:
            if balance.asset == "USDT":
                total_value += balance.free + balance.locked
            else:
                symbol: Symbol = balance.asset + "USDT"
                price: float = api.price(symbol, interval="1m")
                total_value += (balance.free + balance.locked) * price
        cherries.log_metric("Total Value (USDT)", total_value, step=clock.step)


def main(cfg: Config) -> None:
    if cfg.online:
        qoc.set_clock(qoc.ClockOnline(interval="5m"))
    else:
        qoc.set_clock(
            qoc.ClockOffline(interval="5m", start="2024-01-01", end="2025-01-01")
        )
    api: qoc.Api = qoc.ApiBinanceSpot() if cfg.online else qoc.ApiOfflineSpot()
    strategy = StrategyRev()

    for _ in qoc.loop():
        try:
            strategy.step(api=api)
            strategy.dump(api=api)
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logger.exception("error")


if __name__ == "__main__":
    cherries.run(main)
