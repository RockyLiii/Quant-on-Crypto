import collections
from decimal import Decimal

import attrs
import pendulum
import polars as pl
from environs import env
from liblaf import cherries
from loguru import logger
from pendulum import DateTime, Duration

import qoc
from qoc.api.usds import ApiUsds, ApiUsdsOffline, ApiUsdsOnline
from qoc.api.usds.models import Account, MarginType, OrderResponse, OrderSide
from qoc.typing import SymbolName


@attrs.frozen
class Order:
    quantity: Decimal
    symbol: SymbolName
    time: DateTime


@attrs.define
class Strategy(qoc.PersistableMixin):
    api: ApiUsds = attrs.field(metadata={"persist": False})

    symbols: list[SymbolName] = attrs.field(factory=lambda: ["BTCUSDT"])

    # -------------------------------- Config -------------------------------- #

    past_window: Duration = attrs.field(factory=lambda: pendulum.duration(hours=5))
    """过去窗口"""

    future_end: Duration = attrs.field(factory=lambda: pendulum.duration(hours=30))
    """持有期"""

    bullet_size: float = 100
    """单次下单资金 (USDT)"""

    past_threshold: float = -0.02
    """买入阈值 (跌幅)"""

    max_holdings: int = 5
    """单标最大持仓 (单)"""

    # --------------------------------- State -------------------------------- #
    orders: collections.defaultdict[str, collections.deque[Order]] = attrs.field(
        factory=lambda: collections.defaultdict(collections.deque)
    )

    def init(self) -> None:
        self.api.change_multi_assets_mode(multi_assets_margin=False)
        self.api.change_position_mode(dual_side_position=False)
        for symbol in self.symbols:
            self.api.change_leverage(symbol, 1)
            self.api.change_margin_type(symbol, MarginType.ISOLATED)

    def step(self) -> None:
        now: DateTime = qoc.now()
        for symbol in self.symbols:
            orders: collections.deque[Order] = self.orders[symbol]
            klines: pl.DataFrame = self.api.klines(
                symbol, "1m", end_time=now - self.past_window, limit=1
            )
            past_price: float = klines["close"].last()  # pyright: ignore[reportAssignmentType]
            price: float = self.api.ticker_price(symbol).price
            past_growth: float = (price - past_price) / past_price

            if past_growth < self.past_threshold and len(orders) < self.max_holdings:
                quantity: float = self.bullet_size / price
                response: OrderResponse = self.api.order_market(
                    symbol, OrderSide.BUY, quantity=quantity
                )
                self.orders[symbol].append(
                    Order(
                        quantity=response.orig_qty,
                        symbol=symbol,
                        time=response.update_time,
                    )
                )
            while orders and orders[0].time + self.future_end < now:
                order: Order = orders.popleft()
                self.api.order_market(
                    order.symbol, OrderSide.SELL, quantity=order.quantity
                )
            self.orders[symbol] = orders

    def log_stats(self) -> None:
        clock: qoc.Clock = qoc.get_clock()
        now: DateTime = clock.now
        step: int = clock.step
        account: Account = self.api.account()
        asset_metrics: dict[str, dict[str, float]] = {}
        # position_metrics: dict[str, dict[str, float]] = {}
        for name, asset in account.assets.items():
            asset_metrics[name] = {"margin_balance": asset.margin_balance}
        # for symbol, position in account.positions.items():
        #     position_metrics[symbol] = {
        #         "position_amt": position.position_amt,
        #         "isolated_margin": position.isolated_margin,
        #         "isolated_wallet": position.isolated_wallet,
        #     }
        cherries.log_metrics(
            {
                "assets": asset_metrics,
                "open_orders": len(self.orders["BTCUSDT"]),
                "price": self.api.ticker_price("BTCUSDT").price,
                "time": now.timestamp(),
                "positions": sum(len(orders) for orders in self.orders.values()),
                # "positions": position_metrics,
            },
            step=step,
        )


class Config(cherries.BaseConfig):
    online: bool = env.bool("ONLINE", False)


def main(cfg: Config) -> None:
    env.read_env(verbose=True, override=True)
    api: ApiUsds
    if cfg.online:
        qoc.set_clock(qoc.ClockOnline("5m"))
        api = ApiUsdsOnline()
    else:
        qoc.set_clock(qoc.ClockOffline("5m", start="2024-01-01", end="2025-01-01"))
        api = ApiUsdsOffline()
    strategy = Strategy(api=api)
    strategy.init()
    strategy.load_state()
    for _ in qoc.loop():
        try:
            strategy.step()
        except Exception as err:  # noqa: BLE001
            logger.exception(err)
        strategy.dump_state()
        strategy.log_stats()


if __name__ == "__main__":
    cherries.run(main)
