import collections
import logging
from collections import deque
from decimal import Decimal

import attrs
import pendulum
import polars as pl
from liblaf import cherries
from pendulum import DateTime, Duration

import qoc
import qoc.logging
from qoc.api.usds import ApiUsds, ApiUsdsOffline, ApiUsdsOnline
from qoc.api.usds.models import Account, MarginType, OrderResponse, OrderSide
from qoc.typing import SymbolName

logger = logging.getLogger(__name__)


@attrs.frozen
class Order:
    quantity: Decimal
    symbol: SymbolName
    time: DateTime
    direction: str  # 'BUY' or 'SELL'


@attrs.define
class Strategy(qoc.PersistableMixin):
    api: ApiUsds

    symbols: list[SymbolName] = attrs.field(factory=lambda: ["ETHUSDT"])

    # -------------------------------- Config -------------------------------- #

    past_window: Duration = attrs.field(factory=lambda: pendulum.duration(days=4))
    """过去窗口"""

    bullet_size: float = 50
    """单次下单资金 (USDT)"""

    max_holdings: int = 1
    """单标最大持仓 (单)"""

    # --------------------------------- State -------------------------------- #
    orders: collections.defaultdict[str, deque[Order]] = attrs.field(
        factory=lambda: collections.defaultdict(deque)
    )

    t: int = 0

    # Add history tracking fields
    asset_history: dict[str, list[float]] = attrs.field(
        factory=lambda: collections.defaultdict(list)
    )
    price_history: dict[str, list[float]] = attrs.field(
        factory=lambda: collections.defaultdict(list)
    )
    time_steps: list[int] = attrs.field(factory=list)

    volumes: collections.defaultdict[str, deque[float]] = attrs.field(
        factory=lambda: collections.defaultdict(deque)
    )
    """Historical prices for each symbol"""

    prices_times_volumes: collections.defaultdict[str, deque[float]] = attrs.field(
        factory=lambda: collections.defaultdict(deque)
    )
    """Historical price*volume products for each symbol"""

    vwaps: collections.defaultdict[str, deque[float]] = attrs.field(
        factory=lambda: collections.defaultdict(deque)
    )
    """Historical VWAP values for each symbol"""

    vwaps_deri_1: collections.defaultdict[str, deque[float]] = attrs.field(
        factory=lambda: collections.defaultdict(deque)
    )
    """First derivative of VWAP for each symbol"""

    vwaps_deri_2: collections.defaultdict[str, deque[float]] = attrs.field(
        factory=lambda: collections.defaultdict(deque)
    )
    """Second derivative of VWAP for each symbol"""

    vwaps_deri_3: collections.defaultdict[str, deque[float]] = attrs.field(
        factory=lambda: collections.defaultdict(deque)
    )
    """Third derivative of VWAP for each symbol"""

    past_window_length: int = attrs.field(default=4 * 24 * 60)

    # Track last logged date for daily logging
    last_logged_date: str = attrs.field(default="")
    """Last logged date for daily progress tracking"""

    def update_indicators(self, symbol: str, price: float, volume: float):
        """Update price and volume based indicators"""
        if volume == 0:
            volume = 1e-6  # Prevent division by zero
        self.volumes[symbol].append(volume)
        if len(self.volumes[symbol]) > self.past_window_length:
            self.volumes[symbol].popleft()

        self.prices_times_volumes[symbol].append(price * volume)
        if len(self.prices_times_volumes[symbol]) > self.past_window_length:
            self.prices_times_volumes[symbol].popleft()

        self.vwaps[symbol].append(
            sum(self.prices_times_volumes[symbol]) / sum(self.volumes[symbol])
        )
        if len(self.vwaps[symbol]) > self.past_window_length:
            self.vwaps[symbol].popleft()

        self.vwaps_deri_1[symbol].append(self.vwaps[symbol][-1] - self.vwaps[symbol][0])
        if len(self.vwaps_deri_1[symbol]) > self.past_window_length:
            self.vwaps_deri_1[symbol].popleft()

        self.vwaps_deri_2[symbol].append(
            self.vwaps_deri_1[symbol][-1] - self.vwaps_deri_1[symbol][0]
        )
        if len(self.vwaps_deri_2[symbol]) > self.past_window_length:
            self.vwaps_deri_2[symbol].popleft()

        self.vwaps_deri_3[symbol].append(
            self.vwaps_deri_2[symbol][-1] - self.vwaps_deri_2[symbol][0]
        )
        if len(self.vwaps_deri_3[symbol]) > self.past_window_length:
            self.vwaps_deri_3[symbol].popleft()

    def __attrs_post_init__(self) -> None:
        """在所有字段初始化后执行的自定义初始化逻辑"""
        print("Initializing StrategyTrend state...")

        # 可以在这里设置初始值或执行其他初始化逻辑
        for symbol in self.symbols:
            print(f"Initialized state for symbol: {symbol}")
            self.past_window_length = int(self.past_window.in_minutes())

            now: DateTime = qoc.now()

            klines: pl.DataFrame = self.api.klines(
                symbol, "1m", end_time=now, limit=self.past_window_length * 4
            )

            prices = klines["close"].to_list()
            volumes = klines["volume"].to_list()

            for i in range(len(prices)):
                self.update_indicators(symbol, prices[i], volumes[i])

        print(f"Strategy initialized with {len(self.symbols)} symbols: {self.symbols}")

    def init(self) -> None:
        try:
            self.api.change_multi_assets_mode(multi_assets_margin=False)
            self.api.change_position_mode(dual_side_position=False)
            for symbol in self.symbols:
                self.api.change_leverage(symbol, 1)
                self.api.change_margin_type(symbol, MarginType.ISOLATED)
        except Exception:
            logger.exception("")

    def step(self) -> None:
        now: DateTime = qoc.now()
        for symbol in self.symbols:
            orders = self.orders[symbol]
            klines: pl.DataFrame = self.api.klines(symbol, "1m", end_time=now, limit=2)
            price: float = klines["close"].last()  # pyright: ignore[reportAssignmentType]
            volume: float = klines["volume"].last()  # pyright: ignore[reportAssignmentType]
            self.update_indicators(symbol, price, volume)

            if (
                self.vwaps_deri_1[symbol][-1] > 0
                and self.vwaps_deri_2[symbol][-1] > 0
                # and self.vwaps_deri_3[symbol][-1] > 0
                and len(orders) < self.max_holdings
            ):
                quantity: float = self.bullet_size / price

                response: OrderResponse = self.api.order_market(
                    symbol, OrderSide.SELL, quantity=quantity
                )
                logger.warning("Placed SELL order: %s", response)
                self.orders[symbol].append(
                    Order(
                        quantity=abs(response.orig_qty),
                        symbol=symbol,
                        time=response.update_time,
                        direction="SELL",
                    )
                )

            if (
                self.vwaps_deri_1[symbol][-1] < 0
                and self.vwaps_deri_2[symbol][-1] < 0
                # and self.vwaps_deri_3[symbol][-1] < 0
                and len(orders) < self.max_holdings
            ):
                quantity: float = self.bullet_size / price

                response: OrderResponse = self.api.order_market(
                    symbol, OrderSide.BUY, quantity=quantity
                )
                logger.warning("Placed BUY order: %s", response)
                self.orders[symbol].append(
                    Order(
                        quantity=abs(response.orig_qty),
                        symbol=symbol,
                        time=response.update_time,
                        direction="BUY",
                    )
                )

            while (
                orders
                and (
                    not (
                        self.vwaps_deri_1[symbol][-1] > 0
                        and self.vwaps_deri_2[symbol][-1] > 0
                        # and self.vwaps_deri_3[symbol][-1] > 0
                    )
                )
                and orders[-1].direction == "SELL"
            ):
                order: Order = orders[0]
                response = self.api.order_market(
                    order.symbol, OrderSide.BUY, quantity=abs(order.quantity)
                )
                orders.popleft()
                logger.warning("Closed SELL order: %s", response)
            # self.orders[symbol] = orders

            while (
                orders
                and (
                    not (
                        self.vwaps_deri_1[symbol][-1] < 0
                        and self.vwaps_deri_2[symbol][-1] < 0
                        # and self.vwaps_deri_3[symbol][-1] < 0
                    )
                )
                and orders[-1].direction == "BUY"
            ):
                order: Order = orders[0]
                response = self.api.order_market(
                    order.symbol, OrderSide.SELL, quantity=abs(order.quantity)
                )
                orders.popleft()
                logger.warning("Closed BUY order: %s", response)
            self.orders[symbol] = orders

        self.t += 1

    def log_stats(self) -> None:
        clock: qoc.Clock = qoc.get_clock()
        now: DateTime = clock.now
        step: int = clock.step
        account: Account = self.api.account()
        asset_metrics: dict[str, dict[str, float]] = {}
        position_metrics: dict[str, dict[str, float]] = {}

        # Record current metrics
        for name, asset in account.assets.items():
            asset_metrics[name] = {"margin_balance": asset.margin_balance}
            # Store in history
            self.asset_history[name].append(asset.margin_balance)
            self.price_history[name].append(self.api.ticker_price("ETHUSDT").price)
        self.time_steps.append(self.t)

        for symbol, position in account.positions.items():
            position_metrics[symbol] = {
                "position_amt": position.position_amt,
                "isolated_margin": position.isolated_margin,
                "isolated_wallet": position.isolated_wallet,
            }

        if self.t % 288 == 0:
            import matplotlib.pyplot as plt

            # Create figure with 2 subplots
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1])

            # ===== First subplot: Asset balance and price =====
            # Plot asset balance on primary y-axis
            for asset_name, balance_history in self.asset_history.items():
                ax1.plot(
                    self.time_steps, balance_history, "-", label=f"{asset_name} Balance"
                )
            ax1.set_xlabel("Time Steps")
            ax1.set_ylabel("Balance (USDT)", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            # Create secondary y-axis and plot price
            ax2 = ax1.twinx()
            for asset_name, price_history in self.price_history.items():
                ax2.plot(
                    self.time_steps,
                    price_history,
                    "--",
                    color="tab:orange",
                    label=f"{asset_name} Price",
                )
            ax2.set_ylabel("Price (USDT)", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            ax1.set_title("Asset Balance and Price Over Time")
            ax1.grid(True)

            # ===== Second subplot: VWAP and derivatives =====
            for symbol in self.symbols:
                # Calculate the correct time steps for VWAP data
                # VWAP data starts from negative indices (historical data)
                vwap_length = len(self.vwaps[symbol])
                current_step = self.time_steps[-1]
                vwap_steps = list(
                    range(current_step - vwap_length + 1, current_step + 1)
                )

                ax3.plot(
                    vwap_steps, list(self.vwaps[symbol]), "-", label=f"{symbol} VWAP"
                )
                ax3.plot(
                    vwap_steps,
                    list(self.vwaps_deri_1[symbol]),
                    "--",
                    label=f"{symbol} VWAP Deri 1",
                )
                ax3.plot(
                    vwap_steps,
                    list(self.vwaps_deri_2[symbol]),
                    "-.",
                    label=f"{symbol} VWAP Deri 2",
                )

            ax3.set_xlabel("Time Steps")
            ax3.set_ylabel("Value")
            ax3.set_title("VWAP and Derivatives")
            ax3.legend(loc="upper left")
            ax3.grid(True)
            ax3.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

            plt.tight_layout()
            plt.savefig("examples/trend-usds/asset_metrics.png")
            plt.close()  # 关闭图表释放内存
        stats: dict[str, dict[str, float]] = {}
        for symbol in self.symbols:
            stats[symbol] = {
                "vwap": self.vwaps[symbol][-1],
                "vwap_deri_1": self.vwaps_deri_1[symbol][-1],
                "vwap_deri_2": self.vwaps_deri_2[symbol][-1],
                "vwap_deri_3": self.vwaps_deri_3[symbol][-1],
            }
        cherries.log_metrics(
            {
                "assets": asset_metrics,
                "open_orders": len(self.orders["ETHUSDT"]),
                "price": self.api.ticker_price("ETHUSDT").price,
                "time": now.timestamp(),
                "stats": stats,
                # "positions": position_metrics,
            },
            step=step,
        )


class Config(cherries.BaseConfig):
    online: bool = True
    group_key: str = "Trend USDS 2025-12-18"


def main(cfg: Config) -> None:
    qoc.logging.init()
    api: ApiUsds
    if cfg.online:
        qoc.set_clock(qoc.ClockOnline("1m"))
        api = ApiUsdsOnline()
    else:
        qoc.set_clock(qoc.ClockOffline("1m", start="2024-01-01", end="2025-11-20"))
        api = ApiUsdsOffline()
    strategy = Strategy(api=api)
    strategy.init()
    strategy.load_state()
    for _ in qoc.loop():
        try:
            strategy.step()
            strategy.dump_state()
            strategy.log_stats()
        except Exception:
            logger.exception("")


if __name__ == "__main__":
    cherries.main(main)
    # main(Config())
