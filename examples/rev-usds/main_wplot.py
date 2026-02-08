import collections
from decimal import Decimal

import attrs
import pendulum
import polars as pl
from liblaf import cherries
from pendulum import DateTime, Duration

import qoc
from qoc.api.usds import ApiUsds, ApiUsdsOffline, ApiUsdsOnline
from qoc.api.usds.models import Account, MarginType, OrderResponse, OrderSide
from qoc.typing import SymbolName

import pandas as pd  # For data handling in plotting


@attrs.frozen
class Order:
    quantity: Decimal
    symbol: SymbolName
    time: DateTime
    direction: str  # 'BUY' or 'SELL'


@attrs.define
class Strategy:
    api: ApiUsds

    symbols: list[SymbolName] = attrs.field(factory=lambda: ["BTCUSDT"])

    # -------------------------------- Config -------------------------------- #

    past_window: Duration = attrs.field(factory=lambda: pendulum.duration(hours=5))
    """过去窗口"""

    future_end: Duration = attrs.field(factory=lambda: pendulum.duration(hours=30))
    """持有期"""

    bullet_size: float = 10
    """单次下单资金 (USDT)"""

    past_threshold: float = -0.02
    """买入阈值 (跌幅)"""

    max_holdings: int = 5
    """单标最大持仓 (单)"""

    # --------------------------------- State -------------------------------- #
    orders: collections.defaultdict[str, collections.deque[Order]] = attrs.field(
        factory=lambda: collections.defaultdict(collections.deque)
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
                        direction='BUY',
                    )
                )
            # if past_growth >- self.past_threshold/2 and len(orders) < self.max_holdings:
            #     quantity: float = self.bullet_size / price
            #     response: OrderResponse = self.api.order_market(
            #         symbol, OrderSide.SELL, quantity=quantity
            #     )
            #     self.orders[symbol].append(
            #         Order(
            #             quantity=response.orig_qty,
            #             symbol=symbol,
            #             time=response.update_time,
            #             direction='SELL',
            #         )
            #     )
            while orders and orders[0].time + self.future_end < now:
                order: Order = orders.popleft()
                if order.direction == 'BUY':
                    self.api.order_market(
                        order.symbol, OrderSide.SELL, quantity=order.quantity
                    )
                else:
                    self.api.order_market(
                        order.symbol, OrderSide.BUY, quantity=order.quantity
                    )
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
            self.price_history[name].append(
                self.api.ticker_price("BTCUSDT").price
            )
        self.time_steps.append(self.t)

        for symbol, position in account.positions.items():
            position_metrics[symbol] = {
                "position_amt": position.position_amt,
                "isolated_margin": position.isolated_margin,
                "isolated_wallet": position.isolated_wallet,
            }
        cherries.log_metrics(
            {
                "assets": asset_metrics,
                "open_orders": len(self.orders["BTCUSDT"]),
                "price": self.api.ticker_price("BTCUSDT").price,
                "time": now.timestamp(),
                # "positions": position_metrics,
            },
            step=step,
        )
        if self.t % 288 == 0:
            import matplotlib.pyplot as plt
            
            # Create figure and primary axis
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Plot asset balance on primary y-axis
            for asset_name, balance_history in self.asset_history.items():
                ax1.plot(
                    self.time_steps,
                    balance_history,
                    '-',
                    label=f"{asset_name} Balance"
                )
                balance_history_pd = pd.DataFrame({
                    'Time Steps': self.time_steps,
                    'Balance': balance_history
                })
                balance_history_pd.to_csv('examples/rev-usds/asset_balance_btc.csv')
            ax1.set_xlabel("Time Steps")
            ax1.set_ylabel("Balance (USDT)", color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            # Create secondary y-axis and plot price
            ax2 = ax1.twinx()
            for asset_name, price_history in self.price_history.items():
                ax2.plot(
                    self.time_steps,
                    price_history,
                    '--',
                    color='tab:orange',
                    label=f"{asset_name} Price"
                )
            ax2.set_ylabel("Price (USDT)", color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            
            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.title("Asset Balance and Price Over Time")
            plt.grid(True)
            plt.savefig('examples/rev-usds/asset_metrics.png')
            plt.close()  # 关闭图表释放内存

class Config(cherries.BaseConfig):
    online: bool = False


def main(cfg: Config) -> None:
    api: ApiUsds
    if cfg.online:
        qoc.set_clock(qoc.ClockOnline("1m"))
        api = ApiUsdsOnline()
    else:
        qoc.set_clock(qoc.ClockOffline("1m", start="2025-10-01", end="2026-01-03"))
        api = ApiUsdsOffline()
    strategy = Strategy(api=api)
    strategy.init()
    for _ in qoc.loop():
        strategy.step()
        strategy.log_stats()


if __name__ == "__main__":
    # cherries.run(main)
    main(Config())



# BINANCE_USDS_BASE_URL="https://fapi.binance.com" /opt/anaconda3/bin/python examples/rev-usds/main_wplot.py