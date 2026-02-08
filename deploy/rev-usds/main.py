import collections
from decimal import Decimal
from typing import Dict, List

import attrs
import matplotlib.pyplot as plt
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

    bullet_size: float = 200
    """单次下单资金 (USDT)"""

    past_threshold: float = -0.02
    """买入阈值 (跌幅)"""

    max_holdings: int = 1
    """单标最大持仓 (单)"""

    # --------------------------------- State -------------------------------- #
    orders: collections.defaultdict[str, collections.deque[Order]] = attrs.field(
        factory=lambda: collections.defaultdict(collections.deque)
    )
    
    # 新增：时间序列数据
    asset_time_series: List[Dict] = attrs.field(factory=list)
    """资产时间序列数据"""
    
    plot_counter: int = 0
    """绘图计数器"""

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
        
        # 收集资产数据
        total_balance = 0.0
        for name, asset in account.assets.items():
            balance = asset.margin_balance
            asset_metrics[name] = {"margin_balance": balance}
            total_balance += balance

        # 记录时间序列数据
        self.asset_time_series.append({
            "timestamp": now.timestamp(),
            "datetime": now,
            "step": step,
            "total_balance": total_balance,
            "asset_metrics": asset_metrics.copy(),
            "open_orders": len(self.orders["BTCUSDT"]),
            "total_positions": sum(len(orders) for orders in self.orders.values()),
            "price": self.api.ticker_price("BTCUSDT").price
        })

        # 每1000步创建一次PNL曲线图
        if step > 0 and step % 1000 == 0:
            self.create_pnl_chart()

        cherries.log_metrics(
            {
                "assets": asset_metrics,
                "open_orders": len(self.orders["BTCUSDT"]),
                "price": self.api.ticker_price("BTCUSDT").price,
                "time": now.timestamp(),
                "positions": sum(len(orders) for orders in self.orders.values()),
                "total_balance": total_balance,
            },
            step=step,
        )
        # print(f"[{now}] step={step} assets={asset_metrics} open_orders={len(self.orders['BTCUSDT'])} positions={sum(len(orders) for orders in self.orders.values())} total_balance={total_balance:.2f}")

    def create_pnl_chart(self) -> None:
        """创建PNL曲线图"""
        if len(self.asset_time_series) < 2:
            return
            
        self.plot_counter += 1
        
        # 提取数据
        datetimes = [point["datetime"] for point in self.asset_time_series]
        total_balances = [point["total_balance"] for point in self.asset_time_series]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制总余额曲线
        ax.plot(datetimes, total_balances, 'g-', linewidth=2, label='Total Balance')
        ax.set_ylabel('Total Balance (USDT)', color='green')
        ax.set_xlabel('Time')
        ax.tick_params(axis='y', labelcolor='green')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        ax.set_title(f'Total Balance - Step {self.asset_time_series[-1]["step"]} - Chart #{self.plot_counter}')
        
        # 旋转时间标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        
        # 保存图表
        filename = "balance_chart.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Balance chart saved: {filename}")
        print(f"Current Total Balance: {total_balances[-1]:.2f} USDT")


class Config(cherries.BaseConfig):
    online: bool = False

def main(cfg: Config) -> None:
    # cherries.log_param("group_key", "Rev USDS 2025-12-18")
    env.read_env(verbose=True, override=True)
    api: ApiUsds
    if cfg.online:
        qoc.set_clock(qoc.ClockOnline("5m"))
        api = ApiUsdsOnline()
    else:
        qoc.set_clock(qoc.ClockOffline("5m", start="2025-10-01", end="2026-01-03"))
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
    # cherries.main(main)
    main(Config())

# BINANCE_USDS_BASE_URL="https://fapi.binance.com" /opt/anaconda3/bin/python deploy/rev-usds/main.py