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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@attrs.frozen
class Order:
    price_l: float
    quantity_l: Decimal
    symbol_l: SymbolName
    time_l: DateTime

    price_s: float
    quantity_s: Decimal
    symbol_s: SymbolName
    time_s: DateTime


@attrs.define
class Strategy(qoc.PersistableMixin):
    api: ApiUsds

    symbols: list[SymbolName] = attrs.field(
        factory=lambda: [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "ADAUSDT",
            "DOGEUSDT",
            "AVAXUSDT",
            "LINKUSDT",
            "TRXUSDT",
            "DOTUSDT",
            "MATICUSDT",
            "LTCUSDT",
            "UNIUSDT",
            "ATOMUSDT",
            "ETCUSDT",
            "ICPUSDT",
            "APTUSDT",
            "FILUSDT",
            # "NEARUSDT",
            # "HBARUSDT",
            # "INJUSDT",
            # "OPUSDT",
            # "ARBUSDT",
            # "SUIUSDT",
            # "SEIUSDT",
            # "RENDERUSDT",
            # "TIAUSDT",
            # "FTMUSDT",
            # "EGLDUSDT",
            # "AAVEUSDT",
            # "GALAUSDT",
            # "IMXUSDT",
            # "PEPEUSDT",
            # "SHIBUSDT",
            # "FLOKIUSDT",
            # "BONKUSDT",
            # "WIFUSDT",
            # "PENGUUSDT",
            # "TRUMPUSDT",
            # "POLUSDT",
            # "ENSUSDT",
            # "JUPUSDT",
            # "PYTHUSDT",
            # "ORDIUSDT",
            # "SATOSHIUSDT",
            # "SATSUSDT",
            # "MOVRUSDT",
            # "DYMUSDT",
            # "NOTUSDT",
            # "MAGAUSDT",
            # "SPXUSDT",
        ]
    )
    # -------------------------------- Config -------------------------------- #

    forward_window: Duration = attrs.field(factory=lambda: pendulum.duration(minutes=6))

    back_window_in_mins: int = 12

    back_window: Duration = attrs.field(factory=lambda: pendulum.duration(minutes=12))

    bullet_size: float = 5000

    stop_loss: float = 99999  # 5%
    take_profit: float = 99999  # 10%

    # --------------------------------- State -------------------------------- #
    orders: collections.defaultdict[str, deque[Order]] = attrs.field(
        factory=lambda: collections.defaultdict(deque)
    )

    temp: float = 0.0
    temps: list[float] = attrs.field(factory=list)

    t: int = 0

    # Add history tracking fields
    asset_history: dict[str, list[float]] = attrs.field(
        factory=lambda: collections.defaultdict(list)
    )
    price_history: dict[str, list[float]] = attrs.field(
        factory=lambda: collections.defaultdict(list)
    )

    # mkt_idx_1: deque = attrs.field(factory=lambda: deque(maxlen=200))
    # Price and volume history
    # coin_closes: dict[str, deque] = attrs.field(factory=dict)
    # coin_volumes: dict[str, deque] = attrs.field(factory=dict)

    # Statistical calculation deques
    xy_1: dict[str, deque] = attrs.field(factory=dict)
    xx_1: dict[str, deque] = attrs.field(factory=dict)
    yy_1: dict[str, deque] = attrs.field(factory=dict)
    x_1: dict[str, deque] = attrs.field(factory=dict)
    y_1: dict[str, deque] = attrs.field(factory=dict)

    # Sum variables for efficient rolling calculations
    xy_1_sum: dict[str, float] = attrs.field(factory=dict)
    xx_1_sum: dict[str, float] = attrs.field(factory=dict)
    yy_1_sum: dict[str, float] = attrs.field(factory=dict)
    x_1_sum: dict[str, float] = attrs.field(factory=dict)
    y_1_sum: dict[str, float] = attrs.field(factory=dict)

    # Statistics tracking
    revenues: list[float] = attrs.field(factory=list)
    # coin_revenues: dict[str, list[float]] = attrs.field(factory=dict)

    def update_indicators(
        self,
        symbol: str,
        price: float,
        volume: float,
        btc_price: float,
        btc_volume: float,
    ):
        """Update price and volume based indicators"""

        if len(self.xy_1[symbol]) >= self.back_window_in_mins:
            self.xy_1_sum[symbol] -= self.xy_1[symbol][0]
            self.xx_1_sum[symbol] -= self.xx_1[symbol][0]
            self.yy_1_sum[symbol] -= self.yy_1[symbol][0]
            self.x_1_sum[symbol] -= self.x_1[symbol][0]
            self.y_1_sum[symbol] -= self.y_1[symbol][0]

            self.xy_1[symbol].popleft()
            self.xx_1[symbol].popleft()
            self.yy_1[symbol].popleft()
            self.x_1[symbol].popleft()
            self.y_1[symbol].popleft()

        # Add new values
        xy_1_val = price * btc_price
        xx_1_val = btc_price**2
        yy_1_val = price**2
        x_1_val = btc_price
        y_1_val = price

        self.xy_1[symbol].append(xy_1_val)
        self.xx_1[symbol].append(xx_1_val)
        self.yy_1[symbol].append(yy_1_val)
        self.x_1[symbol].append(x_1_val)
        self.y_1[symbol].append(y_1_val)

        # Update sums
        self.xy_1_sum[symbol] += xy_1_val
        self.xx_1_sum[symbol] += xx_1_val
        self.yy_1_sum[symbol] += yy_1_val
        self.x_1_sum[symbol] += x_1_val
        self.y_1_sum[symbol] += y_1_val

    def __attrs_post_init__(self) -> None:
        """Initialize symbol-based dictionaries after object creation"""
        # Initialize all symbol-based dictionaries first
        for symbol in self.symbols:
            # Statistical calculation deques
            self.xy_1[symbol] = deque(maxlen=self.back_window_in_mins)
            self.xx_1[symbol] = deque(maxlen=self.back_window_in_mins)
            self.yy_1[symbol] = deque(maxlen=self.back_window_in_mins)
            self.x_1[symbol] = deque(maxlen=self.back_window_in_mins)
            self.y_1[symbol] = deque(maxlen=self.back_window_in_mins)

            # Sum variables
            self.xy_1_sum[symbol] = 0.0
            self.xx_1_sum[symbol] = 0.0
            self.yy_1_sum[symbol] = 0.0
            self.x_1_sum[symbol] = 0.0
            self.y_1_sum[symbol] = 0.0

        now: DateTime = qoc.now()

        btc_klines: pl.DataFrame = self.api.klines(
            "BTCUSDT", "1m", end_time=now, limit=2 * self.back_window_in_mins
        )

        btc_prices = btc_klines["close"].to_list()
        btc_volumes = btc_klines["volume"].to_list()

        # Keep track of symbols that fail to load data
        symbols_to_remove = []

        for symbol in self.symbols:
            try:
                klines: pl.DataFrame = self.api.klines(
                    symbol, "1m", end_time=now, limit=2 * self.back_window_in_mins
                )

                prices = klines["close"].to_list()
                volumes = klines["volume"].to_list()

                for i in range(len(prices)):
                    self.update_indicators(
                        symbol, prices[i], volumes[i], btc_prices[i], btc_volumes[i]
                    )

            except Exception as e:
                logger.warning(
                    "Failed to load data for %s: %s. Removing from symbols.", symbol, e
                )
                symbols_to_remove.append(symbol)

        # Remove failed symbols
        for symbol in symbols_to_remove:
            self.symbols.remove(symbol)
            # Clean up initialized dictionaries for removed symbols
            if symbol in self.xy_1:
                del self.xy_1[symbol]
                del self.xx_1[symbol]
                del self.yy_1[symbol]
                del self.x_1[symbol]
                del self.y_1[symbol]
                del self.xy_1_sum[symbol]
                del self.xx_1_sum[symbol]
                del self.yy_1_sum[symbol]
                del self.x_1_sum[symbol]
                del self.y_1_sum[symbol]

        print(f"Strategy initialized with {len(self.symbols)} symbols: {self.symbols}")
        if symbols_to_remove:
            print(
                f"Removed {len(symbols_to_remove)} symbols due to data loading failures: {symbols_to_remove}"
            )

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

        klines: pl.DataFrame = self.api.klines("BTCUSDT", "1m", end_time=now, limit=2)

        btc_price: float = klines["close"].last()  # pyright: ignore[reportAssignmentType]
        btc_volume: float = klines["volume"].last()  # pyright: ignore[reportAssignmentType]

        coef_records = {}

        for symbol in self.symbols:
            orders = self.orders[symbol]
            klines: pl.DataFrame = self.api.klines(symbol, "1m", end_time=now, limit=2)
            price: float = klines["close"].last()  # pyright: ignore[reportAssignmentType]
            volume: float = klines["volume"].last()  # pyright: ignore[reportAssignmentType]
            self.update_indicators(symbol, price, volume, btc_price, btc_volume)

            # With intercept
            # beta = (self.back_window_in_mins * self.xy_1_sum[symbol] - self.x_1_sum[symbol]*self.y_1_sum[symbol]) / (self.back_window_in_mins * self.xx_1_sum[symbol] - self.x_1_sum[symbol]**2)

            # alpha = (self.y_1_sum[symbol] - beta * self.x_1_sum[symbol]) / self.back_window_in_mins

            # residual = price - (beta * btc_price + alpha)

            beta = self.xy_1_sum[symbol] / self.xx_1_sum[symbol]

            residual = price - (beta * btc_price)

            coef_records[symbol] = {
                "Close": price,
                "beta": beta,
                # 'alpha': alpha,
                "residual": residual / beta if beta != 0 else 0,
            }

        coef_df = pd.DataFrame.from_dict(coef_records, orient="index")  # index=coin

        # Select coins based on residual values
        l_coin = coef_df.sort_values(by="residual", ascending=True).index[0]
        s_coin = coef_df.sort_values(by="residual", ascending=False).index[0]

        price_l_now = coef_records[l_coin]["Close"]
        price_s_now = coef_records[s_coin]["Close"]

        beta_l = coef_records[l_coin]["beta"]
        beta_s = coef_records[s_coin]["beta"]

        q_l = beta_s * self.bullet_size / (beta_l * price_s_now + beta_s * price_l_now)
        q_s = beta_l * self.bullet_size / (beta_l * price_s_now + beta_s * price_l_now)

        if q_l > 0 and q_s > 0:
            # long
            response_l: OrderResponse = self.api.order_market(
                l_coin, OrderSide.BUY, quantity=q_l
            )
            # short
            response_s: OrderResponse = self.api.order_market(
                s_coin, OrderSide.SELL, quantity=q_s
            )

            self.temp += price_l_now * (
                q_l - float(abs(response_l.orig_qty))
            ) + price_s_now * (float(abs(response_s.orig_qty)) - q_s)

            # logger.warning("Placed BUY order: %s", response)
            self.orders[l_coin].append(
                Order(
                    price_l=price_l_now,
                    quantity_l=abs(response_l.orig_qty),
                    symbol_l=l_coin,
                    time_l=response_l.update_time,
                    price_s=price_s_now,
                    quantity_s=abs(response_s.orig_qty),
                    symbol_s=s_coin,
                    time_s=response_s.update_time,
                )
            )

        for symbol in self.symbols:
            orders = self.orders[symbol]
            orders_to_process = list(orders)

            for order in reversed(orders_to_process):
                symbol_l = order.symbol_l
                symbol_s = order.symbol_s

                price_l_now = (
                    self.price_history[symbol_l][-1]
                    if self.price_history[symbol_l]
                    else self.api.ticker_price(symbol_l).price
                )
                price_s_now = (
                    self.price_history[symbol_s][-1]
                    if self.price_history[symbol_s]
                    else self.api.ticker_price(symbol_s).price
                )

                price_l_his = order.price_l
                price_s_his = order.price_s

                order_profit = -(price_s_now - price_s_his) * float(
                    order.quantity_s
                ) + (price_l_now - price_l_his) * float(order.quantity_l)

                if (
                    (
                        order.time_l + self.forward_window < now
                        and order.time_s + self.forward_window < now
                    )
                    or order_profit / self.bullet_size <= -self.stop_loss
                    or order_profit / self.bullet_size >= self.take_profit
                ):
                    # Close long position
                    response_l = self.api.order_market(
                        order.symbol_l, OrderSide.SELL, quantity=abs(order.quantity_l)
                    )
                    # logger.warning("Closed LONG order: %s", response)

                    # Close short position
                    response_s = self.api.order_market(
                        order.symbol_s, OrderSide.BUY, quantity=abs(order.quantity_s)
                    )
                    # logger.warning("Closed SHORT order: %s", response)

                    self.orders[symbol].remove(order)

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
            for symbol in self.symbols:
                if name in symbol:
                    price = self.api.ticker_price(symbol).price
                    self.price_history[symbol].append(price)
        # self.time_steps.append(self.t)

        for symbol, position in account.positions.items():
            position_metrics[symbol] = {
                "position_amt": position.position_amt,
                "isolated_margin": position.isolated_margin,
                "isolated_wallet": position.isolated_wallet,
            }
        self.temps.append(self.temp)

        if self.t % 60 == 0:
            import matplotlib.pyplot as plt

            # Create figure with 2 subplots
            fig, ax = plt.subplots(figsize=(12, 6))

            # ===== First subplot: Asset balance and price =====
            # Plot asset balance on primary y-axis
            for asset_name, balance_history in self.asset_history.items():
                ax.plot(
                    range(len(balance_history)),
                    balance_history,
                    "-",
                    label=f"{asset_name} Balance",
                )
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Balance (USDT)", color="tab:blue")
            ax.tick_params(axis="y", labelcolor="tab:blue")
            ax.legend()

            plt.savefig("examples/stat_arbi-usds/asset_metrics.png")
            plt.close()  # 关闭图表释放内存

            # plot temps
            plt.figure(figsize=(10, 5))
            plt.plot(
                range(len(self.temps)), self.temps, label="Temp Value", color="orange"
            )
            plt.xlabel("Time Steps")
            plt.ylabel("Temp Value")
            plt.title("Temp Value Over Time")
            plt.legend()
            plt.savefig("examples/stat_arbi-usds/temp_values.png")
            plt.close()

        # cherries.log_metrics(
        #     {
        #         "assets": asset_metrics,
        #         "open_orders": len(self.orders["ETHUSDT"]),
        #         "price": self.api.ticker_price("ETHUSDT").price,
        #         "time": now.timestamp(),
        #         # "stats": stats,
        #         # "positions": position_metrics,
        #     },
        #     step=step,
        # )


class Config(cherries.BaseConfig):
    online: bool = False


def main(cfg: Config) -> None:
    # cherries.log_param("group_key", "Trend USDS 2026-01-06 ETH")
    # qoc.logging.init()
    api: ApiUsds
    if cfg.online:
        qoc.set_clock(qoc.ClockOnline("1m"))
        api = ApiUsdsOnline()
    else:
        # qoc.set_clock(qoc.ClockOffline("1m", start="2025-10-15", end="2025-12-21"))
        qoc.set_clock(qoc.ClockOffline("1m", start="2025-10-01", end="2026-01-03"))

        api = ApiUsdsOffline()
    strategy = Strategy(api=api)
    strategy.init()
    strategy.load_state()
    for _ in qoc.loop():
        try:
            strategy.log_stats()
            strategy.step()
            if cfg.online:
                strategy.dump_state()
        except Exception:
            logger.exception("")


if __name__ == "__main__":
    # cherries.main(main)
    main(Config())

# BINANCE_USDS_BASE_URL="https://fapi.binance.com" /opt/anaconda3/bin/python examples/stat_arbi-usds/main.py
