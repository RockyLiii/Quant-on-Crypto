import collections
import os
from collections import deque
from pathlib import Path

import attrs
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pendulum
import polars as pl
from liblaf import cherries
from loguru import logger
from pendulum import DateTime, Duration
from scipy.optimize import linprog

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
class TimeSeriesData:
    """Flexible time series storage structure"""

    data: dict[str, list[tuple[DateTime, float]]] = attrs.field(factory=dict)

    def add_metric(self, metric_name: str, timestamp: DateTime, value: float) -> None:
        """Add a metric value at a specific timestamp"""
        if metric_name not in self.data:
            self.data[metric_name] = []
        self.data[metric_name].append((timestamp, value))

    def get_metric(self, metric_name: str) -> list[tuple[DateTime, float]]:
        """Get time series data for a specific metric"""
        return self.data.get(metric_name, [])

    def to_dataframe(self, metric_name: str) -> pl.DataFrame:
        """Convert a metric's time series to Polars DataFrame"""
        if metric_name not in self.data:
            return pl.DataFrame()

        timestamps, values = zip(*self.data[metric_name], strict=False)
        return pl.DataFrame({"timestamp": timestamps, "value": values})

    def get_all_metrics(self) -> list[str]:
        """Get list of all available metrics"""
        return list(self.data.keys())


@attrs.define
class StrategyRev:
    symbols: list[Symbol] = attrs.field(
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
            "NEARUSDT",
            "HBARUSDT",
            "INJUSDT",
            "OPUSDT",
            "ARBUSDT",
            "SUIUSDT",
            "SEIUSDT",
            "RENDERUSDT",
            "TIAUSDT",
            "FTMUSDT",
            "EGLDUSDT",
            "AAVEUSDT",
            "GALAUSDT",
            "IMXUSDT",
            "PEPEUSDT",
            "SHIBUSDT",
            "FLOKIUSDT",
            "BONKUSDT",
            "WIFUSDT",
            "PENGUUSDT",
            "TRUMPUSDT",
            "POLUSDT",
            "ENSUSDT",
            "JUPUSDT",
            "PYTHUSDT",
            "ORDIUSDT",
            "SATOSHIUSDT",
            "SATSUSDT",
            "MOVRUSDT",
            "DYMUSDT",
            "NOTUSDT",
            "MAGAUSDT",
            "SPXUSDT",
        ]
    )

    # -------------------------------- Config -------------------------------- #

    back_window: Duration = attrs.field(factory=lambda: pendulum.duration(minutes=200))
    """过去窗口"""

    forward_window: Duration = attrs.field(
        factory=lambda: pendulum.duration(minutes=200)
    )
    """持有期"""

    bullet_size: float = 100
    """单次下单资金 (USDT)"""

    max_holdings: int = 200
    """单标最大持仓 (单)"""

    stop_loss_ratio: float = 0.02
    """止损比例"""

    take_profit_ratio: float = 0.02
    """止盈比例"""

    # --------------------------------- State -------------------------------- #
    orders: collections.defaultdict[str, deque[Order]] = attrs.field(
        factory=lambda: collections.defaultdict(deque)
    )

    # Current step counter and data window
    current_step: int = 0
    back_window_size: int = 200
    forward_window_size: int = 200

    # Market index deques
    mkt_idx_1: deque = attrs.field(factory=lambda: deque(maxlen=200))
    mkt_idx_2: deque = attrs.field(factory=lambda: deque(maxlen=200))
    mkt_idx_3: deque = attrs.field(factory=lambda: deque(maxlen=200))

    # Price and volume history
    coin_closes: dict[str, deque] = attrs.field(factory=dict)
    coin_volumes: dict[str, deque] = attrs.field(factory=dict)
    btc_close: deque = attrs.field(factory=lambda: deque(maxlen=200))
    btc_volume: deque = attrs.field(factory=lambda: deque(maxlen=200))

    # Statistical calculation deques
    xy_1: dict[str, deque] = attrs.field(factory=dict)
    xx_1: dict[str, deque] = attrs.field(factory=dict)
    yy_1: dict[str, deque] = attrs.field(factory=dict)
    x_1: dict[str, deque] = attrs.field(factory=dict)
    y_1: dict[str, deque] = attrs.field(factory=dict)
    xy_2: dict[str, deque] = attrs.field(factory=dict)
    xx_2: dict[str, deque] = attrs.field(factory=dict)
    yy_2: dict[str, deque] = attrs.field(factory=dict)
    x_2: dict[str, deque] = attrs.field(factory=dict)
    y_2: dict[str, deque] = attrs.field(factory=dict)
    xy_3: dict[str, deque] = attrs.field(factory=dict)
    xx_3: dict[str, deque] = attrs.field(factory=dict)
    yy_3: dict[str, deque] = attrs.field(factory=dict)
    x_3: dict[str, deque] = attrs.field(factorjy=dict)
    y_3: dict[str, deque] = attrs.field(factory=dict)

    # Amihud liquidity measure
    amihud: dict[str, deque] = attrs.field(factory=dict)
    amihud_avg: dict[str, float] = attrs.field(factory=dict)

    # Sum variables for efficient rolling calculations
    xy_1_sum: dict[str, float] = attrs.field(factory=dict)
    xx_1_sum: dict[str, float] = attrs.field(factory=dict)
    yy_1_sum: dict[str, float] = attrs.field(factory=dict)
    x_1_sum: dict[str, float] = attrs.field(factory=dict)
    y_1_sum: dict[str, float] = attrs.field(factory=dict)
    xy_2_sum: dict[str, float] = attrs.field(factory=dict)
    xx_2_sum: dict[str, float] = attrs.field(factory=dict)
    yy_2_sum: dict[str, float] = attrs.field(factory=dict)
    x_2_sum: dict[str, float] = attrs.field(factory=dict)
    y_2_sum: dict[str, float] = attrs.field(factory=dict)
    xy_3_sum: dict[str, float] = attrs.field(factory=dict)
    xx_3_sum: dict[str, float] = attrs.field(factory=dict)
    yy_3_sum: dict[str, float] = attrs.field(factory=dict)
    x_3_sum: dict[str, float] = attrs.field(factory=dict)
    y_3_sum: dict[str, float] = attrs.field(factory=dict)

    # Statistics tracking
    revenues: list[float] = attrs.field(factory=list)
    coin_revenues: dict[str, list[float]] = attrs.field(factory=dict)
    stats_1: list[float] = attrs.field(factory=list)  # objective_value
    stats_2: list[float] = attrs.field(factory=list)  # coef_1_mean
    stats_3: list[float] = attrs.field(factory=list)  # coef_2_mean
    stats_4: list[float] = attrs.field(factory=list)  # coef_3_mean
    stats_5: list[float] = attrs.field(factory=list)  # corr_1_mean
    stats_6: list[float] = attrs.field(factory=list)  # corr_2_mean
    stats_7: list[float] = attrs.field(factory=list)  # corr_3_mean
    stats_8: list[float] = attrs.field(factory=list)  # cmi_mean
    stats_9: list[float] = attrs.field(factory=list)  # cmi_std
    stats_10: list[float] = attrs.field(factory=list)  # amihud_mean

    # Other tracking variables
    num_trades: int = 0
    sum_of_theta_price: float = 0.0
    print_df: pd.DataFrame = attrs.field(factory=lambda: pd.DataFrame())

    # Time series storage
    time_series: TimeSeriesData = attrs.field(factory=TimeSeriesData)
    last_logged_date: str = attrs.field(default="")

    def __attrs_post_init__(self):
        """Initialize symbol-based dictionaries after object creation"""
        # Initialize all symbol-based dictionaries
        for symbol in self.symbols:
            self.coin_closes[symbol] = deque(maxlen=self.back_window_size)
            self.coin_volumes[symbol] = deque(maxlen=self.back_window_size)

            # Statistical calculation deques
            self.xy_1[symbol] = deque(maxlen=self.back_window_size)
            self.xx_1[symbol] = deque(maxlen=self.back_window_size)
            self.yy_1[symbol] = deque(maxlen=self.back_window_size)
            self.x_1[symbol] = deque(maxlen=self.back_window_size)
            self.y_1[symbol] = deque(maxlen=self.back_window_size)
            self.xy_2[symbol] = deque(maxlen=self.back_window_size)
            self.xx_2[symbol] = deque(maxlen=self.back_window_size)
            self.yy_2[symbol] = deque(maxlen=self.back_window_size)
            self.x_2[symbol] = deque(maxlen=self.back_window_size)
            self.y_2[symbol] = deque(maxlen=self.back_window_size)
            self.xy_3[symbol] = deque(maxlen=self.back_window_size)
            self.xx_3[symbol] = deque(maxlen=self.back_window_size)
            self.yy_3[symbol] = deque(maxlen=self.back_window_size)
            self.x_3[symbol] = deque(maxlen=self.back_window_size)
            self.y_3[symbol] = deque(maxlen=self.back_window_size)

            # Amihud liquidity measure
            self.amihud[symbol] = deque(maxlen=self.back_window_size)
            self.amihud_avg[symbol] = 0.0

            # Sum variables
            self.xy_1_sum[symbol] = 0.0
            self.xx_1_sum[symbol] = 0.0
            self.yy_1_sum[symbol] = 0.0
            self.x_1_sum[symbol] = 0.0
            self.y_1_sum[symbol] = 0.0
            self.xy_2_sum[symbol] = 0.0
            self.xx_2_sum[symbol] = 0.0
            self.yy_2_sum[symbol] = 0.0
            self.x_2_sum[symbol] = 0.0
            self.y_2_sum[symbol] = 0.0
            self.xy_3_sum[symbol] = 0.0
            self.xx_3_sum[symbol] = 0.0
            self.yy_3_sum[symbol] = 0.0
            self.x_3_sum[symbol] = 0.0
            self.y_3_sum[symbol] = 0.0

            # Revenue tracking
            self.coin_revenues[symbol] = []

    def get_time_series(self) -> TimeSeriesData:
        """Return the complete time series data"""
        return self.time_series

    def step(self, api: qoc.Api) -> None:
        now: DateTime = qoc.now()

        # Check if we've entered a new day and log progress
        current_date = now.format("YYYY-MM-DD")
        if current_date != self.last_logged_date:
            logger.info(f"Trading Date: {current_date}")
            self.last_logged_date = current_date

        # Get current price data for all symbols
        symbol_data = {}
        for symbol in self.symbols:
            klines: pl.DataFrame = api.klines(symbol, "1m", limit=1)
            price = float(klines["close"].last())
            volume = float(klines["volume"].last())
            symbol_data[symbol] = {"price": price, "volume": volume}

        # Get BTC data for market index calculation
        btc_klines: pl.DataFrame = api.klines("BTCUSDT", "1m", limit=1)
        btc_price = float(btc_klines["close"].last())
        btc_vol = float(btc_klines["volume"].last())

        # Update historical data
        self.btc_close.append(btc_price)
        self.btc_volume.append(btc_vol)

        for symbol in self.symbols:
            self.coin_closes[symbol].append(symbol_data[symbol]["price"])
            self.coin_volumes[symbol].append(symbol_data[symbol]["volume"])

        # Statistical arbitrage logic
        if self.current_step >= self.back_window_size:
            # Calculate market indices
            if len(self.btc_close) >= 2:
                self.mkt_idx_1.append(self.btc_close[-1])
                self.mkt_idx_2.append(
                    self.btc_close[-2]
                    if len(self.btc_close) >= 2
                    else self.btc_close[-1]
                )

                # Calculate market index 3 (volume-weighted growth)
                fz = 0
                fm = 0
                for symbol in self.symbols:
                    if len(self.coin_closes[symbol]) >= 2:
                        growth = (
                            self.coin_closes[symbol][-1] - self.coin_closes[symbol][-2]
                        )
                        fz += growth * symbol_data[symbol]["volume"]
                        fm += (
                            symbol_data[symbol]["volume"] * symbol_data[symbol]["price"]
                        )

                if fm != 0:
                    self.mkt_idx_3.append(fz / fm)
                else:
                    self.mkt_idx_3.append(0)

                # Calculate market index differences
                mkt_idx_1_diff = (
                    self.mkt_idx_1[-1] - self.mkt_idx_1[-2]
                    if len(self.mkt_idx_1) >= 2
                    else 0
                )
                mkt_idx_2_diff = (
                    self.mkt_idx_2[-1] - self.mkt_idx_2[-2]
                    if len(self.mkt_idx_2) >= 2
                    else 0
                )
                mkt_idx_3_diff = self.mkt_idx_3[-1] if len(self.mkt_idx_3) >= 1 else 0

                # Update statistical measures for each symbol
                coef_records = {}
                for symbol in self.symbols:
                    if len(self.coin_closes[symbol]) >= 2:
                        growth = (
                            self.coin_closes[symbol][-1] - self.coin_closes[symbol][-2]
                        )

                        # Calculate CMI (Cumulative Market Index)
                        closes_array = np.array(list(self.coin_closes[symbol]))
                        if len(closes_array) > 1:
                            diff = np.diff(closes_array)
                            cmi = (
                                (closes_array[-1] - closes_array[0])
                                / np.sum(np.abs(diff))
                                * 100
                                if np.sum(np.abs(diff)) != 0
                                else 0
                            )
                        else:
                            cmi = 0

                        # Update rolling sums (remove oldest values if at capacity)
                        if len(self.xy_1[symbol]) >= self.back_window_size:
                            self.xy_1_sum[symbol] -= self.xy_1[symbol][0]
                            self.xx_1_sum[symbol] -= self.xx_1[symbol][0]
                            self.yy_1_sum[symbol] -= self.yy_1[symbol][0]
                            self.x_1_sum[symbol] -= self.x_1[symbol][0]
                            self.y_1_sum[symbol] -= self.y_1[symbol][0]

                            self.xy_2_sum[symbol] -= self.xy_2[symbol][0]
                            self.xx_2_sum[symbol] -= self.xx_2[symbol][0]
                            self.yy_2_sum[symbol] -= self.yy_2[symbol][0]
                            self.x_2_sum[symbol] -= self.x_2[symbol][0]
                            self.y_2_sum[symbol] -= self.y_2[symbol][0]

                            self.xy_3_sum[symbol] -= self.xy_3[symbol][0]
                            self.xx_3_sum[symbol] -= self.xx_3[symbol][0]
                            self.yy_3_sum[symbol] -= self.yy_3[symbol][0]
                            self.x_3_sum[symbol] -= self.x_3[symbol][0]
                            self.y_3_sum[symbol] -= self.y_3[symbol][0]

                            self.amihud_avg[symbol] = (
                                self.amihud_avg[symbol] * self.back_window_size
                                - self.amihud[symbol][0]
                            ) / self.back_window_size

                        # Add new values
                        xy_1_val = growth * mkt_idx_1_diff
                        xx_1_val = mkt_idx_1_diff**2
                        yy_1_val = growth**2
                        x_1_val = mkt_idx_1_diff
                        y_1_val = growth

                        xy_2_val = growth * mkt_idx_2_diff
                        xx_2_val = mkt_idx_2_diff**2
                        yy_2_val = growth**2
                        x_2_val = mkt_idx_2_diff
                        y_2_val = growth

                        xy_3_val = growth * mkt_idx_3_diff
                        xx_3_val = mkt_idx_3_diff**2
                        yy_3_val = growth**2
                        x_3_val = mkt_idx_3_diff
                        y_3_val = growth

                        self.xy_1[symbol].append(xy_1_val)
                        self.xx_1[symbol].append(xx_1_val)
                        self.yy_1[symbol].append(yy_1_val)
                        self.x_1[symbol].append(x_1_val)
                        self.y_1[symbol].append(y_1_val)
                        self.xy_2[symbol].append(xy_2_val)
                        self.xx_2[symbol].append(xx_2_val)
                        self.yy_2[symbol].append(yy_2_val)
                        self.x_2[symbol].append(x_2_val)
                        self.y_2[symbol].append(y_2_val)
                        self.xy_3[symbol].append(xy_3_val)
                        self.xx_3[symbol].append(xx_3_val)
                        self.yy_3[symbol].append(yy_3_val)
                        self.x_3[symbol].append(x_3_val)
                        self.y_3[symbol].append(y_3_val)

                        # Calculate and append Amihud illiquidity measure
                        amihud_val = (
                            abs(growth)
                            / (symbol_data[symbol]["volume"] + 1e-9)
                            / symbol_data[symbol]["price"]
                            / symbol_data[symbol]["price"]
                        )
                        self.amihud[symbol].append(amihud_val)

                        # Update sums
                        self.xy_1_sum[symbol] += xy_1_val
                        self.xx_1_sum[symbol] += xx_1_val
                        self.yy_1_sum[symbol] += yy_1_val
                        self.x_1_sum[symbol] += x_1_val
                        self.y_1_sum[symbol] += y_1_val
                        self.xy_2_sum[symbol] += xy_2_val
                        self.xx_2_sum[symbol] += xx_2_val
                        self.yy_2_sum[symbol] += yy_2_val
                        self.x_2_sum[symbol] += x_2_val
                        self.y_2_sum[symbol] += y_2_val
                        self.xy_3_sum[symbol] += xy_3_val
                        self.xx_3_sum[symbol] += xx_3_val
                        self.yy_3_sum[symbol] += yy_3_val
                        self.x_3_sum[symbol] += x_3_val
                        self.y_3_sum[symbol] += y_3_val

                        self.amihud_avg[symbol] = (
                            self.amihud_avg[symbol] * self.back_window_size + amihud_val
                        ) / self.back_window_size

                        # Calculate coefficients and correlations if we have enough data
                        if self.current_step >= 2 * self.back_window_size:
                            coef_1 = (
                                self.xy_1_sum[symbol] / self.xx_1_sum[symbol] * 1000
                                if self.xx_1_sum[symbol] != 0
                                else 0
                            )
                            coef_2 = (
                                self.xy_2_sum[symbol] / self.xx_2_sum[symbol] * 1000
                                if self.xx_2_sum[symbol] != 0
                                else 0
                            )
                            coef_3 = (
                                self.xy_3_sum[symbol] / self.xx_3_sum[symbol] * 1000
                                if self.xx_3_sum[symbol] != 0
                                else 0
                            )

                            # Calculate correlations
                            denom_1 = (
                                self.back_window_size * self.xx_1_sum[symbol]
                                - self.x_1_sum[symbol] ** 2
                            ) * (
                                self.back_window_size * self.yy_1_sum[symbol]
                                - self.y_1_sum[symbol] ** 2
                            )
                            corr_1 = (
                                (
                                    self.back_window_size * self.xy_1_sum[symbol]
                                    - self.x_1_sum[symbol] * self.y_1_sum[symbol]
                                )
                                / np.sqrt(denom_1)
                                if denom_1 > 0
                                else 0
                            )

                            denom_2 = (
                                self.back_window_size * self.xx_2_sum[symbol]
                                - self.x_2_sum[symbol] ** 2
                            ) * (
                                self.back_window_size * self.yy_2_sum[symbol]
                                - self.y_2_sum[symbol] ** 2
                            )
                            corr_2 = (
                                (
                                    self.back_window_size * self.xy_2_sum[symbol]
                                    - self.x_2_sum[symbol] * self.y_2_sum[symbol]
                                )
                                / np.sqrt(denom_2)
                                if denom_2 > 0
                                else 0
                            )

                            denom_3 = (
                                self.back_window_size * self.xx_3_sum[symbol]
                                - self.x_3_sum[symbol] ** 2
                            ) * (
                                self.back_window_size * self.yy_3_sum[symbol]
                                - self.y_3_sum[symbol] ** 2
                            )
                            corr_3 = (
                                (
                                    self.back_window_size * self.xy_3_sum[symbol]
                                    - self.x_3_sum[symbol] * self.y_3_sum[symbol]
                                )
                                / np.sqrt(denom_3)
                                if denom_3 > 0
                                else 0
                            )

                            coef_records[symbol] = {
                                "coef_1": coef_1,
                                "coef_2": coef_2,
                                "coef_3": coef_3,
                                "corr_1": corr_1,
                                "corr_2": corr_2,
                                "corr_3": corr_3,
                                "cmi": cmi,
                                "amihud": self.amihud_avg[symbol],
                            }

                # Portfolio optimization and trading logic
                if self.current_step >= 2 * self.back_window_size and coef_records:
                    try:
                        # Convert to DataFrame
                        coef_df = pd.DataFrame.from_dict(coef_records, orient="index")

                        # Calculate means
                        coef_1_mean = coef_df["coef_1"].mean()
                        coef_2_mean = coef_df["coef_2"].mean()
                        coef_3_mean = coef_df["coef_3"].mean()
                        corr_1_mean = coef_df["corr_1"].mean()
                        corr_2_mean = coef_df["corr_2"].mean()
                        corr_3_mean = coef_df["corr_3"].mean()
                        cmi_mean = coef_df["cmi"].mean()
                        cmi_std = coef_df["cmi"].std()
                        amihud_mean = coef_df["amihud"].mean()

                        names = coef_df.index.tolist()
                        coef1 = coef_df["coef_1"].values.astype(float)
                        prices = np.array(
                            [symbol_data[symbol]["price"] for symbol in names]
                        )
                        cmis = coef_df["cmi"].values.astype(float)

                        k = len(prices)

                        # Linear programming optimization
                        c_x = -cmis * prices
                        c = np.concatenate([c_x, np.zeros(k)])

                        # Constraints
                        A_eq = np.hstack([np.vstack([coef1]), np.zeros((1, k))])
                        b_eq = np.zeros(1)

                        A_ub = []
                        b_ub = []

                        for i in range(k):
                            row_pos = np.zeros(2 * k)
                            row_pos[i] = prices[i]
                            row_pos[k + i] = -1
                            A_ub.append(row_pos)
                            b_ub.append(0.0)

                            row_neg = np.zeros(2 * k)
                            row_neg[i] = -prices[i]
                            row_neg[k + i] = -1
                            A_ub.append(row_neg)
                            b_ub.append(0.0)

                        row_sum = np.zeros(2 * k)
                        row_sum[k:] = 1.0
                        A_ub.append(row_sum)
                        b_ub.append(1000.0)

                        A_ub = np.array(A_ub)
                        b_ub = np.array(b_ub)

                        bounds = [(-np.inf, np.inf)] * k + [(0, np.inf)] * k

                        # Solve optimization
                        res = linprog(
                            c=c,
                            A_eq=A_eq,
                            b_eq=b_eq,
                            A_ub=A_ub,
                            b_ub=b_ub,
                            bounds=bounds,
                            method="highs",
                        )

                        objective_value = res.fun

                        # Check trading conditions
                        if (
                            res.success
                            and corr_2_mean > 0.1
                            and corr_1_mean > 0.6
                            and cmi_std < 6
                            and objective_value < -7500
                            and abs(cmi_mean) < 10
                        ):
                            theta = res.x[:k]

                            # Execute trades based on theta values
                            for i, symbol in enumerate(names):
                                if (
                                    abs(theta[i]) > 0.001
                                ):  # Only trade if position is significant
                                    quantity = abs(theta[i])
                                    side = (
                                        OrderSide.BUY
                                        if theta[i] > 0
                                        else OrderSide.SELL
                                    )

                                    try:
                                        response = api.order_market(
                                            symbol, side, quantity=quantity
                                        )
                                        self.orders[symbol].append(
                                            Order(
                                                quantity=response.executed_qty,
                                                symbol=symbol,
                                                time=response.transact_time,
                                            )
                                        )
                                        logger.info(
                                            f"Executed {side} order for {symbol}: {quantity}"
                                        )
                                    except Exception as e:
                                        logger.error(
                                            f"Failed to execute order for {symbol}: {e}"
                                        )

                            # Update statistics
                            self.num_trades += 1
                            self.stats_1.append(objective_value)
                            self.stats_2.append(coef_1_mean)
                            self.stats_3.append(coef_2_mean)
                            self.stats_4.append(coef_3_mean)
                            self.stats_5.append(corr_1_mean)
                            self.stats_6.append(corr_2_mean)
                            self.stats_7.append(corr_3_mean)
                            self.stats_8.append(cmi_mean)
                            self.stats_9.append(cmi_std)
                            self.stats_10.append(amihud_mean)

                            self.sum_of_theta_price += sum(abs(theta * prices))

                            # Store metrics in time series
                            self.time_series.add_metric(
                                "objective_value", now, objective_value
                            )
                            self.time_series.add_metric("corr_1_mean", now, corr_1_mean)
                            self.time_series.add_metric("corr_2_mean", now, corr_2_mean)
                            self.time_series.add_metric("cmi_std", now, cmi_std)

                    except Exception as e:
                        logger.error(f"Error in portfolio optimization: {e}")

        # Clean up expired orders
        for symbol in self.symbols:
            orders = self.orders[symbol]
            while orders and orders[0].time + self.forward_window < now:
                order = orders.popleft()
                try:
                    # Close position
                    opposite_side = (
                        OrderSide.SELL if order.quantity > 0 else OrderSide.BUY
                    )
                    api.order_market(
                        order.symbol, opposite_side, quantity=abs(order.quantity)
                    )
                    logger.info(f"Closed position for {order.symbol}")
                except Exception as e:
                    logger.error(f"Failed to close position for {order.symbol}: {e}")

        self.current_step += 1

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
                try:
                    price: float = api.price(symbol, interval="1m")
                    total_value += (balance.free + balance.locked) * price
                except:
                    pass

        # Store total value with datetime index
        self.time_series.add_metric("total_value", clock.now, total_value)
        cherries.log_metric("Total Value (USDT)", total_value, step=clock.step)
        cherries.log_metric("Number of Trades", self.num_trades, step=clock.step)


def plot_time_series(tss: TimeSeriesData, output_dir: str = "./plots") -> None:
    """Plot time series data and save to specified directory

    Args:
        tss: TimeSeriesData object containing the time series data
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all available metrics
    metrics = tss.get_all_metrics()

    if not metrics:
        logger.warning("No metrics found in time series data")
        return

    logger.info(f"Plotting {len(metrics)} metrics: {metrics}")

    # Plot each metric separately
    for metric in metrics:
        data = tss.get_metric(metric)

        if not data:
            logger.warning(f"No data found for metric: {metric}")
            continue

        # Extract timestamps and values
        timestamps, values = zip(*data, strict=False)

        # Convert pendulum DateTime to python datetime for matplotlib
        datetime_list = []
        for ts in timestamps:
            if hasattr(ts, "to_datetime"):
                # For pendulum DateTime objects
                datetime_list.append(ts.to_datetime())
            elif hasattr(ts, "in_timezone"):
                # Alternative method for pendulum DateTime
                datetime_list.append(ts.in_timezone("UTC").replace(tzinfo=None))
            else:
                # For regular datetime objects
                datetime_list.append(ts)

        # Create the plot
        plt.figure(figsize=(14, 8))
        plt.plot(datetime_list, values, linewidth=2, marker="o", markersize=2)

        # Format the plot
        plt.title(f"Time Series: {metric}", fontsize=16, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Calculate time span to set appropriate tick interval
        time_span = datetime_list[-1] - datetime_list[0]
        days = time_span.days

        # Set appropriate time locator based on data span
        if days <= 1:
            # For data spanning 1 day or less, show every 2 hours
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        elif days <= 7:
            # For data spanning up to a week, show every 12 hours
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        elif days <= 30:
            # For data spanning up to a month, show daily
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        else:
            # For longer periods, show weekly
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        # Limit maximum number of ticks to prevent the warning
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

        plt.xticks(rotation=45)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        safe_filename = metric.replace("/", "_").replace(" ", "_")
        output_path = os.path.join(output_dir, f"{safe_filename}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot for {metric} to: {output_path}")


def main(cfg: Config) -> None:
    if cfg.online:
        qoc.set_clock(qoc.ClockOnline(interval="5m"))
    else:
        qoc.set_clock(
            qoc.ClockOffline(interval="5m", start="2025-10-01", end="2025-10-15")
        )
    api: qoc.Api = qoc.ApiBinanceSpot() if cfg.online else qoc.ApiOfflineSpot()
    strategy = StrategyRev()

    logger.info("Starting strategy execution...")

    for _ in qoc.loop():
        try:
            strategy.step(api=api)
            strategy.dump(api=api)
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logger.exception("error")

    logger.info("Strategy execution completed, generating plots...")
    tss = strategy.get_time_series()
    plot_time_series(tss, output_dir="deploy/rev/plots")


if __name__ == "__main__":
    main(cfg=Config())


# BINANCE_BASE_URL='https://api.binance.us' python /Users/lizeyu/Desktop/Quant-on-Crypto/deploy/rev/main.py
