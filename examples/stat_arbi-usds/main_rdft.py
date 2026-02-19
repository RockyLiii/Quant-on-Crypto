from __future__ import annotations

import collections
import logging
import pickle
from collections import deque
from decimal import Decimal
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import pandas as pd
import pendulum
import polars as pl
from environs import env
from liblaf import cherries
from pendulum import DateTime, Duration

import qoc
import qoc.logging
from qoc.api.usds import ApiUsds, ApiUsdsOffline, ApiUsdsOnline
from qoc.api.usds.models import Account, MarginType, OrderResponse, OrderSide
from qoc.typing import SymbolName

logger = logging.getLogger(__name__)


class DataLogger(collections.UserDict[str, list[Any]]):
    def append(self, key: str, value: Any) -> None:
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def dump(self, filename: str | Path | None = None) -> None:
        df: pl.DataFrame = self.to_polars()
        if filename is None:
            filename = qoc.entrypoint().parent / "data_log.csv"
        df.write_csv(filename)

    def step(self, time: DateTime) -> None:
        self.append("time", time)

    def to_polars(self) -> pl.DataFrame:
        return pl.from_dict(self.data)


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


class RollingMax:
    def __init__(self, window: int):
        self.window = window
        self.deque = deque()  # (value, index)
        self.index = -1

    def update(self, value: float) -> float:
        self.index += 1

        # 删除比当前小的
        while self.deque and self.deque[-1][0] <= value:
            self.deque.pop()

        self.deque.append((value, self.index))

        # 删除窗口外
        while self.deque and self.deque[0][1] <= self.index - self.window:
            self.deque.popleft()

        return self.deque[0][0]


class RollingMin:
    def __init__(self, window: int):
        self.window = window
        self.deque = deque()
        self.index = -1

    def update(self, value: float) -> float:
        self.index += 1

        # 删除比当前大的
        while self.deque and self.deque[-1][0] >= value:
            self.deque.pop()

        self.deque.append((value, self.index))

        # 删除窗口外
        while self.deque and self.deque[0][1] <= self.index - self.window:
            self.deque.popleft()

        return self.deque[0][0]


@attrs.define
class Strategy(qoc.PersistableMixin):
    api: ApiUsds

    # -------------------------------- Config -------------------------------- #
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
            "NEARUSDT",
            "HBARUSDT",
            "INJUSDT",
            "OPUSDT",
            "ARBUSDT",
            "SUIUSDT",
            "SEIUSDT",
            "RENDERUSDT",
            "TIAUSDT",
            # "FTMUSDT",
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

    stats_list: list[str] = attrs.field(
        factory=lambda: [
            "close_ret",
            "amihud",
            "cmi",
            "coin",
            "atr",
            "bb",
            "std",
        ]
    )

    windows_ratio = [0.25, 0.5, 1, 4, 16, 64, 256]
    preload_window: int = 30720

    forward_window: Duration = attrs.field(
        factory=lambda: pendulum.duration(minutes=30)
    )

    back_window_in_mins: int = 120

    back_window: Duration = attrs.field(factory=lambda: pendulum.duration(minutes=120))

    bullet_size: float = 100
    max_concurrent_orders: int = 2

    stop_loss: float = 0.04
    take_profit: float = 0.04

    freeze_window: int = 60  # in minutes

    # --------------------------------- State -------------------------------- #
    orders: deque[Order] = attrs.field(factory=deque)

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

    # Price and volume history
    coin_closes: dict[str, deque] = attrs.field(factory=dict)
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

    windows: list[float] = attrs.field(factory=list)

    model_trained: any = None  # Placeholder for potential predictive model
    feature_order: list[str] = attrs.field(factory=list)

    model_input_dict: dict[str, float] = attrs.field(factory=dict)

    # # features
    # close_ret
    log_close: dict[str, dict] = attrs.field(factory=dict)

    log_close_tail_minus_head: dict[str, dict] = attrs.field(factory=dict)

    close_ret: dict[str, dict] = attrs.field(factory=dict)

    # amihud
    close_x_vol: dict[str, dict] = attrs.field(factory=dict)

    # amihud: dict[str, dict] = attrs.field(factory=dict)

    # cmi
    tr: dict[str, dict] = attrs.field(factory=dict)
    highs: dict[str, dict] = attrs.field(factory=dict)
    lows: dict[str, dict] = attrs.field(factory=dict)

    tr_sum: dict[str, dict] = attrs.field(factory=dict)
    highs_max: dict[str, dict] = attrs.field(factory=dict)
    lows_min: dict[str, dict] = attrs.field(factory=dict)

    cmi: dict[str, dict] = attrs.field(factory=dict)

    # coin
    # atr
    atr: dict[str, dict] = attrs.field(factory=dict)
    # bb
    closes: dict[str, dict] = attrs.field(factory=dict)
    close_sqr: dict[str, dict] = attrs.field(factory=dict)
    
    closes_sum: dict[str, dict] = attrs.field(factory=dict)
    closes_sqr_sum: dict[str, dict] = attrs.field(factory=dict)

    bb: dict[str, dict] = attrs.field(factory=dict)

    # std
    log_close_diff: dict[str, dict] = attrs.field(factory=dict)
    log_close_diff_sqr: dict[str, dict] = attrs.field(factory=dict)

    log_close_diff_sum: dict[str, dict] = attrs.field(factory=dict)
    log_close_diff_sqr_sum: dict[str, dict] = attrs.field(factory=dict)

    std: dict[str, dict] = attrs.field(factory=dict)

    # temp:
    y_pred: float = 0.0
    count: int = 0

    freeze: int = 0

    data_logger: DataLogger = attrs.field(factory=DataLogger)

    def update_indicators(
        self,
        symbol: str,
        price: float,
        volume: float,
        btc_price: float,
        btc_volume: float,
        high: float,
        low: float,
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
        self.coin_closes[symbol].append(price)

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

        # Update states for features
        for w in self.windows:
            # state arrays
            last_log_close = (
                self.log_close[symbol][w][-1]
                if len(self.log_close[symbol][w]) > 0
                else np.log(price)
            )

            self.log_close[symbol][w].append(np.log(price))

            log_price_diff = np.log(price) - last_log_close
            self.log_close_diff[symbol][w].append(log_price_diff)
            self.log_close_diff_sqr[symbol][w].append(log_price_diff**2)

            last_close = (
                self.closes[symbol][w][-1] if len(self.closes[symbol][w]) > 0 else price
            )
            tr = max(high - low, abs(high - last_close), abs(low - last_close))
            self.tr[symbol][w].append(tr)
            self.highs[symbol][w].append(high)
            self.lows[symbol][w].append(low)

            self.closes[symbol][w].append(self.coin_closes[symbol][-1])
            self.close_sqr[symbol][w].append(self.coin_closes[symbol][-1] ** 2)

            # states
            self.log_close_tail_minus_head[symbol][w] = (
                self.log_close[symbol][w][-1] - self.log_close[symbol][w][0]
            )
            # self.tr_sum[symbol][w] = sum(self.tr[symbol][w])
            self.tr_sum[symbol][w] += self.tr[symbol][w][-1] - (
                self.tr[symbol][w][0] if len(self.tr[symbol][w]) >= w else 0
            )

            # [optmz]
            # self.highs_max[symbol][w] = max(self.highs[symbol][w])
            # self.lows_min[symbol][w] = min(self.lows[symbol][w])

            self.highs_max[symbol][w] = self.rolling_max[symbol][w].update(high)

            self.lows_min[symbol][w] = self.rolling_min[symbol][w].update(low)

            self.closes_sum[symbol][w] += self.closes[symbol][w][-1] - (self.closes[symbol][w][0] if len(self.closes[symbol][w]) >= w else 0)
            self.closes_sqr_sum[symbol][w] += self.close_sqr[symbol][w][-1] - (self.close_sqr[symbol][w][0] if len(self.close_sqr[symbol][w]) >= w else 0)          

            self.log_close_diff_sum[symbol][w] += self.log_close_diff[symbol][w][-1] - (
                self.log_close_diff[symbol][w][0]
                if len(self.log_close_diff[symbol][w]) >= w
                else 0
            )
            self.log_close_diff_sqr_sum[symbol][w] += self.log_close_diff_sqr[symbol][
                w
            ][-1] - (
                self.log_close_diff_sqr[symbol][w][0]
                if len(self.log_close_diff_sqr[symbol][w]) >= w
                else 0
            )

            # features
            self.close_ret[symbol][w] = (
                self.log_close_tail_minus_head[symbol][w]
                if self.log_close_tail_minus_head[symbol][w] != 0
                else 0
            )

            self.cmi[symbol][w] = (
                100
                * np.log(
                    self.tr_sum[symbol][w]
                    / (self.highs_max[symbol][w] - self.lows_min[symbol][w])
                )
                / np.log(w)
                if self.tr_sum[symbol][w] != 0
                and self.highs_max[symbol][w] != self.lows_min[symbol][w]
                else 0
            )

            self.atr[symbol][w] = (
                self.tr_sum[symbol][w] / w if self.tr_sum[symbol][w] != 0 else 0
            )

            self.bb[symbol][w] = (price - self.closes_sum[symbol][w] / w) / (np.sqrt(self.closes_sqr_sum[symbol][w] / w - (self.closes_sum[symbol][w] / w)**2)) if self.closes_sqr_sum[symbol][w] != 0 else 0


            self.std[symbol][w] = (
                np.sqrt(
                    self.log_close_diff_sqr_sum[symbol][w] / w
                    - (self.log_close_diff_sum[symbol][w] / w) ** 2
                )
                if w > 1
                else 0
            )

    def __attrs_post_init__(self) -> None:
        """Initialize symbol-based dictionaries after object creation"""

        # window list
        for ratio in self.windows_ratio:
            self.windows.append(int(self.back_window_in_mins * ratio))
        # states for features
        for state in [
            self.log_close,
            self.close_x_vol,
            self.tr,
            self.highs,
            self.lows,
            self.closes,
            self.close_sqr,
            self.log_close_diff,
            self.log_close_diff_sqr,
        ]:
            for symbol in self.symbols:
                state[symbol] = {}
                for w in self.windows:
                    state[symbol][w] = deque(maxlen=w)
        for state_result in [
            self.log_close_tail_minus_head,
            self.tr_sum,
            self.highs_max,
            self.lows_min,
            self.closes_sum, 
            self.closes_sqr_sum,
            self.log_close_diff_sum,
            self.log_close_diff_sqr_sum,
        ]:
            for symbol in self.symbols:
                state_result[symbol] = {}
                for w in self.windows:
                    state_result[symbol][w] = 0.0
        for feature in [self.close_ret, self.cmi, self.atr, self.bb, self.std]:
            for symbol in self.symbols:
                feature[symbol] = {}
                for w in self.windows:
                    feature[symbol][w] = 0.0

        # 每个 symbol 单独一套
        self.rolling_max = {}
        self.rolling_min = {}

        self.highs_max = {}
        self.lows_min = {}

        for symbol in self.symbols:
            self.rolling_max[symbol] = {}
            self.rolling_min[symbol] = {}

            self.highs_max[symbol] = {}
            self.lows_min[symbol] = {}

            for w in self.windows:
                self.rolling_max[symbol][w] = RollingMax(w)
                self.rolling_min[symbol][w] = RollingMin(w)

                self.highs_max[symbol][w] = 0.0
                self.lows_min[symbol][w] = 0.0

        # import model
        self.model_trained = pickle.load(
            open("examples/stat_arbi-usds/random_forest_model_rct.pkl", "rb")
        )
        self.feature_order = self.model_trained.feature_names_in_.tolist()

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
            "BTCUSDT", "1m", end_time=now, limit=self.preload_window
        )

        btc_prices = btc_klines["close"].to_list()
        btc_volumes = btc_klines["volume"].to_list()

        # Keep track of symbols that fail to load data
        symbols_to_remove = []

        for symbol in self.symbols:
            print("Loading historical data for %s...", symbol)
            self.coin_closes[symbol] = deque(maxlen=self.back_window_in_mins)
            try:
                klines: pl.DataFrame = self.api.klines(
                    symbol, "1m", end_time=now, limit=self.preload_window
                )

                prices = klines["close"].to_list()
                volumes = klines["volume"].to_list()
                highs = klines["high"].to_list()
                lows = klines["low"].to_list()

                for i in range(len(prices)):
                    self.update_indicators(
                        symbol,
                        prices[i],
                        volumes[i],
                        btc_prices[i],
                        btc_volumes[i],
                        highs[i],
                        lows[i],
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
        self.data_logger.step(now)
        self.data_logger.append("t", self.t)

        klines: pl.DataFrame = self.api.klines("BTCUSDT", "1m", end_time=now, limit=2)

        btc_price: float = klines["close"].last()  # pyright: ignore[reportAssignmentType]
        self.data_logger.append("btc_price", btc_price)
        btc_volume: float = klines["volume"].last()  # pyright: ignore[reportAssignmentType]
        self.data_logger.append("btc_volume", btc_volume)

        coef_records = {}

        for symbol in self.symbols:
            orders = self.orders
            klines: pl.DataFrame = self.api.klines(symbol, "1m", end_time=now, limit=2)
            price: float = klines["close"].last()  # pyright: ignore[reportAssignmentType]
            # self.data_logger.append(f"{symbol}_price", price)
            volume: float = klines["volume"].last()  # pyright: ignore[reportAssignmentType]
            # self.data_logger.append(f"{symbol}_volume", volume)
            high: float = klines["high"].last()  # pyright: ignore[reportAssignmentType]
            # self.data_logger.append(f"{symbol}_high", high)
            low: float = klines["low"].last()  # pyright: ignore[reportAssignmentType]
            # self.data_logger.append(f"{symbol}_low", low)
            self.update_indicators(
                symbol, price, volume, btc_price, btc_volume, high, low
            )

            beta = self.xy_1_sum[symbol] / self.xx_1_sum[symbol]
            # self.data_logger.append(f"{symbol}_beta", beta)

            residual = price - (beta * btc_price)
            # self.data_logger.append(f"{symbol}_residual", residual)

            corr = (
                (
                    self.back_window_in_mins * self.xy_1_sum[symbol]
                    - self.x_1_sum[symbol] * self.y_1_sum[symbol]
                )
                / np.sqrt(
                    (
                        self.back_window_in_mins * self.xx_1_sum[symbol]
                        - self.x_1_sum[symbol] ** 2
                    )
                    * (
                        self.back_window_in_mins * self.yy_1_sum[symbol]
                        - self.y_1_sum[symbol] ** 2
                    )
                )
                if (
                    self.back_window_in_mins * self.xx_1_sum[symbol]
                    - self.x_1_sum[symbol] ** 2
                )
                * (
                    self.back_window_in_mins * self.yy_1_sum[symbol]
                    - self.y_1_sum[symbol] ** 2
                )
                != 0
                else 0
            )
            # self.data_logger.append(f"{symbol}_corr", corr)
            # debug
            if (
                self.back_window_in_mins * self.xx_1_sum[symbol]
                - self.x_1_sum[symbol] ** 2
            ) * (
                self.back_window_in_mins * self.yy_1_sum[symbol]
                - self.y_1_sum[symbol] ** 2
            ) <= 0:
                logger.warning(
                    "Zero division in corr calculation for %s",
                    symbol,
                    self.back_window_in_mins * self.xx_1_sum[symbol],
                    self.x_1_sum[symbol],
                    self.back_window_in_mins * self.yy_1_sum[symbol],
                    self.y_1_sum[symbol],
                )

            ys = np.asarray(self.coin_closes[symbol], dtype=float)
            xs = np.asarray(self.coin_closes["BTCUSDT"], dtype=float)
            residuals = ys - beta * xs
            residual_z = np.sum(residuals < residuals[-1]) + 1

            coef_records[symbol] = {
                "Close": price,
                "beta": beta,
                "beta_adj": (
                    beta
                    * np.array(self.coin_closes["BTCUSDT"])[-1]
                    / np.array(self.coin_closes[symbol])[-1]
                    if np.array(self.coin_closes[symbol])[-1] != 0
                    else 0
                ),
                "corr": corr,
                "residual": (
                    residual / beta / np.array(self.coin_closes["BTCUSDT"])[-1]
                    if beta != 0
                    else 0
                ),
                "residual_z": residual_z,
            }
            for k, v in coef_records[symbol].items():
                self.data_logger.append(f"{symbol}_{k}", v)

        coef_df = pd.DataFrame.from_dict(coef_records, orient="index")  # index=coin

        # Select coins based on residual values
        l_coin = coef_df.sort_values(by="residual", ascending=True).index[0]
        s_coin = coef_df.sort_values(by="residual", ascending=False).index[0]

        price_l_now = coef_records[l_coin]["Close"]
        price_s_now = coef_records[s_coin]["Close"]

        beta_l = coef_records[l_coin]["beta"]
        beta_s = coef_records[s_coin]["beta"]

        residual_diff = float(coef_df.loc[l_coin, "residual"]) - float(
            coef_df.loc[s_coin, "residual"]
        )
        residual_mean = coef_df["residual"].mean()
        residual_std = coef_df["residual"].std()
        residual_z_all = coef_df["residual_z"].mean()
        residual_z_selected = (
            float(coef_df.loc[l_coin, "residual_z"])
            + float(coef_df.loc[s_coin, "residual_z"])
        ) / 2
        residual_z_selected_mns = (float(coef_df.loc[l_coin, 'residual_z']) - float(coef_df.loc[s_coin, 'residual_z'])) / 2

        residual_sign = np.abs((coef_df["residual"] > 0).sum() - 0.5 * len(coef_df))
        corr_all = coef_df["corr"].mean()
        corr_selected = (
            float(coef_df.loc[l_coin, "corr"]) + float(coef_df.loc[s_coin, "corr"])
        ) / 2
        corr_selected_mns = (float(coef_df.loc[l_coin, "corr"]) - float(coef_df.loc[s_coin, "corr"])) / 2
        coef_adj_all = coef_df["beta_adj"].mean()
        coef_adj_selected = (
            float(coef_df.loc[l_coin, "beta_adj"])
            + float(coef_df.loc[s_coin, "beta_adj"])
        ) / 2
        coef_adj_selected_mns = (float(coef_df.loc[l_coin, "beta_adj"]) - float(coef_df.loc[s_coin, "beta_adj"])) / 2

        q_l = self.bullet_size / 2 / price_l_now
        q_s = self.bullet_size / 2 / price_s_now
        self.data_logger.append("l_coin", l_coin)
        self.data_logger.append("s_coin", s_coin)
        self.data_logger.append("price_l_now", price_l_now)
        self.data_logger.append("price_s_now", price_s_now)
        self.data_logger.append("beta_l", beta_l)
        self.data_logger.append("beta_s", beta_s)
        self.data_logger.append("residual_diff", residual_diff)
        self.data_logger.append("residual_mean", residual_mean)
        self.data_logger.append("residual_std", residual_std)
        self.data_logger.append("residual_z_all", residual_z_all)
        self.data_logger.append("residual_z_selected", residual_z_selected)
        self.data_logger.append("residual_z_selected_mns", residual_z_selected_mns)
        self.data_logger.append("residual_sign", residual_sign)
        self.data_logger.append("corr_all", corr_all)
        self.data_logger.append("corr_selected", corr_selected)
        self.data_logger.append("corr_selected_mns", corr_selected_mns)
        self.data_logger.append("coef_adj_all", coef_adj_all)
        self.data_logger.append("coef_adj_selected", coef_adj_selected)
        self.data_logger.append("coef_adj_selected_mns", coef_adj_selected_mns)

        # engineer features for model prediction
        latest_paras = {}
        latest_paras.update(
            {
                "residual_diff": residual_diff,
                "residual_std": residual_std,
                "residual_mean": residual_mean,
                "residual_z_all": residual_z_all,
                "residual_z_selected": residual_z_selected,
                "residual_z_selected_mns": residual_z_selected_mns,
                "residual_sign": residual_sign,
                "corr_all": corr_all,
                "corr_selected": corr_selected,
                "corr_selected_mns": corr_selected_mns,
                "coef_adj_all": coef_adj_all,
                "coef_adj_selected": coef_adj_selected,
                "coef_adj_selected_mns": coef_adj_selected_mns,
            }
        )
        for w in self.windows:
            for stat in self.stats_list:
                if stat == "close_ret":
                    latest_paras[f"{stat}_selected_w{w}"] = (
                        self.close_ret[l_coin][w] + self.close_ret[s_coin][w]
                    ) / 2
                    latest_paras[f"{stat}_selected_mns_w{w}"] = (
                        self.close_ret[l_coin][w] - self.close_ret[s_coin][w]
                    ) / 2
                    latest_paras[f"{stat}_all_w{w}"] = np.mean(
                        [self.close_ret[symbol][w] for symbol in self.symbols]
                    )
                elif stat == "cmi":
                    latest_paras[f"{stat}_selected_w{w}"] = (
                        self.cmi[l_coin][w] + self.cmi[s_coin][w]
                    ) / 2
                    latest_paras[f"{stat}_selected_mns_w{w}"] = (
                        self.cmi[l_coin][w] - self.cmi[s_coin][w]
                    ) / 2
                    latest_paras[f"{stat}_all_w{w}"] = np.mean(
                        [self.cmi[symbol][w] for symbol in self.symbols]
                    )
                elif stat == "atr":
                    latest_paras[f"{stat}_selected_w{w}"] = (
                        self.atr[l_coin][w] + self.atr[s_coin][w]
                    ) / 2
                    latest_paras[f"{stat}_selected_mns_w{w}"] = (
                        self.atr[l_coin][w] - self.atr[s_coin][w]
                    ) / 2
                    latest_paras[f"{stat}_all_w{w}"] = np.mean(
                        [self.atr[symbol][w] for symbol in self.symbols]
                    )
                elif stat == "std":
                    latest_paras[f"{stat}_selected_w{w}"] = (
                        self.std[l_coin][w] + self.std[s_coin][w]
                    ) / 2
                    latest_paras[f"{stat}_selected_mns_w{w}"] = (
                        self.std[l_coin][w] - self.std[s_coin][w]
                    ) / 2
                    latest_paras[f"{stat}_all_w{w}"] = np.mean(
                        [self.std[symbol][w] for symbol in self.symbols]
                    )
                elif stat == "bb":
                    latest_paras[f"{stat}_selected_w{w}"] = (
                        self.bb[l_coin][w] + self.bb[s_coin][w]
                    ) / 2
                    latest_paras[f"{stat}_selected_mns_w{w}"] = (
                        self.bb[l_coin][w] - self.bb[s_coin][w]
                    ) / 2
                    latest_paras[f"{stat}_all_w{w}"] = np.mean(
                        [self.bb[symbol][w] for symbol in self.symbols]
                    )
                else:
                    latest_paras[f"{stat}_selected_w{w}"] = 0
                    latest_paras[f"{stat}_selected_mns_w{w}"] = 0
                    latest_paras[f"{stat}_all_w{w}"] = 0

        paras_df = pd.DataFrame([latest_paras])
        paras_rerank = paras_df[self.feature_order]
        for col in self.feature_order:
            paras_rerank[col] = paras_df[col]
        self.y_pred = self.model_trained.predict(paras_rerank)[0]
        self.data_logger.append("y_pred", self.y_pred)

        if (
            q_l > 0
            and q_s > 0
            and self.y_pred >25
            and len(self.orders) < self.max_concurrent_orders
            and self.freeze == 0
        ):
            self.count += 1
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

            logger.warning("Placed BUY order: %s", response_l)
            self.orders.append(
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

        orders = self.orders
        orders_to_process = list(orders)

        if self.freeze != 0:
            self.freeze += 1
        if self.freeze > self.freeze_window:
            self.freeze = 0

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

            order_profit = -(price_s_now - price_s_his) * float(order.quantity_s) + (
                price_l_now - price_l_his
            ) * float(order.quantity_l)

            if order_profit / self.bullet_size <= -self.stop_loss:
                self.freeze = 1

            logger.warning(
                "Evaluating order: %s, profit: %f, price_l_now: %f, price_s_now: %f, price_l_his: %f, price_s_his: %f, quantity_l: %f, quantity_s: %f, coin_l: %s, coin_s: %s, time_l: %s, time_s: %s",
                order,
                order_profit,
                price_l_now,
                price_s_now,
                price_l_his,
                price_s_his,
                float(order.quantity_l),
                float(order.quantity_s),
                order.symbol_l,
                order.symbol_s,
                order.time_l,
                order.time_s,
            )

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
                logger.warning("Closed LONG order: %s", response_l)
                # Close short position
                response_s = self.api.order_market(
                    order.symbol_s, OrderSide.BUY, quantity=abs(order.quantity_s)
                )
                logger.warning("Closed SHORT order: %s", response_s)
                logger.warning("Closed order: %s, profit: %f", order, order_profit)

                self.orders.remove(order)

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
        self.temps.append(self.y_pred)

        if self.t % 60 == 0:
            logger.info(
                "Logging stats at step %d: count: %d: temp: %f",
                step,
                self.count,
                self.y_pred,
            )
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

        if self.t % 10 == 0:
            self.data_logger.dump()

        cherries.log_metrics(
            {
                "assets": asset_metrics,
                # "open_orders": len(self.orders["ETHUSDT"]),
                # "price": self.api.ticker_price("ETHUSDT").price,
                "time": now.timestamp(),
                # "stats": stats,
                "positions": position_metrics,
                "y_pred": self.y_pred,
            },
            step=step,
        )


class Config(cherries.BaseConfig):
    online: bool = env.bool("ONLINE", default=False)


def main(cfg: Config) -> None:
    # cherries.log_param("group_key", "stat_arbi-usds 2026-02-14")
    # qoc.logging.init()
    api: ApiUsds
    if cfg.online:
        qoc.set_clock(qoc.ClockOnline("1m"))
        api = ApiUsdsOnline()
    else:
        # qoc.set_clock(qoc.ClockOffline("1m", start="2026-01-10", end="2026-02-07"))
        qoc.set_clock(
            qoc.ClockOffline(
                "1m",
                start="2026-02-14",
                end="2026-02-22",
            )
        )
        api = ApiUsdsOffline()

    strategy = Strategy(api=api)
    strategy.init()
    strategy.load_state()
    for _ in qoc.loop():
        try:
            strategy.step()
            strategy.log_stats()
            if cfg.online:
                strategy.dump_state()
        except Exception:
            logger.exception("")


if __name__ == "__main__":
    # cherries.main(main)
    main(Config())


# BINANCE_USDS_BASE_URL="https://fapi.binance.com" ./.venv/bin/python examples/stat_arbi-usds/main_rdft.py
