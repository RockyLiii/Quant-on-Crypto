import logging
import time
from collections import defaultdict, deque
from typing import override

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from ._base import BaseTimeline

logger = logging.getLogger(__name__)


class StatArbitrageTimeline(BaseTimeline):
    @override
    def __init__(
        self,
        starttime: float,
        endtime: float,
        feature_configs: dict[str, dict[str, dict[str, int]]],
        trading_fee: float = 0.001,
        initial_capital: float = 10000,
    ) -> None:
        super().__init__(
            starttime, endtime, feature_configs, trading_fee, initial_capital
        )

        # Initialize feature calculation parameters
        self.window_all = feature_configs["coin_features"]["price"]["window"]
        self.window_regression = feature_configs["coin_features"]["price"][
            "window_calc"
        ]
        self.residual_window = feature_configs["coin_features"]["residual"]["window"]
        self.std_rate = feature_configs["coin_features"]["residual_std"]["std_rate"]

        # Add Bollinger Band parameters
        corr_config = feature_configs["global_features"]["correlation"]

        self.corr_window = corr_config["window"]
        self.correlation_residual = []
        self.correlation_price = []
        self.correlation_residual_deque = deque(maxlen=1440)

        # Initialize feature records dictionary with defaultdict
        self.feature_records = {
            "price": defaultdict(lambda: torch.empty((0, 2), dtype=torch.float32)),
            "beta": defaultdict(lambda: torch.empty((0, 2), dtype=torch.float32)),
            "residual": defaultdict(lambda: torch.empty((0, 2), dtype=torch.float32)),
            "std": defaultdict(lambda: torch.empty((0, 2), dtype=torch.float32)),
        }

    @override
    def feature_calc(self) -> None:
        """计算所有特征."""
        feature_times = {}

        # Price特征计算时间
        start = time.time()
        self._calc_price_feature()
        feature_times["price"] = time.time() - start

        # Beta特征计算时间
        start = time.time()
        self._calc_beta_feature()
        feature_times["beta"] = time.time() - start

        # 残差特征计算时间
        start = time.time()
        self._calc_residual_feature()
        feature_times["residual"] = time.time() - start

        # Correlation特征计算时间
        start = time.time()
        self._calc_correlation_residual()
        feature_times["correlation_r"] = time.time() - start
        # Correlation特征计算时间
        start = time.time()
        self._calc_correlation_price()
        feature_times["correlation_p"] = time.time() - start

        # 仓位特征计算时间
        self._calc_position_feature()
        feature_times["position"] = time.time() - start

        # 如果数据量超过1000条，记录详细信息
        if hasattr(self, "_feature_calc_count"):
            self._feature_calc_count += 1
        else:
            self._feature_calc_count = 1

        if self._feature_calc_count % 1000 == 0:
            total_time = sum(feature_times.values())
            print("\n=== 特征计算详细耗时 ===")
            print(f"计数: {self._feature_calc_count}")
            print(f"数据大小: {len(self.data[list(self.data.keys())[0]])}")
            for feature, t in feature_times.items():
                print(f"{feature}: {t:.4f}s ({t / total_time * 100:.1f}%)")
            print("======================\n")

            # 记录数据大小与计算时间的关系
            if not hasattr(self, "_performance_log"):
                self._performance_log = []
            self._performance_log.append(
                {
                    "count": self._feature_calc_count,
                    "data_size": len(self.data[list(self.data.keys())[0]]),
                    "times": feature_times.copy(),
                }
            )

        # Calculate price and market features

        # Calculate global features

    def _record_feature(self, feature_name: str, coin: str, value: float) -> None:
        """Record feature value with timestamp."""
        record = torch.tensor([[self.current_timestamp, value]], dtype=torch.float32)
        self.feature_records[feature_name][coin] = torch.cat(
            (self.feature_records[feature_name][coin], record), dim=0
        )

    def _calc_price_feature(self) -> None:
        """计算价格特征."""
        for coin in self.coins:
            current_price = self.data[coin][-1, 4]
            self.coin_features["price"]["data"][coin].append(current_price)
            self._record_feature("price", coin, current_price)

    def _calc_correlation_residual(self) -> None:
        """Calculate correlations between residuals."""
        # Skip if insufficient coins
        if len(self.coins) < 2:
            return
        weighted_avg = 1
        # Initialize sliding window sums if not exists
        if not hasattr(self, "_corr_sums"):
            self._corr_sums = defaultdict(
                lambda: {
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "sum_xy": 0.0,
                    "sum_x2": 0.0,
                    "sum_y2": 0.0,
                    "window_values_x": deque(maxlen=self.residual_window),
                    "window_values_y": deque(maxlen=self.residual_window),
                }
            )

        valid_pairs = []

        # Calculate correlations for each pair
        for i, coin1 in enumerate(self.coins):
            if coin1 == "BTC":
                continue

            residuals1 = self.coin_features["residual"]["data"][coin1]
            if len(residuals1) < self.residual_window:
                continue

            for coin2 in self.coins[i + 1 :]:
                if coin2 == "BTC":
                    continue

                residuals2 = self.coin_features["residual"]["data"][coin2]
                if len(residuals2) < self.residual_window:
                    continue

                pair_key = f"{coin1}_{coin2}"

                # Get current residual values
                new_residual1 = residuals1[-1]
                new_residual2 = residuals2[-1]

                sums = self._corr_sums[pair_key]

                # Update sliding window
                if len(sums["window_values_x"]) == self.residual_window:
                    # Remove oldest values from sums
                    old_x = sums["window_values_x"][0]
                    old_y = sums["window_values_y"][0]
                    sums["sum_x"] -= old_x
                    sums["sum_y"] -= old_y
                    sums["sum_xy"] -= old_x * old_y
                    sums["sum_x2"] -= old_x * old_x
                    sums["sum_y2"] -= old_y * old_y

                # Add new values
                sums["window_values_x"].append(new_residual1)
                sums["window_values_y"].append(new_residual2)
                sums["sum_x"] += new_residual1
                sums["sum_y"] += new_residual2
                sums["sum_xy"] += new_residual1 * new_residual2
                sums["sum_x2"] += new_residual1 * new_residual1
                sums["sum_y2"] += new_residual2 * new_residual2

                # Calculate correlation if we have enough data
                n = len(sums["window_values_x"])
                if n >= self.residual_window:
                    try:
                        numerator = n * sums["sum_xy"] - sums["sum_x"] * sums["sum_y"]
                        denominator = np.sqrt(
                            (n * sums["sum_x2"] - sums["sum_x"] ** 2)
                            * (n * sums["sum_y2"] - sums["sum_y"] ** 2)
                        )

                        if denominator != 0:
                            corr = numerator / denominator
                            # # Calculate volatility for weighting using residual std
                            # std1 = self.coin_features['residual_std']['data'][coin1][-1]
                            # std2 = self.coin_features['residual_std']['data'][coin2][-1]
                            # weight = 1 / (std1 * std2 + 1e-8)

                            weight = 1
                            valid_pairs.append((corr, weight))
                    except:
                        continue

        # Calculate weighted average correlation
        if valid_pairs:
            correlations, weights = zip(*valid_pairs, strict=False)
            weights = np.array(weights)
            weighted_avg = np.average(correlations, weights=weights)
            self.cor_r = self.std_rate * weighted_avg + (1 - self.std_rate) * self.cor_r

            self.correlation_residual.append((self.current_timestamp, self.cor_r))
            self.correlation_residual_deque.append(self.cor_r)

    def _calc_correlation_price(self) -> None:
        """Calculate correlations between price returns using sliding window."""
        # Skip if insufficient coins
        if len(self.coins) < 2:
            return

        # Initialize sliding window sums if not exists
        if not hasattr(self, "_price_corr_sums"):
            self._price_corr_sums = defaultdict(
                lambda: {
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "sum_xy": 0.0,
                    "sum_x2": 0.0,
                    "sum_y2": 0.0,
                    "window_values_x": deque(maxlen=self.window_all),
                    "window_values_y": deque(maxlen=self.window_all),
                }
            )

        valid_pairs = []
        weighted_avg = 1

        # Calculate correlations for each pair
        for i, coin1 in enumerate(self.coins):
            if coin1 == "BTC":
                continue

            prices1 = self.coin_features["price"]["data"][coin1]
            if len(prices1) < 2:  # Need at least 2 prices for returns
                continue

            # Calculate return
            new_return1 = (prices1[-1] - prices1[-2]) / prices1[-2]

            for coin2 in self.coins[i + 1 :]:
                if coin2 == "BTC":
                    continue

                prices2 = self.coin_features["price"]["data"][coin2]
                if len(prices2) < 2:
                    continue

                # Calculate return
                new_return2 = (prices2[-1] - prices2[-2]) / prices2[-2]

                pair_key = f"{coin1}_{coin2}"
                sums = self._price_corr_sums[pair_key]

                # Update sliding window
                if len(sums["window_values_x"]) == self.window_all:
                    # Remove oldest values from sums
                    old_x = sums["window_values_x"][0]
                    old_y = sums["window_values_y"][0]
                    sums["sum_x"] -= old_x
                    sums["sum_y"] -= old_y
                    sums["sum_xy"] -= old_x * old_y
                    sums["sum_x2"] -= old_x * old_x
                    sums["sum_y2"] -= old_y * old_y

                # Add new values
                sums["window_values_x"].append(new_return1)
                sums["window_values_y"].append(new_return2)
                sums["sum_x"] += new_return1
                sums["sum_y"] += new_return2
                sums["sum_xy"] += new_return1 * new_return2
                sums["sum_x2"] += new_return1 * new_return1
                sums["sum_y2"] += new_return2 * new_return2

                # Calculate correlation if we have enough data
                n = len(sums["window_values_x"])
                if n >= self.window_all:
                    try:
                        numerator = n * sums["sum_xy"] - sums["sum_x"] * sums["sum_y"]
                        denominator = np.sqrt(
                            (n * sums["sum_x2"] - sums["sum_x"] ** 2)
                            * (n * sums["sum_y2"] - sums["sum_y"] ** 2)
                        )

                        if denominator != 0:
                            corr = numerator / denominator
                            # Weight by inverse volatility
                            # weight = 1 / (np.std(list(sums['window_values_x'])) *
                            #             np.std(list(sums['window_values_y'])) + 1e-8)
                            weight = 1
                            valid_pairs.append((corr, weight))
                    except:
                        continue

        # Calculate weighted average correlation

        if valid_pairs:
            correlations, weights = zip(*valid_pairs, strict=False)
            weights = np.array(weights)
            weighted_avg = np.average(correlations, weights=weights)
            self.cor_p = self.std_rate * weighted_avg + (1 - self.std_rate) * self.cor_p
            self.correlation_price.append((self.current_timestamp, self.cor_p))

    def _calc_beta_feature(self) -> None:
        """使用滑动窗口增量计算优化的beta计算方法."""
        # 初始化存储结构，用于保存每个币种的求和结果
        if not hasattr(self, "_beta_sums"):
            self._beta_sums = {}

        for coin in self.coins:
            if coin == "BTC" or coin not in self.data or "BTC" not in self.data:
                continue

            # 获取价格数据
            price_deque = self.coin_features["price"]["data"][coin]
            btc_price_deque = self.coin_features["price"]["data"]["BTC"]

            # 检查数据是否足够
            if (
                len(price_deque) < self.window_all
                or len(btc_price_deque) < self.window_all
            ):
                continue

            x = np.array(list(price_deque)[0 : self.window_regression]).reshape(-1, 1)
            y = np.array(list(btc_price_deque)[0 : self.window_regression]).reshape(
                -1, 1
            )

            # 检查数据有效性
            if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                continue

            # 使用无截距项的线性回归
            reg = LinearRegression(fit_intercept=False).fit(x, y)
            beta = reg.coef_[0][0]  # 获取beta值

            # 存储beta值
            self.coin_features["beta"]["data"][coin].append(beta)
            self._record_feature("beta", coin, beta)

    def _calc_residual_feature(self) -> None:
        for coin in self.coins:
            if coin == "BTC":
                continue

            # Check if we have beta value
            beta_deque = self.coin_features["beta"]["data"][coin]
            if len(beta_deque) == 0:
                continue

            # Get current prices
            current_coin_price = self.data[coin][-1, 4].item()
            current_btc_price = self.data["BTC"][-1, 4].item()

            # Get most recent beta
            beta = beta_deque[-1]

            # Calculate residual
            pred = beta * current_coin_price
            residual = current_btc_price - pred

            # Store residual
            self.coin_features["residual"]["data"][coin].append(residual)
            self._record_feature("residual", coin, residual)

        for coin in self.coins:
            if coin == "BTC":
                continue

            # Check if we have residual value
            residual_deque = self.coin_features["residual"]["data"][coin]
            if len(residual_deque) == 0:
                continue

            # Get latest residual
            residual = residual_deque[-1]

            # Get previous std or initialize
            std_deque = self.coin_features["residual_std"]["data"][coin]
            if len(std_deque) == 0:
                std_temp = 0.02  # Initial std value
            else:
                std_temp = std_deque[-1]

            # Update std using EWMA
            new_std = np.sqrt(
                std_temp**2 * (1 - self.std_rate) + residual**2 * self.std_rate
            )

            # Store new std
            self.coin_features["residual_std"]["data"][coin].append(new_std)
            self._record_feature("std", coin, new_std)

    def _calc_position_feature(self) -> None:
        """计算持仓量特征."""
        for coin in self.coins:
            if self.positions[coin].size(0) == 0:
                # No positions yet, use default values
                position_size = 0.0
            else:
                # Get latest position
                position_size = self.positions[coin][-1, 1].item()

            # Store in features
            if coin not in self.coin_features["position_size"]["data"]:
                self.coin_features["position_size"]["data"][coin] = deque(maxlen=1)
            self.coin_features["position_size"]["data"][coin].append(position_size)

        for coin in self.coins:
            # Check if we have any positions
            if self.positions[coin].size(0) == 0:
                # No positions yet, use default values
                avg_price = 0.0
            else:
                # Get latest average price
                avg_price = self.positions[coin][
                    -1, 2
                ].item()  # Using -1 to get last row, 2 for avg_price

            # Store in features
            if coin not in self.coin_features["position_avg_price"]["data"]:
                self.coin_features["position_avg_price"]["data"][coin] = deque(maxlen=1)
            self.coin_features["position_avg_price"]["data"][coin].append(avg_price)
