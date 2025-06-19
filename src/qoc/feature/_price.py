from typing import override

from qoc import struct

from ._abc import PerCoinFeature


class FeaturePrice(PerCoinFeature):
    """计算价格特征."""

    @override
    def compute_per_coin(
        self, timestamp: struct.Timestamp, state: struct.State, coin: struct.CoinName
    ) -> struct.State:
        current_price: float = state.raw[coin, "low"].latest
        state.strategy.append({(coin, "price"): current_price}, timestamp)
        return state
