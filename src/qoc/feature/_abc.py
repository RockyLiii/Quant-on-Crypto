import abc
from typing import override

import attrs

from qoc import struct


@attrs.define
class Feature(abc.ABC):
    @abc.abstractmethod
    def compute(self, timestamp: struct.Timestamp, state: struct.State) -> struct.State:
        raise NotImplementedError


@attrs.define
class PerCoinFeature(Feature):
    @override
    def compute(self, timestamp: struct.Timestamp, state: struct.State) -> struct.State:
        for coin in state.coins:
            state = self.compute_per_coin(timestamp, state, coin)
        return state

    @abc.abstractmethod
    def compute_per_coin(
        self, timestamp: struct.Timestamp, state: struct.State, coin: struct.CoinName
    ) -> struct.State:
        raise NotImplementedError


@attrs.define
class GlobalFeature(Feature): ...
