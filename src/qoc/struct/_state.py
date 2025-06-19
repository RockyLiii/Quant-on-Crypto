import attrs

from ._time_frame import TimeFrame
from .typed import CoinName


@attrs.define
class State:
    coins: list[CoinName] = attrs.field(factory=list)
    raw: TimeFrame = attrs.field(factory=TimeFrame)
    trade: TimeFrame = attrs.field(factory=TimeFrame)
    strategy: TimeFrame = attrs.field(factory=TimeFrame)
