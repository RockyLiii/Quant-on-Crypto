import enum
from typing import Literal

from qoc import utils


class OrderType(utils.CaseInsensitiveEnum):
    LIMIT = enum.auto()
    MARKET = enum.auto()
    STOP_LOSS = enum.auto()
    STOP_LOSS_LIMIT = enum.auto()
    TAKE_PROFIT = enum.auto()
    TAKE_PROFIT_LIMIT = enum.auto()
    LIMIT_MAKER = enum.auto()


type OrderTypeLike = (
    OrderType
    | Literal[
        "LIMIT",
        "MARKET",
        "STOP_LOSS",
        "STOP_LOSS_LIMIT",
        "TAKE_PROFIT",
        "TAKE_PROFIT_LIMIT",
        "LIMIT_MAKER",
    ]
)


class OrderSide(utils.CaseInsensitiveEnum):
    BUY = enum.auto()
    SELL = enum.auto()


type OrderSideLike = OrderSide | Literal["BUY", "SELL"]
