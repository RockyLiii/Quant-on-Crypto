from decimal import Decimal

from pydantic_extra_types.pendulum_dt import DateTime

from qoc.typing import SymbolName

from ._base import BaseModel
from ._enum import OrderSide, PositionSide


class OrderResponse(BaseModel):
    orig_qty: Decimal
    side: OrderSide
    position_side: PositionSide
    symbol: SymbolName
    update_time: DateTime
