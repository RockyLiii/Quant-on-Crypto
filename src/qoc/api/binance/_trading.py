import datetime

import pydantic

from ._enum import OrderSide, OrderType
from ._utils import BaseModel


class OrderResponseFill(BaseModel):
    price: float
    qty: float
    commission: float
    commission_asset: str
    trade_id: int


class OrderResponseFull(BaseModel):
    symbol: str
    client_order_id: str
    transact_time: datetime.datetime
    price: float
    orig_qty: float
    executed_qty: float
    orig_quote_order_qty: float
    cummulative_quote_qty: float
    type_: OrderType = pydantic.Field(alias="type")
    side: OrderSide
    working_time: datetime.datetime
    fills: list[OrderResponseFill]
