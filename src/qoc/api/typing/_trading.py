import datetime

import pydantic

from ._base_model import BaseModel
from ._enum import OrderSide, OrderType


class OrderResponseFill(BaseModel):
    price: float
    qty: float
    commission: float
    commissionasset: str
    tradeid: int


class OrderResponseFull(BaseModel):
    symbol: str
    clientorderid: str
    transacttime: datetime.datetime
    price: float
    origqty: float
    executedqty: float
    origquoteorderqty: float
    cummulativequoteqty: float
    type: OrderType = pydantic.Field(alias="type")
    side: OrderSide
    workingtime: datetime.datetime
    fills: list[OrderResponseFill]
