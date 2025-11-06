import pydantic

from qoc.typing import SymbolName

from ._base import BaseModel


class CommissionRate(BaseModel):
    symbol: SymbolName
    maker: float = pydantic.Field(alias="makerCommissionRate")
    taker: float = pydantic.Field(alias="takerCommissionRate")
