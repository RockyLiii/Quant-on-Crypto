from typing import Any

import pydantic
from pydantic_extra_types.pendulum_dt import DateTime

from qoc.typing import SymbolName

from ._base import BaseModel
from ._mixin import DictMixin
from ._utils import get


class TickerPrice(BaseModel):
    symbol: SymbolName
    price: float
    time: DateTime


class TickerPriceDict(
    DictMixin[SymbolName, TickerPrice],
    pydantic.RootModel[dict[SymbolName, TickerPrice]],
):
    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_dict(cls, value: Any) -> dict[SymbolName, dict[str, Any]]:
        if isinstance(value, list):
            return {get(item, "symbol"): item for item in value}
        return value
