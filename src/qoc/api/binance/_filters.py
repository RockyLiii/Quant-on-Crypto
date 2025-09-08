import enum
from typing import Annotated, Literal

import pydantic

from qoc import utils
from qoc.api.typing import BaseModel


class FilterType(utils.CaseInsensitiveEnum):
    LOT_SIZE = enum.auto()
    MIN_NOTIONAL = enum.auto()
    NOTIONAL = enum.auto()


type FilterTypeLike = FilterType | Literal["LOT_SIZE", "MIN_NOTIONAL", "NOTIONAL"]


class FilterExtra(BaseModel):
    filter_type: str


class LotSize(BaseModel):
    filter_type: Literal[FilterType.LOT_SIZE] = FilterType.LOT_SIZE
    min_qty: float
    max_qty: float
    step_size: float


class MinNotional(BaseModel):
    filter_type: Literal[FilterType.MIN_NOTIONAL] = FilterType.MIN_NOTIONAL
    min_notional: float
    apply_to_market: bool
    avg_price_mins: int


class Notional(BaseModel):
    filter_type: Literal[FilterType.NOTIONAL] = FilterType.NOTIONAL
    min_notional: float
    apply_min_to_market: bool
    max_notional: float
    apply_max_to_market: bool
    avg_price_mins: int


# TODO(liblaf): add more filters
# ref: <https://developers.binance.com/docs/binance-spot-api-docs/filters>
type Filter = (
    Annotated[LotSize | MinNotional, pydantic.Field(discriminator="filter_type")]
    | FilterExtra
)
