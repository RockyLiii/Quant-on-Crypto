import decimal
from decimal import Decimal
from typing import Annotated, Literal

import pydantic

from ._base import BaseModel


class SymbolFilterBase(BaseModel):
    filter_type: str


class LotSize(BaseModel):
    filter_type: Literal["LOT_SIZE"] = "LOT_SIZE"  # pyright: ignore[reportIncompatibleVariableOverride]
    min_qty: Decimal
    max_qty: Decimal
    step_size: Decimal

    def round(
        self,
        quantity: Decimal | float | str,
        rounding: str | None = decimal.ROUND_HALF_EVEN,
        context: decimal.Context | None = None,
    ) -> Decimal:
        quantity = Decimal(quantity)
        steps: Decimal = (quantity - self.min_qty) / self.step_size
        steps = steps.quantize(1, rounding=rounding, context=context)
        return self.min_qty + steps * self.step_size


class MarketLotSize(BaseModel):
    filter_type: Literal["MARKET_LOT_SIZE"] = "MARKET_LOT_SIZE"  # pyright: ignore[reportIncompatibleVariableOverride]
    min_qty: Decimal
    max_qty: Decimal
    step_size: Decimal

    def round(
        self,
        quantity: Decimal | float | str,
        rounding: str | None = decimal.ROUND_HALF_EVEN,
        context: decimal.Context | None = None,
    ) -> Decimal:
        quantity = Decimal(quantity)
        steps: Decimal = max(quantity - self.min_qty, Decimal(0)) / self.step_size
        steps = steps.quantize(1, rounding=rounding, context=context)
        return self.min_qty + steps * self.step_size


type SymbolFilter = (
    Annotated[LotSize | MarketLotSize, pydantic.Field(discriminator="filter_type")]
    | SymbolFilterBase
)
