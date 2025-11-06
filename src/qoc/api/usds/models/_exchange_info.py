from typing import Any, cast

import pydantic

from qoc.typing import AssetName, SymbolName

from ._base import BaseModel
from ._mixin import DictMixin
from ._symbol_filters import LotSize, MarketLotSize, SymbolFilter
from ._utils import get


class ExchangeInfoSymbol(BaseModel):
    symbol: SymbolName
    base_asset: AssetName
    quote_asset: AssetName
    margin_asset: AssetName
    filters: list[SymbolFilter]

    @property
    def lot_size(self) -> LotSize | None:
        for f in self.filters:
            if f.filter_type == "LOT_SIZE":
                return cast("LotSize", f)
        return None

    @property
    def market_lot_size(self) -> MarketLotSize | None:
        for f in self.filters:
            if f.filter_type == "MARKET_LOT_SIZE":
                return cast("MarketLotSize", f)
        return None


class ExchangeInfoSymbolDict(
    DictMixin[SymbolName, ExchangeInfoSymbol],
    pydantic.RootModel[dict[SymbolName, ExchangeInfoSymbol]],
):
    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_dict(cls, value: Any) -> dict[SymbolName, dict[str, Any]]:
        if isinstance(value, list):
            return {get(item, "symbol"): item for item in value}
        return value


class ExchangeInfo(BaseModel):
    symbols: ExchangeInfoSymbolDict
