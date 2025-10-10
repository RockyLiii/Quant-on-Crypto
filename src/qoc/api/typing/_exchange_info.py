from ._base_model import BaseModel
from ._filters import Filter, FilterType, FilterTypeLike, LotSize, MinNotional, Notional


class ExchangeInfoSymbol(BaseModel):
    symbol: str
    base_asset: str
    quote_asset: str
    filters: list[Filter]

    def get_filter(self, filter_type: FilterTypeLike) -> Filter | None:
        for f in self.filters:
            if f.filter_type == filter_type:
                return f
        return None

    # region Filters

    @property
    def lot_size(self) -> LotSize | None:
        return self.get_filter(FilterType.LOT_SIZE)  # pyright: ignore[reportReturnType]

    @property
    def min_notional(self) -> MinNotional | None:
        return self.get_filter(FilterType.MIN_NOTIONAL)  # pyright: ignore[reportReturnType]

    @property
    def notional(self) -> Notional | None:
        return self.get_filter(FilterType.NOTIONAL)  # pyright: ignore[reportReturnType]

    # endregion Filters


class ExchangeInfo(BaseModel):
    symbols: list[ExchangeInfoSymbol]

    def get_symbol(self, symbol: str) -> ExchangeInfoSymbol:
        for s in self.symbols:
            if s.symbol == symbol:
                return s
        raise KeyError(symbol)
