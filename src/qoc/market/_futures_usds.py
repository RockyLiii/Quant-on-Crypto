import attrs
import polars as pl
from pendulum import DateTime

from qoc.api.usds.online import ApiUsdsOnline
from qoc.typing import SymbolName


@attrs.define
class MarketDataFuturesUsds:
    _api: ApiUsdsOnline = attrs.field(factory=ApiUsdsOnline, metadata={"dump": False})

    def klines(
        self, symbol: SymbolName, interval: str, start: DateTime, end: DateTime
    ) -> pl.DataFrame:
        return self._api.klines(symbol, interval, start_time=start, end_time=end)
