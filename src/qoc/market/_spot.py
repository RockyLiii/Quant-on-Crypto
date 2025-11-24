import attrs
import polars as pl
from binance_common.constants import SPOT_REST_API_MARKET_URL
from pendulum import DateTime

from qoc.api.binance.spot import ApiBinanceSpot
from qoc.typing import SymbolName


@attrs.define
class MarketDataSpot:
    _api: ApiBinanceSpot = attrs.field(
        factory=lambda: ApiBinanceSpot(base_url=SPOT_REST_API_MARKET_URL),
        metadata={"dump": False},
    )

    def klines(
        self, symbol: SymbolName, interval: str, start: DateTime, end: DateTime
    ) -> pl.DataFrame:
        return self._api.klines(symbol, interval, startTime=start, endTime=end)
