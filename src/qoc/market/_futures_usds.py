import attrs
import polars as pl
from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
from pendulum import DateTime

from qoc.api.usds.online import ApiUsdsOnline
from qoc.typing import SymbolName


@attrs.define
class MarketDataFuturesUsds:
    _api: ApiUsdsOnline = attrs.field(
        factory=lambda: ApiUsdsOnline(
            base_path=DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
        ),
        metadata={"dump": False},
    )

    def klines(
        self, symbol: SymbolName, interval: str, start: DateTime, end: DateTime
    ) -> pl.DataFrame:
        return self._api.klines(symbol, interval, start_time=start, end_time=end)
