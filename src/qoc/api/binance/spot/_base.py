import attrs
import binance.spot

import qoc.time_utils as tu
from qoc.api._abc import TradingApi


@attrs.define
class ApiBinanceSpotBase(TradingApi):
    client: binance.spot.Spot

    @property
    def time_unit(self) -> tu.TimestampUnit:
        """.

        All time and timestamp related fields in the JSON responses are in **milliseconds by default**.

        References:
            1. <https://developers.binance.com/docs/binance-spot-api-docs/testnet/rest-api/general-api-information>
        """
        return tu.TimestampUnit(
            self.client.session.headers.get(
                "X-MBX-TIME-UNIT", tu.TimestampUnit.MILLISECOND
            )
        )
