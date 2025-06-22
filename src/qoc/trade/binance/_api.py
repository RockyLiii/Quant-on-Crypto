import attrs
import binance.spot

from qoc.trade._abc import TradeApi


@attrs.define
class TradeApiBinance(TradeApi):
    client: binance.spot.Spot = attrs.field()

    def ping(self) -> None:
        self.client.ping()
