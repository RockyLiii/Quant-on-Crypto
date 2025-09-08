import datetime
import math
from typing import override

import attrs
import pydantic
from liblaf import cherries, grapes
from loguru import logger

import qoc


class Config(cherries.BaseConfig):
    db: str = qoc.data_dir("database").as_uri().replace("file://", "lmdb://")
    max_duration: datetime.timedelta | None = datetime.timedelta(hours=1)

    # strategy config
    symbol: str = "BTCUSDT"
    quantity: float = 1e-4
    ratio: float = 0.0


@attrs.define
class StrategyGrid(qoc.StrategySingleSymbol):
    # Config
    quantity: float = attrs.field(metadata={"dump": False})
    ratio: float = attrs.field(default=0.01, metadata={"dump": False})

    # State
    base_price: float | None = attrs.field(default=None, metadata={"dump": True})

    @property
    def price_upper(self) -> float:
        if self.base_price is None:
            return -math.inf
        return self.base_price * (1 + self.ratio)

    @property
    def price_lower(self) -> float:
        if self.base_price is None:
            return math.inf
        return self.base_price * (1 - self.ratio)

    @override
    def step(
        self, api: qoc.ApiBinance, market: qoc.Market, now: datetime.datetime
    ) -> None:
        price: float = market.tail(self.symbol, n=1)["close"].iloc[-1]
        if self.base_price is None:
            self.base_price = price
            logger.debug("Initialize base price: {}", self.base_price)
            return
        logger.info("close: {}", price)
        if price < self.price_lower:
            resp = api.order_market(
                self.symbol, qoc.api.OrderSide.BUY, quantity=self.quantity
            )
            logger.debug("BUY > {}: {}", self.symbol, price)
            self.base_price = price
        elif price > self.price_upper:
            resp = api.order_market(
                self.symbol, qoc.api.OrderSide.SELL, quantity=self.quantity
            )
            logger.info("close: {}; price: {}", price, resp.price)
            logger.debug("SELL > {}: {}", self.symbol, price)
            self.base_price = price


def main(cfg: Config) -> None:
    api: qoc.ApiBinance = qoc.ApiBinance.create()
    db = qoc.Database(uri=cfg.db)
    market = qoc.Market(
        library=db.get_library("market"), symbols=[cfg.symbol], interval="1s"
    )
    balance = qoc.Balance(library=db.get_library("balance"), symbols=[cfg.symbol])
    strategy = StrategyGrid(
        library=db.get_library("strategy"),
        symbol=cfg.symbol,
        quantity=cfg.quantity,
        ratio=cfg.ratio,
    )

    logger.debug(api.exchange_info(symbol=cfg.symbol))

    for now in qoc.clock(
        interval=datetime.timedelta(seconds=1), max_duration=cfg.max_duration
    ):
        market.step(api=api)  # get klines
        strategy.step(api=api, market=market, now=now)
        strategy.dump(now=now)
        balance.step(api=api, market=market, now=now)


if __name__ == "__main__":
    cherries.run(main, profile="playground")
