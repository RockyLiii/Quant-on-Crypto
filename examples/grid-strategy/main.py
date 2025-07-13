import datetime
import math
from typing import override

import attrs
import loguru
import pydantic
from environs import env
from liblaf import grapes
from loguru import logger

import qoc


class Config(pydantic.BaseModel):
    db: str = qoc.data_dir("database").as_uri().replace("file://", "lmdb://")
    max_iter: int | None = None

    # strategy config
    symbol: str = "PENGUUSDT"
    quantity: float = 50.0
    ratio: float = 0.001


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
        if price < self.price_lower:
            api.order_market(self.symbol, qoc.api.OrderSide.BUY, quantity=self.quantity)
            logger.debug("BUY > {}: {}", self.symbol, price)
            self.base_price = price
        elif price > self.price_upper:
            api.order_market(
                self.symbol, qoc.api.OrderSide.SELL, quantity=self.quantity
            )
            logger.debug("SELL > {}: {}", self.symbol, price)
            self.base_price = price


def main(cfg: Config) -> None:
    api: qoc.ApiBinance = qoc.ApiBinance.create()
    db = qoc.Database(uri=cfg.db)
    market = qoc.Market(library=db.get_library("market"), symbols=[cfg.symbol])
    balance = qoc.Balance(library=db.get_library("balance"), symbols=[cfg.symbol])
    strategy = StrategyGrid(
        library=db.get_library("strategy"),
        symbol=cfg.symbol,
        quantity=cfg.quantity,
        ratio=cfg.ratio,
    )

    logger.debug(api.exchange_info(symbol=cfg.symbol))

    for now in qoc.clock(interval=datetime.timedelta(seconds=1), max_iter=cfg.max_iter):
        market.step(api=api)
        strategy.step(api=api, market=market, now=now)
        strategy.dump(now=now)
        balance.step(api=api, market=market, now=now)


if __name__ == "__main__":
    LOGGING_FILTER: "loguru.FilterDict" = {"qoc": "DEBUG"}
    grapes.init_logging(
        handlers=[
            grapes.logging.rich_handler(filter=LOGGING_FILTER),
            grapes.logging.file_handler(
                sink=qoc.working_dir() / "run.log", filter=LOGGING_FILTER
            ),
        ],
        filter=LOGGING_FILTER,
    )
    env.read_env()
    config = Config()
    main(config)
