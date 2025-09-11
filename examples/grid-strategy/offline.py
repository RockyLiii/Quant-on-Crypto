import datetime
from datetime import timedelta, datetime
import math
from typing import override

import attrs
import pydantic
from liblaf import cherries, grapes
from loguru import logger
import arcticdb as adb

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import qoc


class Config(cherries.BaseConfig):
    db: str = qoc.data_dir("database_offline").as_uri().replace("file://", "lmdb://")
    max_duration: timedelta | None = timedelta(hours=1)

    # strategy config
    # symbol: str = "BTCUSDT"
    quantity: float = 1e-4
    ratio: float = 0.001



    symbols: list[str] = ["BTCUSDT", "DOGEUSDT"]    
    interval: str = "1m"
    start_date: str = "2025-02-01"
    end_date: str = "2025-03-01"
    output_dir: str = "/Users/lizeyu/Desktop/qoc/tmp/raw/1m_klines_raw"

@attrs.define
class StrategyGrid(qoc.StrategySingleSymbol):
    # Config
    quantity: float = attrs.field(default=0.0, metadata={"dump": False})
    ratio: float = attrs.field(default=0.001, metadata={"dump": False})

    # State - 使用字典按symbol分组
    base_prices: dict[str, float | None] = attrs.field(factory=dict, metadata={"dump": True})

    def get_base_price(self, symbol: str) -> float | None:
        return self.base_prices.get(symbol)

    def set_base_price(self, symbol: str, price: float) -> None:
        self.base_prices[symbol] = price

    def get_price_upper(self, symbol: str) -> float:
        base_price = self.get_base_price(symbol)
        if base_price is None:
            return -math.inf
        return base_price * (1 + self.ratio)

    def get_price_lower(self, symbol: str) -> float:
        base_price = self.get_base_price(symbol)
        if base_price is None:
            return math.inf
        return base_price * (1 - self.ratio)

    @property
    def base_price(self) -> float | None:
        if not self.symbols:
            return None
        return self.get_base_price(self.symbols[0])
    
    @base_price.setter
    def base_price(self, value: float | None) -> None:
        if not self.symbols:
            return
        self.set_base_price(self.symbols[0], value)
    
    @property
    def price_upper(self) -> float:
        if not self.symbols:
            return -math.inf
        return self.get_price_upper(self.symbols[0])
    
    @property
    def price_lower(self) -> float:
        if not self.symbols:
            return math.inf
        return self.get_price_lower(self.symbols[0])

    @override
    def step(
        self, api: qoc.ApiBinance, market: qoc.Market, now: datetime
    ) -> None:
        for symbol in self.symbols:
            price: float = market.tail(symbol, n=1)["close"].iloc[-1]
            if self.get_base_price(symbol) is None:
                self.set_base_price(symbol, price)
                logger.debug("Initialize base price for {}: {}", symbol, price)
                continue
                
            logger.info("{} - Price: {}, Base price: {}, Lower bound: {}, Upper bound: {}", 
                      symbol, price, self.get_base_price(symbol), 
                      self.get_price_lower(symbol), self.get_price_upper(symbol))
                
            if price < self.get_price_lower(symbol):
                resp = api.order_market(
                    symbol, qoc.api.OrderSide.BUY, quantity=self.quantity
                )
                logger.debug("BUY > {}: {}", symbol, price)
                self.set_base_price(symbol, price)
            elif price > self.get_price_upper(symbol):
                resp = api.order_market(
                    symbol, qoc.api.OrderSide.SELL, quantity=self.quantity
                )
                logger.info("{} - close: {}; price: {}", symbol, price, resp.price)
                logger.debug("SELL > {}: {}", symbol, price)
                self.set_base_price(symbol, price)
    
    def step_offline(
        self, market, library, coins, interval, now
    ) -> None:
        for symbol in self.symbols:
            price: float = market.tail(symbol, n=1)["Close"].iloc[-1]
            if self.get_base_price(symbol) is None:
                self.set_base_price(symbol, price)
                logger.debug("Initialize base price for {}: {}", symbol, price)
                continue
                
            logger.info("{} - Price: {}, Base price: {}, Lower bound: {}, Upper bound: {}", 
                      symbol, price, self.get_base_price(symbol), 
                      self.get_price_lower(symbol), self.get_price_upper(symbol))
                
            if price < self.get_price_lower(symbol):
                logger.debug("BUY > {}: {}", symbol, price)
                self.set_base_price(symbol, price)
            elif price > self.get_price_upper(symbol):
                logger.debug("SELL > {}: {}", symbol, price)
                self.set_base_price(symbol, price)


def main(cfg: Config) -> None:
    api: qoc.ApiBinance = qoc.ApiBinance.create()
    db = qoc.Database(uri=cfg.db)
    market = qoc.Market(
        library=db.get_library("market"), symbols=cfg.symbols, interval="1m"
    )

    balance = qoc.Balance(library=db.get_library("balance"), symbols=cfg.symbols)
    strategy = StrategyGrid(
        library=db.get_library("strategy"),
        symbols=cfg.symbols,
        quantity=cfg.quantity,
        ratio=cfg.ratio,
    )

    # logger.debug(api.exchange_info(symbol=cfg.symbol))

    online = False
    if online:
        for now in qoc.clock(
            interval=timedelta(seconds=1), max_duration=cfg.max_duration
        ):

            market.step(api=api)  # get klines
            strategy.step(api=api, market=market, now=now)
            strategy.dump(now=now)
            balance.step(api=api, market=market, now=now)
    else:
        from offline_fetch import fetch_for_offline
        uri = "lmdb://examples/grid-strategy/data/database_offline/"
        ac = adb.Arctic(uri)

        library = ac.get_library('market', create_if_missing=True)
        timestamps = fetch_for_offline(cfg.symbols, cfg.interval, cfg.start_date, cfg.end_date, cfg.output_dir, library)
        
        for ts in timestamps:


            market.step_offline(library=library, coins=cfg.symbols, interval=cfg.interval, now=ts)  # get klines from local db
            strategy.step_offline(library=library, market=market, coins=cfg.symbols, interval=cfg.interval, now=ts)
            # strategy.dump(now=now)
            # balance.step_offline(library=library, market=market, coins=cfg.symbols, interval=cfg.interval, now=ts)


if __name__ == "__main__":
    cherries.run(main, profile="playground")
