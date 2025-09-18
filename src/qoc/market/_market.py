from collections.abc import Sequence

import attrs
import pandas as pd
import polars as pl

from qoc import api, database


@attrs.define
class Market:
    library: database.Library = attrs.field()
    symbols: list[str] = attrs.field(factory=lambda: ["BTCUSDT"])
    interval: api.Interval = attrs.field(default="1s")

    def convert(self, qty: float, base: str, quote: str) -> float:
        if base == quote:
            return qty
        df: pd.DataFrame = self.library.tail(f"{base}{quote}", n=1, columns=["close"])
        price: float = df["close"].iloc[-1]
        return qty * price

    def step(self, api: api.ApiBinance|api.ApiOffline) -> None:
        for symbol in self.symbols:
            klines: pl.DataFrame = api.klines(symbol, self.interval)
            # ic(klines)

            self.library.append(symbol, klines)


    def tail(
        self, symbol: str, n: int = 5, columns: Sequence[str] | None = None
    ) -> pd.DataFrame:
        return self.library.tail(symbol, n=n, columns=columns)
