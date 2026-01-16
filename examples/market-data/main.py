import asyncio

import polars as pl
from liblaf import grapes

from qoc.market import (
    BinanceDataCollectionFuturesUm,
    BinanceMarketData,
    MarketDataBinanceSdkFuturesUm,
)


async def main() -> None:
    # real-time data (REST API)
    market_data: BinanceMarketData = MarketDataBinanceSdkFuturesUm()
    # historical data, faster for large date ranges, but missing recent days
    # queries are cached and stored locally on disk
    # intended for backtesting & research
    market_data: BinanceMarketData = BinanceDataCollectionFuturesUm()

    symbols: list[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    interval: str = "1m"
    start: str = "2025-01-01"
    end: str = "2026-01-01"

    # fetch klines for a single symbol
    # queries are auto chunked and parallelized internally
    data: pl.DataFrame = await market_data.klines("BTCUSDT", interval, start, end)

    with grapes.timer(label="klines()"):
        # fetch klines for multiple symbols concurrently
        klines: dict[str, pl.DataFrame] = await market_data.klines_batch(
            symbols, interval, start, end
        )
    for symbol, data in klines.items():
        print(f"--- {symbol} ---")
        print(data)


if __name__ == "__main__":
    asyncio.run(main())
