import pendulum
import polars as pl

import qoc


def main() -> None:
    qoc.set_clock(qoc.ClockOnline("1m"))
    data_spot = qoc.MarketDataSpot()
    data_usds = qoc.MarketDataFuturesUsds()
    klines_spot: pl.DataFrame = data_spot.klines(
        "BTCUSDT",
        "1m",
        start=pendulum.datetime(2020, 1, 1),
        end=pendulum.datetime(2020, 1, 2),
    )
    klines_usds: pl.DataFrame = data_usds.klines(
        "BTCUSDT",
        "1m",
        start=pendulum.datetime(2020, 1, 1),
        end=pendulum.datetime(2020, 1, 2),
    )
    print(klines_spot)
    print(klines_usds)


if __name__ == "__main__":
    main()
