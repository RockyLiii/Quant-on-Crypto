import httpx
import pendulum
import polars as pl
import scipy.stats
from hishel.httpx import SyncCacheClient
from liblaf import grapes
from pendulum import DateTime

from qoc.market import MarketDataFuturesUsds

METRIC_NAMES: list[str] = [
    "daa",
    "fdv",
    "fees",
    "market_cap",
    "profit",
    "rent_paid",
    "stables_mcap",
    "throughput",
    "tvl",
    "txcosts",
    "txcount",
]


def main() -> None:
    grapes.logging.init()
    chain: str = "ethereum"
    symbol: str = "ETHUSDT"
    start: DateTime = pendulum.datetime(2023, 1, 1)
    end: DateTime = pendulum.datetime(2024, 1, 1)

    market_data = MarketDataFuturesUsds()
    klines: pl.DataFrame = market_data.klines(symbol, "1d", start=start, end=end)
    price: pl.DataFrame = klines.select(
        date=pl.col("open_time").dt.date(), price=pl.col("close")
    )

    client = SyncCacheClient()
    for metric_name in METRIC_NAMES:
        response: httpx.Response = client.get(
            f"https://api.growthepie.com/v1/export/{metric_name}.json",
        )
        metric: pl.DataFrame = pl.from_records(response.json()).cast(
            {"date": pl.Date()}
        )
        metric = (
            metric.filter(pl.col("origin_key") == chain)
            .sort(pl.col("date"))
            .filter(pl.col("date").is_between(start.date(), end.date()))
        )
        metric_names: list[str] = metric["metric_key"].unique().to_list()
        metric = metric.pivot("metric_key", index="date", values="value")

        data: pl.DataFrame = price.join(metric, on="date", how="inner")
        data = data.with_columns(
            (pl.col("price").shift(-31) - pl.col("price").shift(-1)).alias(
                "price_30d_change"
            ),
            *(
                (pl.col(name).shift(1) - pl.col(name).shift(31)).alias(
                    f"{name}_30d_change"
                )
                for name in metric_names
            ),
        )
        for name in metric_names:
            valid_data: pl.DataFrame = data.drop_nulls()
            r: float
            p: float
            r, p = scipy.stats.pearsonr(
                valid_data[f"{name}_30d_change"].to_numpy(),
                valid_data["price_30d_change"].to_numpy(),
            )
            ic(name, r, p)


if __name__ == "__main__":
    main()
