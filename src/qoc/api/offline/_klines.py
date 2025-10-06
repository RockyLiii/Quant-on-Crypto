import collections
from typing import Any

import attrs
import cachetools
import hishel
import httpx
import pendulum
import polars as pl
import pooch
from polars._typing import PolarsDataType

import qoc.time_utils as tu
from qoc.api.typing import Interval, Symbol
from qoc.time_utils import DateTimeLike


@attrs.define
class KLines:
    cache: cachetools.Cache[Any, pl.DataFrame] = attrs.field(
        factory=lambda: cachetools.LRUCache(maxsize=128)
    )
    client: hishel.CacheClient = attrs.field(factory=hishel.CacheClient)

    def __call__(
        self,
        symbol: Symbol,
        interval: Interval,
        *,
        startTime: DateTimeLike,
        endTime: DateTimeLike,
    ) -> pl.DataFrame:
        start: pendulum.DateTime = tu.as_datetime(startTime)
        end: pendulum.DateTime = tu.as_datetime(endTime)
        chunks: list[pl.DataFrame] = []
        for date in pendulum.interval(
            start=start.start_of("month"), end=end.start_of("month")
        ).range("months"):
            delta: pl.DataFrame = self._klines(symbol, interval, date)
            chunks.append(delta)
        data: pl.DataFrame = pl.concat(chunks)
        data = data.drop("ignore")
        data = data.filter(
            (pl.col("open_time") >= start) & (pl.col("open_time") <= end)
        )
        return data

    @cachetools.cachedmethod(lambda self: self.cache)
    def _klines(
        self, symbol: Symbol, interval: Interval, date: DateTimeLike
    ) -> pl.DataFrame:
        interval: str = interval.replace("M", "mo")
        date: pendulum.DateTime = tu.as_datetime(date)
        date_str: str = date.strftime("%Y-%m")
        filename: str = f"{symbol}-{interval}-{date_str}"
        url: str = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{filename}.zip"
        response: httpx.Response = self.client.get(f"{url}.CHECKSUM")
        sha256: str
        sha256, *_ = response.text.split()
        files: list[str] = pooch.retrieve(
            url, f"sha256:{sha256}", processor=pooch.Unzip([f"{filename}.csv"])
        )  # pyright: ignore[reportAssignmentType]
        data: pl.DataFrame = self._read_csv(files[0])
        return data

    def _read_csv(self, file: str) -> pl.DataFrame:
        schema_csv: collections.OrderedDict[str, PolarsDataType] = (
            collections.OrderedDict(
                [
                    ("open_time", pl.Int64),
                    ("open", pl.Float64),
                    ("high", pl.Float64),
                    ("low", pl.Float64),
                    ("close", pl.Float64),
                    ("volume", pl.Float64),
                    ("close_time", pl.Int64),
                    ("quote_asset_volume", pl.Float64),
                    ("number_of_trades", pl.Int64),
                    ("taker_buy_base_asset_volume", pl.Float64),
                    ("taker_buy_quote_asset_volume", pl.Float64),
                    ("ignore", pl.String),
                ]
            )
        )
        raw: pl.DataFrame = pl.read_csv(
            file,
            has_header=False,
            new_columns=list(schema_csv.keys()),
            schema_overrides=schema_csv,
        )
        data: pl.DataFrame = raw.with_columns(
            pl.col("open_time").cast(pl.Datetime("ms", pendulum.UTC)),
            pl.col("close_time").cast(pl.Datetime("ms", pendulum.UTC)),
        )
        return data
