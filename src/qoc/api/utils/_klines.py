import polars as pl
from polars._typing import PolarsDataType


def klines_schema(
    datetime_schema: PolarsDataType | None = None,
) -> list[tuple[str, PolarsDataType]]:
    if datetime_schema is None:
        datetime_schema = pl.Datetime(time_unit="ms", time_zone="UTC")
    return [
        ("open_time", datetime_schema),
        ("open", pl.Float64),
        ("high", pl.Float64),
        ("low", pl.Float64),
        ("close", pl.Float64),
        ("volume", pl.Float64),
        ("close_time", datetime_schema),
        ("quote_asset_volume", pl.Float64),
        ("number_of_trades", pl.Int64),
        ("taker_buy_base_asset_volume", pl.Float64),
        ("taker_buy_quote_asset_volume", pl.Float64),
        ("ignore", pl.String),
    ]
