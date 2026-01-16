from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl
from pendulum import DateTime

import qoc.time_utils as tu
from qoc.time_utils import DateTimeLike

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

logger: logging.Logger = logging.getLogger(__name__)

KLINES_SCHEMAS: OrderedDict[str, PolarsDataType] = OrderedDict(
    [
        ("open_time", pl.Datetime(time_unit="ms", time_zone="UTC")),
        ("open", pl.Float64),
        ("high", pl.Float64),
        ("low", pl.Float64),
        ("close", pl.Float64),
        ("volume", pl.Float64),
        ("close_time", pl.Datetime(time_unit="ms", time_zone="UTC")),
        ("quote_volume", pl.Float64),
        ("count", pl.Int64),
        ("taker_buy_volume", pl.Float64),
        ("taker_buy_quote_volume", pl.Float64),
        ("ignore", pl.String),
    ]
)

KLINES_CSV_SCHEMAS: OrderedDict[str, PolarsDataType] = OrderedDict(
    [
        ("open_time", pl.Int64),
        ("open", pl.Float64),
        ("high", pl.Float64),
        ("low", pl.Float64),
        ("close", pl.Float64),
        ("volume", pl.Float64),
        ("close_time", pl.Int64),
        ("quote_volume", pl.Float64),
        ("count", pl.Int64),
        ("taker_buy_volume", pl.Float64),
        ("taker_buy_quote_volume", pl.Float64),
        ("ignore", pl.String),
    ]
)


def filter_klines(
    data: pl.DataFrame, symbol: str, interval: str, start: DateTime, end: DateTime
) -> pl.DataFrame:
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    data = data.sort("open_time").filter(
        pl.col("open_time").is_between(
            tu.as_polars_datetime(start, unit="ms"),
            tu.as_polars_datetime(end, unit="ms"),
        )
    )
    if data.is_empty():
        logger.warning(
            "klines(%s, %s, start='%s', end='%s') is empty",
            symbol,
            interval,
            start.strftime(DATE_FORMAT),
            end.strftime(DATE_FORMAT),
        )
        return data
    close_time_last: datetime = data["close_time"].last()  # pyright: ignore[reportAssignmentType]
    if close_time_last < end:
        logger.warning(
            "klines(%s, %s, start='%s', end='%s') may be incomplete, last close time is '%s'",
            symbol,
            interval,
            start.strftime(DATE_FORMAT),
            end.strftime(DATE_FORMAT),
            close_time_last.strftime(DATE_FORMAT),
        )
    return data


def parse_start_end(
    start: DateTimeLike | None, end: DateTimeLike | None, interval: str
) -> tuple[DateTime, DateTime]:
    end: DateTime = tu.now() if end is None else tu.as_datetime(end)
    if start is None:
        interval: tu.Interval = tu.Interval.parse(interval)
        start: DateTime = end - 500 * interval.duration
    else:
        start: DateTime = tu.as_datetime(start)
    return start, end
