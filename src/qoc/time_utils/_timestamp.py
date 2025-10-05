import enum
from typing import Any, Literal, override

import pendulum
from loguru import logger

from ._datetime import DateTimeLike, as_datetime

type PolarsTimeUnit = Literal["ns", "us", "ms"]
type TimestampUnitLike = Literal["s", "ms", "us", "ns"] | TimestampUnit | str  # noqa: PYI051


class TimestampUnit(enum.StrEnum):
    SECOND = "s"
    MILLISECOND = "ms"
    MICROSECOND = "us"
    NANOSECOND = "ns"

    @override
    @classmethod
    def _missing_(cls, value: Any) -> "TimestampUnit | None":
        if not isinstance(value, str):
            return None
        value = value.lower()
        return _VALUE_TO_TIME_UNIT.get(value)

    @property
    def polars_time_unit(self) -> PolarsTimeUnit:
        return _TIME_UNIT_TO_POLARS_TIME_UNIT[self]

    def as_int_timestamp(self, time: DateTimeLike) -> int:
        time = as_datetime(time)
        timestamp: float = time.timestamp()
        return round(timestamp * _TIME_UNIT_TO_SECONDS[self])

    def to_seconds(self, timestamp: float) -> float:
        return timestamp / _TIME_UNIT_TO_SECONDS[self]


_VALUE_TO_TIME_UNIT: dict[str, TimestampUnit] = {
    "s": TimestampUnit.SECOND,
    "second": TimestampUnit.SECOND,
    "ms": TimestampUnit.MILLISECOND,
    "millisecond": TimestampUnit.MILLISECOND,
    "us": TimestampUnit.MICROSECOND,
    "microsecond": TimestampUnit.MICROSECOND,
    "ns": TimestampUnit.NANOSECOND,
    "nanosecond": TimestampUnit.NANOSECOND,
}

_TIME_UNIT_TO_POLARS_TIME_UNIT: dict[TimestampUnit, PolarsTimeUnit] = {
    TimestampUnit.MILLISECOND: "ms",
    TimestampUnit.MICROSECOND: "us",
    TimestampUnit.NANOSECOND: "ns",
}

_TIME_UNIT_TO_SECONDS: dict[TimestampUnit, float] = {
    TimestampUnit.SECOND: 1.0,
    TimestampUnit.MILLISECOND: 1e3,
    TimestampUnit.MICROSECOND: 1e6,
    TimestampUnit.NANOSECOND: 1e9,
}


_DATETIME_MAX_TIMESTAMP: float = pendulum.DateTime.max.timestamp()


def infer_timestamp_unit(timestamp: float) -> TimestampUnit:
    logger.warning("Inferring timestamp unit is not secure.", once=True)
    if timestamp < _DATETIME_MAX_TIMESTAMP:
        return TimestampUnit.SECOND
    if timestamp / 1e3 < _DATETIME_MAX_TIMESTAMP:
        return TimestampUnit.MILLISECOND
    if timestamp / 1e6 < _DATETIME_MAX_TIMESTAMP:
        return TimestampUnit.MICROSECOND
    if timestamp / 1e9 < _DATETIME_MAX_TIMESTAMP:
        return TimestampUnit.NANOSECOND
    raise ValueError(timestamp)
