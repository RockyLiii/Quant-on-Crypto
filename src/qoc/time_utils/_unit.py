import enum
from typing import Any, Literal, override

import pendulum
from loguru import logger

type PolarsTimeUnit = Literal["ns", "us", "ms"]
type TimeUnitLike = Literal["s", "ms", "us", "ns"] | TimeUnit | str  # noqa: PYI051


class TimeUnit(enum.StrEnum):
    SECOND = "s"
    MILLISECOND = "ms"
    MICROSECOND = "us"
    NANOSECOND = "ns"

    @override
    @classmethod
    def _missing_(cls, value: Any) -> "TimeUnit | None":
        if not isinstance(value, str):
            return None
        value = value.lower()
        return _VALUE_TO_TIME_UNIT.get(value)

    @property
    def polars_time_unit(self) -> PolarsTimeUnit:
        return _TIME_UNIT_TO_POLARS_TIME_UNIT[self]

    def from_seconds(self, timestamp: float) -> float:
        return timestamp * _TIME_UNIT_TO_SECONDS[self]

    def to_seconds(self, timestamp: float) -> float:
        return timestamp / _TIME_UNIT_TO_SECONDS[self]


_VALUE_TO_TIME_UNIT: dict[str, TimeUnit] = {
    "s": TimeUnit.SECOND,
    "second": TimeUnit.SECOND,
    "ms": TimeUnit.MILLISECOND,
    "millisecond": TimeUnit.MILLISECOND,
    "us": TimeUnit.MICROSECOND,
    "microsecond": TimeUnit.MICROSECOND,
    "ns": TimeUnit.NANOSECOND,
    "nanosecond": TimeUnit.NANOSECOND,
}

_TIME_UNIT_TO_POLARS_TIME_UNIT: dict[TimeUnit, PolarsTimeUnit] = {
    TimeUnit.MILLISECOND: "ms",
    TimeUnit.MICROSECOND: "us",
    TimeUnit.NANOSECOND: "ns",
}

_TIME_UNIT_TO_SECONDS: dict[TimeUnit, float] = {
    TimeUnit.SECOND: 1.0,
    TimeUnit.MILLISECOND: 1e3,
    TimeUnit.MICROSECOND: 1e6,
    TimeUnit.NANOSECOND: 1e9,
}


_DATETIME_MAX_TIMESTAMP: float = pendulum.DateTime.max.timestamp()


def infer_timestamp_unit(timestamp: float) -> TimeUnit:
    logger.warning("Inferring timestamp unit is not secure.", once=True)
    if timestamp < _DATETIME_MAX_TIMESTAMP:
        return TimeUnit.SECOND
    if timestamp / 1e3 < _DATETIME_MAX_TIMESTAMP:
        return TimeUnit.MILLISECOND
    if timestamp / 1e6 < _DATETIME_MAX_TIMESTAMP:
        return TimeUnit.MICROSECOND
    if timestamp / 1e9 < _DATETIME_MAX_TIMESTAMP:
        return TimeUnit.NANOSECOND
    raise ValueError(timestamp)
