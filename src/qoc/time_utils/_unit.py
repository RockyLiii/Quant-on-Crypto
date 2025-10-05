import enum
from typing import Any, Literal, override

from ._datetime import DateTimeLike, as_datetime

type PolarsTimeUnit = Literal["ns", "us", "ms"]


class TimeUnit(enum.StrEnum):
    MILLISECOND = "ms"
    MICROSECOND = "us"

    @override
    @classmethod
    def _missing_(cls, value: Any) -> "TimeUnit | None":
        return _VALUE_TO_TIME_UNIT.get(value)

    @property
    def polars_time_unit(self) -> PolarsTimeUnit:
        return _TIME_UNIT_TO_POLARS_TIME_UNIT[self]

    def as_int_timestamp(self, time: DateTimeLike) -> int:
        time = as_datetime(time)
        timestamp: float = time.timestamp()
        return int(timestamp * _TIME_UNIT_TO_SECONDS[self])


_VALUE_TO_TIME_UNIT: dict[str, TimeUnit] = {
    # MILLISECOND
    "millisecond": TimeUnit.MILLISECOND,
    "MILLISECOND": TimeUnit.MILLISECOND,
    # MICROSECOND
    "microsecond": TimeUnit.MICROSECOND,
    "MICROSECOND": TimeUnit.MICROSECOND,
}

_TIME_UNIT_TO_POLARS_TIME_UNIT: dict[TimeUnit, PolarsTimeUnit] = {
    TimeUnit.MILLISECOND: "ms",
    TimeUnit.MICROSECOND: "us",
}

_TIME_UNIT_TO_SECONDS: dict[TimeUnit, float] = {
    TimeUnit.MILLISECOND: 1e3,
    TimeUnit.MICROSECOND: 1e6,
}
