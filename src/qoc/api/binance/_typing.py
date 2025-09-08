import datetime
import enum
from collections.abc import Mapping
from typing import Literal

type TimeUnitPolars = Literal["ns", "us", "ms"]


class TimeUnit(enum.StrEnum):
    MICROSECOND = enum.auto()
    MILLISECOND = enum.auto()

    @property
    def to_polars(self) -> TimeUnitPolars:
        return TIME_UNIT_TO_POLARS[self]

    def from_datetime(self, datetime: datetime.datetime) -> int:
        return int(datetime.timestamp() * TIME_UNIT_FROM_SECONDS[self])


TIME_UNIT_TO_POLARS: Mapping[TimeUnit, TimeUnitPolars] = {
    TimeUnit.MICROSECOND: "us",
    TimeUnit.MILLISECOND: "ms",
}

TIME_UNIT_FROM_SECONDS: Mapping[TimeUnit, float] = {
    TimeUnit.MICROSECOND: 10**6,
    TimeUnit.MILLISECOND: 10**3,
}
