import enum
import functools
import re

import pendulum
from liblaf import grapes

from ._datetime import DateTimeLike, as_datetime

type IntervalLike = str | Interval


class IntervalUnit(enum.StrEnum):
    SECONDS = "s"
    MINUTES = "m"
    HOURS = "h"
    DAYS = "d"
    WEEKS = "w"
    MONTHS = "M"

    @classmethod
    def _missing_(cls, value: object) -> "IntervalUnit | None":
        if value == "mo":
            return cls.MONTHS
        return None


@grapes.attrs.frozen
class Interval:
    count: int
    unit: IntervalUnit

    @classmethod
    def parse(cls, interval: IntervalLike) -> "Interval":
        if isinstance(interval, Interval):
            return interval
        matched: re.Match[str] | None = re.fullmatch(
            r"(?P<count>\d+)(?P<unit>[a-zA-Z]+)", interval
        )
        if not matched:
            raise ValueError(interval)
        count: int = int(matched.group("count"))
        unit: IntervalUnit = IntervalUnit(matched.group("unit"))
        return cls(count=count, unit=unit)

    @functools.cached_property
    def duration(self) -> pendulum.Duration:
        match self.unit:
            case IntervalUnit.MONTHS:
                return pendulum.duration(months=self.count)
            case IntervalUnit.WEEKS:
                return pendulum.duration(weeks=self.count)
            case IntervalUnit.DAYS:
                return pendulum.duration(days=self.count)
            case IntervalUnit.HOURS:
                return pendulum.duration(hours=self.count)
            case IntervalUnit.MINUTES:
                return pendulum.duration(minutes=self.count)
            case IntervalUnit.SECONDS:
                return pendulum.duration(seconds=self.count)
            case _:
                raise grapes.MatchError(self.unit)

    @functools.cached_property
    def zero(self) -> pendulum.DateTime:
        # Monday, aligned with 3d interval
        return pendulum.datetime(year=2001, month=1, day=1)

    def to_str(self, *, case_sensitive: bool = True) -> str:
        if case_sensitive:
            return f"{self.count}{self.unit}"
        if self.unit == IntervalUnit.MONTHS:
            return f"{self.count}mo"
        return f"{self.count}{self.unit}"


def index_to_datetime(index: int, interval: IntervalLike) -> pendulum.DateTime:
    interval = Interval.parse(interval)
    match interval.unit:
        case IntervalUnit.MONTHS:
            return interval.zero.add(months=index * interval.count)
        case IntervalUnit.WEEKS:
            return interval.zero.add(weeks=index * interval.count)
        case IntervalUnit.DAYS:
            return interval.zero.add(days=index * interval.count)
        case IntervalUnit.HOURS:
            return interval.zero.add(hours=index * interval.count)
        case IntervalUnit.MINUTES:
            return interval.zero.add(minutes=index * interval.count)
        case IntervalUnit.SECONDS:
            return interval.zero.add(seconds=index * interval.count)
        case _:
            raise grapes.MatchError(interval.unit)


def datetime_to_index_floor(time: DateTimeLike, interval: IntervalLike) -> int:
    time: pendulum.DateTime = as_datetime(time)
    interval = Interval.parse(interval)
    duration: pendulum.Interval = time - interval.zero
    match interval.unit:
        case IntervalUnit.MONTHS:
            return duration.in_months() // interval.count
        case IntervalUnit.WEEKS:
            return duration.in_weeks() // interval.count
        case IntervalUnit.DAYS:
            return duration.in_days() // interval.count
        case IntervalUnit.HOURS:
            return duration.in_hours() // interval.count
        case IntervalUnit.MINUTES:
            return duration.in_minutes() // interval.count
        case IntervalUnit.SECONDS:
            return duration.in_seconds() // interval.count
        case _:
            raise grapes.MatchError(interval.unit)


def datetime_to_index_ceil(time: DateTimeLike, interval: IntervalLike) -> int:
    index_floor: int = datetime_to_index_floor(time, interval)
    time_floor: pendulum.DateTime = index_to_datetime(index_floor, interval)
    if time == time_floor:
        return index_floor
    return index_floor + 1
