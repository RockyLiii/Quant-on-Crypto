import datetime

import pendulum

from ._unit import TimeUnit, TimeUnitLike, infer_timestamp_unit

type DateTimeLike = (
    int
    | float
    | str
    | datetime.date
    | datetime.datetime
    | pendulum.Date
    | pendulum.DateTime
)


def as_datetime(
    time: DateTimeLike,
    *,
    unit: TimeUnitLike | None = None,
    tz: datetime.tzinfo | None = pendulum.UTC,
) -> pendulum.DateTime:
    time: pendulum.DateTime = _as_datetime(time, unit=unit)
    return time.astimezone(tz)


def as_timestamp(time: DateTimeLike, unit: TimeUnitLike = TimeUnit.SECOND) -> float:
    time: pendulum.DateTime = as_datetime(time)
    unit: TimeUnit = TimeUnit(unit)
    return unit.from_seconds(time.timestamp())


def _as_datetime(
    time: DateTimeLike, *, unit: TimeUnitLike | None = None
) -> pendulum.DateTime:
    if isinstance(time, pendulum.DateTime):
        return time
    if isinstance(time, datetime.datetime):
        return pendulum.instance(time)
    if isinstance(time, pendulum.Date):
        return _date_to_datetime(time)
    if isinstance(time, str):
        parsed = pendulum.parse(time)
        if isinstance(parsed, pendulum.DateTime):
            return parsed
        if isinstance(parsed, pendulum.Date):
            return _date_to_datetime(parsed)
        raise ValueError(time)
    if isinstance(time, (int, float)):
        unit = infer_timestamp_unit(time) if unit is None else TimeUnit(unit)
        return pendulum.from_timestamp(unit.to_seconds(time))
    raise TypeError(type(time))


def _date_to_datetime(date: datetime.date) -> pendulum.DateTime:
    return pendulum.datetime(
        year=date.year, month=date.month, day=date.day, tz=pendulum.UTC
    )
