import datetime

import pendulum

from ._unit import TimeUnit, TimeUnitLike, infer_timestamp_unit

type DateTimeLike = int | datetime.datetime | pendulum.DateTime


def as_datetime(
    obj: DateTimeLike, unit: TimeUnitLike | None = None
) -> pendulum.DateTime:
    if isinstance(obj, pendulum.DateTime):
        return obj
    if isinstance(obj, datetime.datetime):
        return pendulum.instance(obj)
    if isinstance(obj, (int, float)):
        unit = infer_timestamp_unit(obj) if unit is None else TimeUnit(unit)
        return pendulum.from_timestamp(unit.to_seconds(obj))
    raise TypeError(type(obj))


def as_timestamp(time: DateTimeLike, unit: TimeUnitLike = TimeUnit.SECOND) -> float:
    time: pendulum.DateTime = as_datetime(time)
    unit: TimeUnit = TimeUnit(unit)
    return unit.from_seconds(time.timestamp())
