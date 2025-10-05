import datetime

import pendulum

from ._timestamp import TimestampUnit, TimestampUnitLike, infer_timestamp_unit

type DateTimeLike = int | datetime.datetime | pendulum.DateTime


def as_datetime(
    obj: DateTimeLike, unit: TimestampUnitLike | None = None
) -> pendulum.DateTime:
    if isinstance(obj, pendulum.DateTime):
        return obj
    if isinstance(obj, datetime.datetime):
        return pendulum.instance(obj)
    if isinstance(obj, (int, float)):
        unit = infer_timestamp_unit(obj) if unit is None else TimestampUnit(unit)
        return pendulum.from_timestamp(unit.to_seconds(obj))
    raise TypeError(type(obj))
