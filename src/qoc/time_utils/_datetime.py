import datetime
import math
import re

import attrs
import pendulum

type DateTimeLike = int | datetime.datetime | pendulum.DateTime
type DurationLike = str | datetime.timedelta | pendulum.Duration


@attrs.define
class DurationParseError(ValueError):
    text: str

    def __str__(self) -> str:
        return f"Unable to parse duration: {self.text}"


def as_datetime(obj: DateTimeLike) -> pendulum.DateTime:
    match obj:
        case pendulum.DateTime():
            return obj
        case int():
            raise NotImplementedError("TODO(liblaf): parse s, ms, us timestamps")
        case datetime.datetime():
            return pendulum.instance(obj)
    return obj


def as_duration(obj: DurationLike) -> pendulum.Duration:
    match obj:
        case pendulum.Duration():
            return obj
        case str():
            return _parse_duration(obj)
        case datetime.timedelta():
            return pendulum.duration(seconds=obj.total_seconds())
        case _:
            return obj
    return obj


def datetime_ceil(time: DateTimeLike, duration: DurationLike) -> DateTimeLike:
    """.

    Examples:
        >>> time = pendulum.parse("2000-01-02T03:04:05.123456+08:00")
        >>> datetime_ceil(time, "1s").to_rfc3339_string()
        '2000-01-02T03:04:06+08:00'
        >>> datetime_ceil(time, "1m").to_rfc3339_string()
        '2000-01-02T03:05:00+08:00'
        >>> datetime_ceil(time, "3m").to_rfc3339_string()
        '2000-01-02T03:06:00+08:00'
        >>> datetime_ceil(time, "5m").to_rfc3339_string()
        '2000-01-02T03:05:00+08:00'
        >>> datetime_ceil(time, "15m").to_rfc3339_string()
        '2000-01-02T03:15:00+08:00'
        >>> datetime_ceil(time, "30m").to_rfc3339_string()
        '2000-01-02T03:30:00+08:00'
        >>> datetime_ceil(time, "1h").to_rfc3339_string()
        '2000-01-02T04:00:00+08:00'
        >>> datetime_ceil(time, "2h").to_rfc3339_string()
        '2000-01-02T04:00:00+08:00'
        >>> datetime_ceil(time, "4h").to_rfc3339_string()
        '2000-01-02T04:00:00+08:00'
        >>> datetime_ceil(time, "6h").to_rfc3339_string()
        '2000-01-02T08:00:00+08:00'
        >>> datetime_ceil(time, "8h").to_rfc3339_string()
        '2000-01-02T08:00:00+08:00'
        >>> datetime_ceil(time, "1d").to_rfc3339_string()
        '2000-01-02T08:00:00+08:00'
        >>> datetime_ceil(time, "3d").to_rfc3339_string()
        '2000-01-03T08:00:00+08:00'
        >>> datetime_ceil(time, "1w").to_rfc3339_string()
        '2000-01-03T08:00:00+08:00'
        >>> datetime_ceil(time, "1M").to_rfc3339_string()
        '2000-02-01T08:00:00+08:00'
    """
    time: pendulum.DateTime = as_datetime(time)
    time_utc: pendulum.DateTime = as_datetime(time).astimezone(pendulum.UTC)
    duration: pendulum.Duration = as_duration(duration)
    result_utc: pendulum.DateTime
    if duration == pendulum.duration(months=1):
        start_of_month: pendulum.DateTime = time_utc.start_of("month")
        result_utc = (
            time_utc if time_utc == start_of_month else start_of_month.add(months=1)
        )
    elif duration == pendulum.duration(weeks=1):
        start_of_week: pendulum.DateTime = time_utc.start_of("week")
        result_utc = time_utc if time_utc == start_of_week else start_of_week + duration
    elif duration < pendulum.duration(weeks=1):
        timestamp: float = time_utc.timestamp()
        duration_seconds: float = duration.total_seconds()
        result_timestamp: float = (
            math.ceil(timestamp / duration_seconds) * duration_seconds
        )
        result_utc = pendulum.from_timestamp(result_timestamp, pendulum.UTC)
    else:
        raise NotImplementedError
    return result_utc.astimezone(time.timezone)


def datetime_floor(time: DateTimeLike, duration: DurationLike) -> pendulum.DateTime:
    """.

    Examples:
        >>> time = pendulum.parse("2000-01-02T03:04:05.123456+08:00")
        >>> datetime_floor(time, "1s").to_rfc3339_string()
        '2000-01-02T03:04:05+08:00'
        >>> datetime_floor(time, "1m").to_rfc3339_string()
        '2000-01-02T03:04:00+08:00'
        >>> datetime_floor(time, "3m").to_rfc3339_string()
        '2000-01-02T03:03:00+08:00'
        >>> datetime_floor(time, "5m").to_rfc3339_string()
        '2000-01-02T03:00:00+08:00'
        >>> datetime_floor(time, "15m").to_rfc3339_string()
        '2000-01-02T03:00:00+08:00'
        >>> datetime_floor(time, "30m").to_rfc3339_string()
        '2000-01-02T03:00:00+08:00'
        >>> datetime_floor(time, "1h").to_rfc3339_string()
        '2000-01-02T03:00:00+08:00'
        >>> datetime_floor(time, "2h").to_rfc3339_string()
        '2000-01-02T02:00:00+08:00'
        >>> datetime_floor(time, "4h").to_rfc3339_string()
        '2000-01-02T00:00:00+08:00'
        >>> datetime_floor(time, "6h").to_rfc3339_string()
        '2000-01-02T02:00:00+08:00'
        >>> datetime_floor(time, "8h").to_rfc3339_string()
        '2000-01-02T00:00:00+08:00'
        >>> datetime_floor(time, "1d").to_rfc3339_string()
        '2000-01-01T08:00:00+08:00'
        >>> datetime_floor(time, "3d").to_rfc3339_string()
        '1999-12-31T08:00:00+08:00'
        >>> datetime_floor(time, "1w").to_rfc3339_string()
        '1999-12-27T08:00:00+08:00'
        >>> datetime_floor(time, "1M").to_rfc3339_string()
        '2000-01-01T08:00:00+08:00'
    """
    time: pendulum.DateTime = as_datetime(time)
    time_utc: pendulum.DateTime = as_datetime(time).astimezone(pendulum.UTC)
    duration: pendulum.Duration = as_duration(duration)
    result_utc: pendulum.DateTime
    if duration == pendulum.duration(months=1):
        result_utc = time_utc.start_of("month")
    elif duration == pendulum.duration(weeks=1):
        result_utc = time_utc.start_of("week")
    elif duration < pendulum.duration(weeks=1):
        timestamp: float = time_utc.timestamp()
        duration_seconds: float = duration.total_seconds()
        result_timestamp: float = (
            math.floor(timestamp / duration_seconds) * duration_seconds
        )
        result_utc = pendulum.from_timestamp(result_timestamp, pendulum.UTC)
    else:
        raise NotImplementedError
    return result_utc.astimezone(time.timezone)


def _parse_duration(text: str) -> pendulum.Duration:
    match = re.fullmatch(
        r"(?P<value>\d+)"
        r"\s*"
        r"(?P<unit>\w+)",
        text,
    )
    if not match:
        raise DurationParseError(text)
    value = int(match["value"])
    unit = match["unit"]
    match unit:
        case "s":
            return pendulum.duration(seconds=value)
        case "m":
            return pendulum.duration(minutes=value)
        case "h":
            return pendulum.duration(hours=value)
        case "d":
            return pendulum.duration(days=value)
        case "w":
            return pendulum.duration(weeks=value)
        case "M" | "mo":
            return pendulum.duration(months=value)
        case _:
            raise DurationParseError(text)
