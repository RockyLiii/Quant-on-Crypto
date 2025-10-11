import pendulum

from qoc.time_utils._datetime import DateTimeLike, as_datetime
from qoc.time_utils._interval import Interval, IntervalLike


def get_end(
    interval: IntervalLike,
    *,
    end: DateTimeLike | None = None,
    max_duration: pendulum.Duration | None = None,
    max_iter: int | None = None,
    start: DateTimeLike,
) -> pendulum.DateTime | None:
    if end is not None:
        end = as_datetime(end)
    interval: Interval = Interval.parse(interval)
    start = as_datetime(start)
    if max_duration is not None:
        new_end: pendulum.DateTime = start + max_duration
        end = min(end, new_end) if end is not None else new_end
    if max_iter is not None:
        new_end = start + interval.duration * max_iter
        end = min(end, new_end) if end is not None else new_end
    return end
