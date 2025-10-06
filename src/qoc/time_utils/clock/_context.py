import contextvars
from collections.abc import Iterable

import pendulum

from qoc.time_utils._datetime import DateTimeLike
from qoc.time_utils._interval import IntervalLike

from ._abc import Clock
from ._offline import ClockOffline
from ._online import ClockOnline

_clock: contextvars.ContextVar[Clock] = contextvars.ContextVar("clock")


def configure(*, online: bool = True) -> None:
    if online:
        _clock.set(ClockOnline())
    else:
        _clock.set(ClockOffline())


def loop(
    interval: IntervalLike,
    *,
    end: DateTimeLike | None = None,
    max_duration: pendulum.Duration | None = None,
    max_iter: int | None = None,
    start: DateTimeLike | None = None,
) -> Iterable[pendulum.DateTime]:
    clock: Clock = _clock.get()
    return clock.loop(
        interval, end=end, max_duration=max_duration, max_iter=max_iter, start=start
    )


def now() -> pendulum.DateTime:
    clock: Clock = _clock.get()
    return clock.now()


configure(online=True)
