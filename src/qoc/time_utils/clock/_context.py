import contextvars
from collections.abc import Iterator

import pendulum

from ._abc import Clock

_clock: contextvars.ContextVar[Clock] = contextvars.ContextVar("clock")


def get_clock() -> Clock:
    return _clock.get()


def loop() -> Iterator[pendulum.DateTime]:
    clock: Clock = _clock.get()
    return clock


def now() -> pendulum.DateTime:
    clock: Clock = _clock.get()
    return clock.now


def set_clock(clock: Clock) -> None:
    _clock.set(clock)
