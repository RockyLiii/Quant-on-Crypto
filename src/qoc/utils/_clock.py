import datetime
import enum
import itertools
import time
from collections.abc import Generator
from typing import Literal

from liblaf import grapes
from loguru import logger


class Snap(enum.StrEnum):
    NONE = enum.auto()
    SECOND = enum.auto()

    def ceil(self, time: datetime.datetime) -> datetime.datetime:
        match self:
            case Snap.NONE:
                return time
            case Snap.SECOND:
                if time.microsecond > 0:
                    time += datetime.timedelta(seconds=1)
                return time.replace(microsecond=0)
            case v:
                raise grapes.error.MatchError(v)


type SnapLike = Snap | Literal["none", "second"]


def clock(
    interval: datetime.timedelta = datetime.timedelta(seconds=1),
    *,
    end: datetime.datetime | None = None,
    max_duration: datetime.timedelta | None = None,
    max_iter: int | None = None,
    offline: bool = False,
    snap: SnapLike = Snap.NONE,
    start: datetime.datetime | None = None,
) -> Generator[datetime.datetime]:
    snap = Snap(snap)
    if start is None:
        start = datetime.datetime.now(tz=datetime.UTC)
    start = snap.ceil(start)
    if max_duration is not None:
        # Ensure the clock does not exceed either the specified 'end' time or the maximum duration.
        if end is not None:
            end = min(end, start + max_duration)
        else:
            end = start + max_duration
    for tick in _ticks(interval, end=end, max_iter=max_iter, start=start):
        if offline:
            yield tick
            continue
        now: datetime.datetime = datetime.datetime.now(tz=datetime.UTC)
        sleep_for: float = (tick - now).total_seconds()
        if sleep_for <= 0:
            logger.warning(
                "clock tick is behind schedule (negative sleep): {} < {}",
                tick.isoformat(),
                now.isoformat(),
            )
            continue
        time.sleep(sleep_for)
        logger.debug("clock tick: {}", tick.isoformat())
        yield tick


def _ticks(
    interval: datetime.timedelta = datetime.timedelta(seconds=1),
    *,
    end: datetime.datetime | None = None,
    max_iter: int | None = None,
    start: datetime.datetime,
) -> Generator[datetime.datetime]:
    now: datetime.datetime = start
    for it in itertools.count():
        if max_iter is not None and it >= max_iter:
            break
        if end is not None and now >= end:
            break
        yield now
        now += interval
