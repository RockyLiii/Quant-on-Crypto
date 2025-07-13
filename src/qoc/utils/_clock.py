import datetime
import enum
import itertools
import time
from collections.abc import Generator
from typing import Literal

from liblaf import grapes
from loguru import logger


class Align(enum.StrEnum):
    A_1S = "1s"
    UNALIGNED = enum.auto()

    def ceil(self, time: datetime.datetime) -> datetime.datetime:
        match self:
            case Align.A_1S:
                if time.microsecond > 0:
                    time += datetime.timedelta(seconds=1)
                return time.replace(microsecond=0)
            case Align.UNALIGNED:
                return time
            case v:
                raise grapes.error.MatchError(v)


type AlignLike = Align | Literal["1s"]


def clock(
    interval: datetime.timedelta = datetime.timedelta(seconds=1),
    *,
    align: AlignLike = Align.UNALIGNED,
    max_duration: datetime.timedelta | None = None,
    max_iter: int | None = None,
) -> Generator[datetime.datetime]:
    align = Align(align)
    now: datetime.datetime = datetime.datetime.now(tz=datetime.UTC)
    next_time: datetime.datetime = align.ceil(now + interval)
    for next_time in _ticks(
        interval, align=align, max_duration=max_duration, max_iter=max_iter
    ):
        now = datetime.datetime.now(tz=datetime.UTC)
        total_seconds: float = (next_time - now).total_seconds()
        if total_seconds < 0:
            logger.warning(
                "clock tick is behind schedule: {} < {}",
                next_time.isoformat(),
                now.isoformat(),
            )
            continue
        time.sleep(total_seconds)
        logger.opt(depth=1).debug("clock tick: {}", next_time.isoformat())
        yield next_time


def _ticks(
    interval: datetime.timedelta = datetime.timedelta(seconds=1),
    *,
    align: AlignLike = Align.UNALIGNED,
    max_duration: datetime.timedelta | None = None,
    max_iter: int | None = None,
) -> Generator[datetime.datetime]:
    align = Align(align)
    start: datetime.datetime = datetime.datetime.now(tz=datetime.UTC)
    now: datetime.datetime = align.ceil(start + interval)
    for it in itertools.count():
        if max_iter is not None and it >= max_iter:
            break
        if max_duration is not None and (now - start) > max_duration:
            break
        yield now
        now += interval
