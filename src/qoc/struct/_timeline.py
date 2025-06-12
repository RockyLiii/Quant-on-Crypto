import enum
from collections.abc import Sequence
from typing import Self, assert_never, overload

import attrs
import numpy as np
import pandas as pd
import polars as pl
import torch
from jaxtyping import UInt64


class TimeUnit(enum.StrEnum):
    MINUTE = "min"
    SECOND = "s"
    MILLISECOND = "ms"
    MICROSECOND = "us"
    NANOSECOND = "ns"

    @property
    def seconds(self) -> float:
        return _TIME_UNIT_SECONDS[self]


_TIME_UNIT_SECONDS: dict[TimeUnit, float] = {
    TimeUnit.MINUTE: 60.0,
    TimeUnit.SECOND: 1.0,
    TimeUnit.MILLISECOND: 1e-3,
    TimeUnit.MICROSECOND: 1e-6,
    TimeUnit.NANOSECOND: 1e-9,
}


type Timestamp = int


@attrs.frozen
class Timeline(Sequence[Timestamp]):
    unit: TimeUnit = TimeUnit.SECOND
    _timestamp: list[Timestamp] = attrs.field(factory=list, init=False)

    @overload
    def __getitem__(self, index: int) -> Timestamp: ...
    @overload
    def __getitem__(self, index: slice) -> Self: ...
    def __getitem__(self, index: int | slice) -> Timestamp | Self:
        if isinstance(index, int):
            return self._timestamp[index]
        if isinstance(index, slice):
            return attrs.evolve(self, _timestamp=self._timestamp[index])
        assert_never()

    def __len__(self) -> int:
        return len(self._timestamp)

    @property
    def latest(self) -> Timestamp:
        """Get the latest timestamp in the timeline.

        Returns:
            The most recent timestamp in the timeline, or 0 if the timeline is empty.
        """
        return self._timestamp[-1] if self._timestamp else 0

    def append(self, timestamp: Timestamp) -> None:
        """Append a new timestamp to the timeline.

        Args:
            timestamp: The timestamp to append to the timeline.

        Raises:
            ValueError: If timestamp is earlier than the latest timestamp (only in debug mode).

        Note:
            If the timestamp is equal to or earlier than the latest timestamp,
            the method returns without appending anything.
        """
        if __debug__ and timestamp < self.latest:
            msg: str = f"Timestamp {timestamp} is earlier than the latest timestamp {self.latest}."
            raise ValueError(msg)
        if timestamp <= self.latest:
            return
        self._timestamp.append(timestamp)

    # region Exchange

    def to_numpy(self) -> UInt64[np.ndarray, " T"]:
        return np.asarray(self._timestamp, dtype=np.uint64)

    def to_pandas(self) -> pd.DataFrame:
        raise NotImplementedError  # TODO: implement @liblaf

    def to_polars(self) -> pl.DataFrame:
        """Convert the timeline to a polars.DataFrame.

        Returns:
            timestamp (pl.UInt64): Timestamps in corresponding time unit.
        """
        return pl.DataFrame(
            {"timestamp": self._timestamp}, schema_overrides={"timestamp": pl.UInt64}
        )

    def to_torch(self) -> UInt64[torch.Tensor, " T"]:
        return torch.as_tensor(self._timestamp, dtype=torch.uint64)

    # endregion Exchange
