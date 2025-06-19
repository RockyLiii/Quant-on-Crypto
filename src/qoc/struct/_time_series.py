from collections.abc import Sequence
from typing import Self, assert_never, overload

import attrs
import numpy as np
import pandas as pd
import polars as pl
import torch
from jaxtyping import Float

from ._time_index import TimeIndex, Timestamp


@attrs.frozen
class TimeSeries[T](Sequence[T]):
    name: str | Sequence[str] = attrs.field(default="value")
    time_index: TimeIndex = attrs.field(factory=TimeIndex)
    _data: list[T] = attrs.field(factory=list, alias="_data")

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Self: ...
    def __getitem__(self, index: int | slice) -> T | Self:
        if isinstance(index, int):
            return self._data[index]
        if isinstance(index, slice):
            return attrs.evolve(self, _data=self._data[index])
        assert_never()

    def __len__(self) -> int:
        return len(self._data)

    @property
    def latest(self) -> T:
        return self._data[-1] if self._data else None  # pyright: ignore[reportReturnType]

    def append(self, value: T, timestamp: Timestamp) -> None:
        """Append a new value with its corresponding timestamp to the time series.

        Args:
            value: The value to append to the time series data.
            timestamp: The timestamp associated with the value.

        Raises:
            AssertionError: In debug mode, if the timeline and data lengths become inconsistent.

        Note:
            This method maintains the invariant that the timeline and data arrays have the same length.
        """
        self.time_index.append(timestamp)
        self._data.append(value)
        if __debug__:
            assert len(self.time_index) == len(self)

    # region Exchange

    def to_numpy(self) -> Float[np.ndarray, " T"]:
        return np.asarray(self._data)

    def to_pandas(self) -> pd.DataFrame:
        raise NotImplementedError  # TODO(liblaf)

    def to_polars(self) -> pl.DataFrame:
        """Converts the time series to a Polars DataFrame.

        Returns:
            A DataFrame with two columns:

                - timestamp (pl.UInt64): Containing the timeline of the series.
                - self.name (T): Containing the data of the series.
        """
        return pl.from_dict({"timestamp": self.time_index, self.name: self._data})

    def to_torch(self) -> Float[torch.Tensor, " T"]:
        return torch.as_tensor(self._data)

    # endregion Exchange
