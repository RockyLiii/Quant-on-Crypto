import collections
from collections.abc import Sequence

import attrs
import numpy as np
import pandas as pd
import polars as pl
import torch
from jaxtyping import Float

from ._time_index import TimeIndex
from ._time_series import TimeSeries
from .typed import Timestamp


@attrs.frozen
class TimeFrame[KT: str | Sequence[str], VT]:
    time_index: TimeIndex = attrs.field(factory=TimeIndex)
    _data: dict[KT, list[VT]] = attrs.field(
        factory=lambda: collections.defaultdict(list)
    )

    def __getitem__(self, index: KT) -> TimeSeries:
        return TimeSeries(
            name=index, time_index=self.time_index, _data=self._data[index]
        )

    def __len__(self) -> int:
        return len(self._data)

    def append(self, value: dict[KT, VT], timestamp: Timestamp) -> None:
        self.time_index.append(timestamp)
        time_cnt: int = len(self.time_index)
        idx: int = time_cnt - 1
        for k in self._data:
            self._data[k] += [None] * (time_cnt - len(self._data[k]))  # pyright: ignore[reportArgumentType]
        for k, v in value.items():
            if k not in self._data:
                self._data[k] = [None] * time_cnt  # pyright: ignore[reportArgumentType]
            self._data[k][idx] = v

    # region Exchange

    def to_numpy(self) -> Float[np.ndarray, "n T"]:
        return np.asarray(self._data.values())

    def to_pandas(self) -> pd.DataFrame:
        raise NotImplementedError  # TODO(liblaf)

    def to_polars(self) -> pl.DataFrame:
        return pl.from_dict({"timestamp": self.time_index} | self._data)

    def to_torch(self) -> Float[torch.Tensor, " T"]:
        return torch.as_tensor(self._data)

    # endregion Exchange
