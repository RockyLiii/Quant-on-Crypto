import datetime
from collections.abc import Iterable, Sequence
from typing import cast

import arcticdb as adb
import attrs
import pandas as pd
import polars as pl
from liblaf import grapes

from qoc.database._types import TimeTypes

type NormalizableType = pd.DataFrame | pl.DataFrame


@attrs.define
class Library:
    library: adb.library.Library = attrs.field()

    @grapes.timer(
        cb_stop=grapes.timing.callback.log_record(level="WARNING", threshold_sec=0.0)
    )
    def append(self, symbol: str, data: NormalizableType, /, **kwargs) -> None:
        data = self._normalize_data(data)
        if self.has_symbol(symbol):
            data = data[data.index > self.get_latest_time(symbol)]
        if data.empty:
            return
        self.library.append(symbol, data, **kwargs)

    @grapes.timer(
        cb_stop=grapes.timing.callback.log_record(level="WARNING", threshold_sec=0.0)
    )
    def append_batch(
        self, data: Iterable[tuple[str, NormalizableType]], /, **kwargs
    ) -> None:
        payloads: list[adb.WritePayload] = []
        for symbol, df in data:
            df_normalized: pd.DataFrame = self._normalize_data(df)
            if not df_normalized.empty:
                payloads.append(adb.WritePayload(symbol, df_normalized))
        self.library.append_batch(payloads, **kwargs)

    def get_latest_time(self, symbol: str, /) -> datetime.datetime:
        data: pd.DataFrame = cast(
            "pd.DataFrame", self.library.tail(symbol=symbol, n=1).data
        )
        return data.index[-1]

    def has_symbol(self, symbol: str, /) -> bool:
        return self.library.has_symbol(symbol)

    def read(
        self,
        symbol: str,
        date_range: tuple[TimeTypes | None, TimeTypes | None] | None = None,
        row_range: tuple[int, int] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        return cast(
            "pd.DataFrame",
            self.library.read(
                symbol=symbol,
                date_range=date_range,
                row_range=row_range,
                columns=columns,  # pyright: ignore[reportArgumentType]
                **kwargs,
            ).data,
        )

    def tail(
        self, symbol: str, n: int = 5, columns: Sequence[str] | None = None
    ) -> pd.DataFrame:
        return cast(
            "pd.DataFrame",
            self.library.tail(symbol=symbol, n=n, columns=columns).data,  # pyright: ignore[reportArgumentType]
        )

    def _normalize_data(self, data: NormalizableType) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, pl.DataFrame):
            return data.to_pandas().set_index(data.columns[0])
        raise TypeError(data)
