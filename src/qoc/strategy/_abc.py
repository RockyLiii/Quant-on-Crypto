import abc
import datetime
from typing import Any, override

import attrs
import polars as pl

from qoc import api, database, market


@attrs.define
class Strategy(abc.ABC):
    library: database.Library = attrs.field(metadata={"dump": False})

    @abc.abstractmethod
    def step(
        self, api: api.ApiBinance, market: market.Market, now: datetime.datetime
    ) -> None: ...

    @abc.abstractmethod
    def dump(self, now: datetime.datetime) -> None: ...


@attrs.define
class StrategySingleSymbol(Strategy):
    symbol: str = attrs.field(metadata={"dump": False})

    @abc.abstractmethod
    def step(
        self, api: api.ApiBinance, market: market.Market, now: datetime.datetime
    ) -> None: ...

    @override
    def dump(self, now: datetime.datetime) -> None:
        data: pl.DataFrame = pl.from_dicts(
            [attrs.asdict(self, filter=self._dump_filter)]
        )
        data = data.insert_column(0, pl.Series("time", [now]))
        self.library.append(self.symbol, data)

    def _dump_filter(self, attr: attrs.Attribute, value: Any) -> bool:  # noqa: ARG002
        return attr.metadata.get("dump", True)
