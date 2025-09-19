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
    symbols: list[str] = attrs.field(factory=list, metadata={"dump": False})

    # For backward compatibility - default to first symbol in list
    @property
    def symbol(self) -> str:
        return self.symbols[0] if self.symbols else ""

    @abc.abstractmethod
    def step(
        self, api: api.ApiBinance, market: market.Market, now: datetime.datetime
    ) -> None: ...

    @override
    def dump(self, now: datetime.datetime) -> None:
        # 获取需要 dump 的属性字典
        obj_dict = attrs.asdict(self, filter=self._dump_filter)
        
        # 创建 DataFrame
        data: pl.DataFrame = pl.from_dicts([obj_dict])
        data = data.insert_column(0, pl.Series("time", [now]))
        
        # Dump data for each symbol
        for symbol in self.symbols:
            try:
                self.library.append(symbol, data)
            except Exception:
                pass

    def _dump_filter(self, attr: attrs.Attribute, value: Any) -> bool:  # noqa: ARG002
        return attr.metadata.get("dump", True)
