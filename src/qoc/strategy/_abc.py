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
        self, api: api.ApiBinanceSpot, market: market.Market, now: datetime.datetime
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
        self, api: api.ApiBinanceSpot, market: market.Market, now: datetime.datetime
    ) -> None: ...

    @override
    def dump(self, now: datetime.datetime) -> None:
        """将策略状态保存到数据库.

        将策略中标记为可保存 (dump=True) 的属性保存到数据库中, 不同类型的属性以不同方式处理:
        - 字典类型: 每个字典作为独立表保存, 字典键作为列名
        - 其他类型: 目前仅保存字典类型, 其他类型可扩展

        Args:
            now: 当前时间戳, 作为数据的时间索引
        """
        # 获取需要 dump 的属性字典
        obj_dict = attrs.asdict(self, filter=self._dump_filter)

        # 遍历顶级属性, 将每个字典类型的属性作为独立表保存
        for key, value in obj_dict.items():
            # 如果属性是字典类型, 将其作为独立表保存
            if isinstance(value, dict):
                if not value:  # 跳过空字典
                    continue

                try:
                    # 创建一个只有一行的DataFrame, 每个交易对一列
                    data_dict = {"time": now}

                    # 将每个交易对添加为一列
                    for symbol, symbol_value in value.items():
                        data_dict[symbol] = symbol_value

                    # 创建DataFrame并转换为pandas
                    dict_df = pl.DataFrame([data_dict])
                    pandas_df = dict_df.to_pandas()
                    pandas_df.set_index("time", inplace=True)

                    # 使用属性名作为表名
                    table_name = key

                    # 将数据添加到库中
                    self.library.append(table_name, pandas_df)
                    # print(f'shape of library now: {self.library.tail(table_name, n=5)}')
                except Exception as e:
                    print(f"保存 {key} 表时出错: {e}")

        # 注意: 未来可以扩展以处理其他类型的属性

    def _dump_filter(self, attr: attrs.Attribute, value: Any) -> bool:  # noqa: ARG002
        return attr.metadata.get("dump", True)
