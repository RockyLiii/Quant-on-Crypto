import datetime
import math
import os
import sys

import attrs
from liblaf import cherries
from tqdm import tqdm

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import qoc
from qoc import api  # pyright: ignore


class Config(cherries.BaseConfig):
    db: str = qoc.data_dir("database").as_uri().replace("file://", "lmdb://")
    # max_duration: datetime.timedelta | None = datetime.timedelta(hours=999999)

    # strategy config

    online: bool = False
    symbols: list[str] = ["BTCUSDT", "DOGEUSDT"]
    quantity: float = 1e-3
    ratio: float = 0.001

    # symbols: list[str] = ["BTCUSDT", "DOGEUSDT"]
    transaction_fee: float = 0.01  # 交易费用为0.1%
    interval: str = "1m"
    start_date: str = "2025-02-01"
    end_date: str = "2025-02-02"
    output_dir: str = "/Users/lizeyu/Desktop/qoc/tmp/raw/1m_klines_raw"

    limit_order_time: int = 3  # minutes


@attrs.define
class StrategyGrid(qoc.StrategySingleSymbol):
    # ===== Config =====
    quantity: float = attrs.field(default=0.0, metadata={"dump": False})
    ratio: float = attrs.field(default=0.1, metadata={"dump": False})

    # ===== State =====
    # 使用一维结构存储价格信息
    base_prices: dict[str, float] = attrs.field(factory=dict, metadata={"dump": True})
    upper_prices: dict[str, float] = attrs.field(factory=dict, metadata={"dump": True})
    lower_prices: dict[str, float] = attrs.field(factory=dict, metadata={"dump": True})
    current_values: dict[str, float] = attrs.field(
        factory=dict, metadata={"dump": True}
    )

    current_value: float = attrs.field(factory=float, metadata={"dump": False})
    value_array: list[float] = attrs.field(factory=list, metadata={"dump": False})

    # ===== 辅助方法 =====
    def _ensure_symbol(self, symbol: str) -> None:
        """保证 symbol 在各个价格字典中存在"""
        if symbol not in self.base_prices:
            self.base_prices[symbol] = 0.0
        if symbol not in self.upper_prices:
            self.upper_prices[symbol] = 0.0
        if symbol not in self.lower_prices:
            self.lower_prices[symbol] = 0.0

    def set_base_price(self, symbol: str, price: float | None) -> None:
        self._ensure_symbol(symbol)
        # 同步 upper/lower
        if price is not None:
            self.base_prices[symbol] = price
            self.upper_prices[symbol] = price * (1 + self.ratio)
            self.lower_prices[symbol] = price * (1 - self.ratio)
        else:
            self.upper_prices[symbol] = 0.0
            self.lower_prices[symbol] = 0.0

    def get_base_price(self, symbol: str) -> float | None:
        return self.base_prices.get(symbol)

    def get_price_upper(self, symbol: str) -> float:
        upper = self.upper_prices.get(symbol)
        return upper if upper is not None else -math.inf

    def get_price_lower(self, symbol: str) -> float:
        lower = self.lower_prices.get(symbol)
        return lower if lower is not None else math.inf

    # ===== 单一 symbol 便捷属性 =====
    @property
    def base_price(self) -> float | None:
        return self.get_base_price(self.symbols[0]) if self.symbols else None

    @base_price.setter
    def base_price(self, value: float | None) -> None:
        if self.symbols:
            self.set_base_price(self.symbols[0], value)

    @property
    def price_upper(self) -> float:
        return self.get_price_upper(self.symbols[0]) if self.symbols else -math.inf

    @property
    def price_lower(self) -> float:
        return self.get_price_lower(self.symbols[0]) if self.symbols else math.inf

    # ===== 核心策略 =====
    def step(
        self,
        api: api.ApiBinance | api.ApiOffline,
        market: qoc.Market,
        now: datetime.datetime,
    ) -> None:
        current_balance: dict[str, float] = {
            b.asset: b.free + b.locked for b in api.account().balances
        }
        current_price: dict[str, float] = {}

        self.current_value = current_balance.get("USDT", 0.0)

        for symbol in self.symbols:
            price: float = market.tail(symbol, n=1)["close"].iloc[-1]

            coinname = symbol.replace("USDT", "")
            current_price[coinname] = price

            if current_balance.get(coinname, 0.0) > 0:
                # 多头持仓：当前价值 = 数量 * 当前价格
                self.current_value += current_balance[coinname] * price
                # print(f"多头持仓: {coinname} = {current_balance[coinname]} * {price} = {current_balance[coinname] * price}")
            elif current_balance.get(coinname, 0.0) < 0:
                # 空头持仓：当前价值 = 锁定保证金 + 盈亏
                # 盈亏 = 持仓数量 * (做空均价 - 当前价格)
                position_size = abs(current_balance.get(coinname, 0.0))
                entry_price = api.symbol_average_price.get(coinname, price)
                pnl = position_size * (entry_price - price)

                self.current_value += pnl
                # print(f"空头持仓: {coinname} 入场价: {entry_price}, 当前价: {price}, 盈亏: {pnl}")

            # logger.info("Current price for {}: {}, base: {}, upper: {}, lower: {}",
            #             symbol, price, self.get_base_price(symbol),
            #             self.get_price_upper(symbol), self.get_price_lower(symbol))
            if self.get_base_price(symbol) is None:
                self.set_base_price(symbol, price)
                # logger.debug("Initialize base price for {}: {}", symbol, price)
                continue

            if price < self.get_price_lower(symbol):
                resp = api.order_market(
                    symbol, qoc.api.OrderSide.BUY, quantity=self.quantity
                )
                # logger.debug("BUY {} @ {}; resp: {}", symbol, price, resp)
                self.set_base_price(symbol, price)

            elif price > self.get_price_upper(symbol):
                resp = api.order_market(
                    symbol, qoc.api.OrderSide.SELL, quantity=self.quantity
                )
                # logger.debug("SELL {} @ {}; resp: {}", symbol, price, resp)
                self.set_base_price(symbol, price)
        # logger.info(f"CURRENT VALUE: {self.current_value}")
        self.value_array.append(self.current_value)

        if "Total" not in self.current_values:
            self.current_values["Total"] = 0.0
        self.current_values["Total"] = self.current_value


# 计算总步数
def calculate_total_steps(start_date, end_date, interval):
    """计算从开始日期到结束日期间的总步数"""
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    # 计算总时间范围（秒）
    total_seconds = (end - start).total_seconds()

    # 获取每步的时间增量
    time_delta = get_time_delta(interval)
    step_seconds = time_delta.total_seconds()

    # 计算总步数
    total_steps = int(total_seconds / step_seconds)

    return total_steps


# 创建一个函数来根据interval获取时间增量
def get_time_delta(interval: str) -> datetime.timedelta:
    """根据间隔字符串返回相应的时间增量"""
    if interval == "1m":
        return datetime.timedelta(minutes=1)
    if interval == "5m":
        return datetime.timedelta(minutes=5)
    if interval == "15m":
        return datetime.timedelta(minutes=15)
    if interval == "30m":
        return datetime.timedelta(minutes=30)
    if interval == "1h":
        return datetime.timedelta(hours=1)
    if interval == "4h":
        return datetime.timedelta(hours=4)
    if interval == "1d":
        return datetime.timedelta(days=1)
    # 默认为1分钟
    return datetime.timedelta(minutes=1)


# 在 main 函数中添加进度条
def main(cfg: Config) -> None:
    api: qoc.ApiBinance | qoc.ApiOffline

    if cfg.online:
        api = qoc.ApiBinance.create()
    else:
        import shutil
        from pathlib import Path

        offline_db_path = Path("strategies/grid-strategy/data/database/")
        if offline_db_path.exists():
            # logger.info(f"删除现有数据库文件夹: {offline_db_path}")
            shutil.rmtree(offline_db_path)

        import arcticdb as adb
        from offline_fetch import fetch_for_offline

        uri = "lmdb://strategies/grid-strategy/data/database_offline/"

        ac = adb.Arctic(uri)

        qoc_library = ac.get_library("market", create_if_missing=True)

        fetch_for_offline(
            cfg.symbols,
            cfg.interval,
            cfg.start_date,
            cfg.end_date,
            cfg.output_dir,
            qoc_library,
        )

        api = qoc.ApiOffline.create(
            qoc_library, cfg.transaction_fee, cfg.symbols, cfg.start_date, cfg.end_date
        )

    db = qoc.Database(uri=cfg.db)
    market = qoc.Market(
        library=db.get_library("market"), symbols=cfg.symbols, interval="1m"
    )
    balance = qoc.Balance(library=db.get_library("balance"), symbols=cfg.symbols)
    strategy = StrategyGrid(
        library=db.get_library("strategy"),
        symbols=cfg.symbols,
        quantity=cfg.quantity,
        ratio=cfg.ratio,
    )

    # logger.debug(api.exchange_info(symbol=cfg.symbol))

    # 计算总步数
    total_steps = calculate_total_steps(cfg.start_date, cfg.end_date, cfg.interval)

    # 获取适当的时间增量
    time_delta = (
        get_time_delta(cfg.interval)
        if not cfg.online
        else datetime.timedelta(seconds=0.1)
    )

    # 创建进度条
    with tqdm(total=total_steps, desc="回测进度", unit="步") as pbar:
        # 跟踪已处理的步数
        step_count = 0

        if not cfg.online:
            now_ts = datetime.datetime.strptime(cfg.start_date, "%Y-%m-%d")

        for now in qoc.clock(
            interval=datetime.timedelta(seconds=0.1), offline=not cfg.online
        ):
            ts = now if cfg.online else now_ts
            # 执行回测逻辑
            end_now = api.step()  # 确保调用了api.step()获取结束状态

            market.step(api=api)  # get klines
            strategy.step(api=api, market=market, now=ts)
            balance.step(api=api, market=market, now=ts)
            strategy.dump(now=ts)

            # 更新进度条
            step_count += 1
            pbar.update(1)
            pbar.set_description(f"回测时间: {ts.strftime('%Y-%m-%d %H:%M')}")

            # 使用动态时间增量
            now_ts += time_delta

            # 检查是否达到最大步数或回测结束
            if end_now:
                break


if __name__ == "__main__":
    cherries.run(main, profile="playground")
