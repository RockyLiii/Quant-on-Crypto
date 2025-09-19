import datetime
import math
import os
import sys
from typing import override

import attrs
from liblaf import cherries
from loguru import logger
from tqdm import tqdm


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import qoc 
import qoc.api as api # pyright: ignore

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
    end_date: str = "2025-02-04"
    output_dir: str = "/Users/lizeyu/Desktop/qoc/tmp/raw/1m_klines_raw"

    limit_order_time: int = 3  # minutes

import math
import datetime
import attrs
from loguru import logger
from typing import Optional

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
    current_value: float = attrs.field(factory=float, metadata={"dump": True})
    value_array: list[float] = attrs.field(factory=list, metadata={"dump": True})

    # ===== 辅助方法 =====
    def _ensure_symbol(self, symbol: str) -> None:
        """保证 symbol 在各个价格字典中存在"""
        if symbol not in self.base_prices:
            self.base_prices[symbol] = 0.0
        if symbol not in self.upper_prices:
            self.upper_prices[symbol] = 0.0
        if symbol not in self.lower_prices:
            self.lower_prices[symbol] = 0.0

    def set_base_price(self, symbol: str, price: Optional[float]) -> None:
        self._ensure_symbol(symbol)
        # 同步 upper/lower
        if price is not None:
            self.base_prices[symbol] = price
            self.upper_prices[symbol] = price * (1 + self.ratio)
            self.lower_prices[symbol] = price * (1 - self.ratio)
        else:
            self.upper_prices[symbol] = 0.0
            self.lower_prices[symbol] = 0.0

    def get_base_price(self, symbol: str) -> Optional[float]:
        return self.base_prices.get(symbol)

    def get_price_upper(self, symbol: str) -> float:
        upper = self.upper_prices.get(symbol)
        return upper if upper is not None else -math.inf

    def get_price_lower(self, symbol: str) -> float:
        lower = self.lower_prices.get(symbol)
        return lower if lower is not None else math.inf

    # ===== 单一 symbol 便捷属性 =====
    @property
    def base_price(self) -> Optional[float]:
        return self.get_base_price(self.symbols[0]) if self.symbols else None

    @base_price.setter
    def base_price(self, value: Optional[float]) -> None:
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
        self, api: api.ApiBinance | api.ApiOffline,
        market: qoc.Market,
        now: datetime.datetime
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
                
                self.current_value +=  pnl
                # print(f"空头持仓: {coinname} 入场价: {entry_price}, 当前价: {price}, 盈亏: {pnl}")

            # logger.info("Current price for {}: {}, base: {}, upper: {}, lower: {}",
            #             symbol, price, self.get_base_price(symbol),
            #             self.get_price_upper(symbol), self.get_price_lower(symbol))
            if self.get_base_price(symbol) is None:
                self.set_base_price(symbol, price)
                # logger.debug("Initialize base price for {}: {}", symbol, price)
                continue

            if price < self.get_price_lower(symbol):
                resp = api.order_market(symbol, qoc.api.OrderSide.BUY, quantity=self.quantity)
                # logger.debug("BUY {} @ {}; resp: {}", symbol, price, resp)
                self.set_base_price(symbol, price)

            elif price > self.get_price_upper(symbol):
                resp = api.order_market(symbol, qoc.api.OrderSide.SELL, quantity=self.quantity)
                # logger.debug("SELL {} @ {}; resp: {}", symbol, price, resp)
                self.set_base_price(symbol, price)
        # logger.info(f"CURRENT VALUE: {self.current_value}")
        self.value_array.append(self.current_value)


# 计算总步数
def calculate_total_steps(start_date, end_date, interval):
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    # 计算总天数
    days = (end - start).days
    
    # 根据间隔计算每天的步数
    if interval == "1m":
        steps_per_day = 24 * 60  # 每分钟一步
    elif interval == "5m":
        steps_per_day = 24 * 12  # 每5分钟一步
    elif interval == "15m":
        steps_per_day = 24 * 4   # 每15分钟一步
    elif interval == "1h":
        steps_per_day = 24       # 每小时一步
    else:
        steps_per_day = 24 * 60  # 默认每分钟一步
    
    return days * steps_per_day

# 在 main 函数中添加进度条
def main(cfg: Config) -> None:

    api: qoc.ApiBinance | qoc.ApiOffline

    if cfg.online:
        api = qoc.ApiBinance.create()
    else:

        import shutil
        from pathlib import Path  
        offline_db_path = Path("examples/grid-strategy/data/database/")
        if offline_db_path.exists():
            # logger.info(f"删除现有数据库文件夹: {offline_db_path}")
            shutil.rmtree(offline_db_path)
        


        from offline_fetch import fetch_for_offline
        import arcticdb as adb

        uri = "lmdb://examples/grid-strategy/data/database_offline/"


        ac = adb.Arctic(uri)

        qoc_library = ac.get_library('market', create_if_missing=True)



        fetch_for_offline(cfg.symbols, cfg.interval, cfg.start_date, cfg.end_date, cfg.output_dir, qoc_library)

        
        api = qoc.ApiOffline.create(qoc_library, cfg.transaction_fee ,cfg.symbols, cfg.start_date, cfg.end_date)
        
    library = qoc.Database(uri=cfg.db).get_library("strategy")
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
    

    # 创建进度条
    with tqdm(total=total_steps, desc="回测进度", unit="步") as pbar:
        # 初始化计时器和计数器
        import time
        step_times = {
            "api.step()": 0.0,
            "market.step()": 0.0,
            "strategy.step()": 0.0,
            "balance.step()": 0.0,
            "strategy.dump()": 0.0,
            "tqdm_update": 0.0,
        }
        iteration_count = 0
        
        for now in qoc.clock(
            interval=datetime.timedelta(seconds=1), offline=not cfg.online
        ):
            iteration_count += 1
            loop_start = time.time()
            
            # 计时 api.step()
            start_time = time.time()
            end_now = api.step()
            step_times["api.step()"] += time.time() - start_time
            
            # 计时 market.step()
            start_time = time.time()
            market.step(api=api)  # get klines
            step_times["market.step()"] += time.time() - start_time
            
            # 计时 strategy.step()
            start_time = time.time()
            strategy.step(api=api, market=market, now=now)
            step_times["strategy.step()"] += time.time() - start_time
            
            # 计时 balance.step()
            start_time = time.time()
            balance.step(api=api, market=market, now=now)
            step_times["balance.step()"] += time.time() - start_time
            
            # 计时 strategy.dump()
            start_time = time.time()
            # strategy.dump(now=now)
            step_times["strategy.dump()"] += time.time() - start_time
            
            # 计时 tqdm 更新
            start_time = time.time()
            pbar.update(1)
            pbar.set_description(f"回测时间: {now.strftime('%Y-%m-%d %H:%M')}")
            step_times["tqdm_update"] += time.time() - start_time
            
            # 每20次迭代打印一次性能分析
            if iteration_count % 20 == 0:
                value_array = strategy.value_array
                fig_path = "examples/grid-strategy/value.png"
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                plt.plot(value_array)
                plt.savefig(fig_path)
                total_time = sum(step_times.values())
                if total_time > 0:  # 避免除零错误
                    print("\n性能分析 (前 {} 次迭代):".format(iteration_count))
                    for step_name, step_time in step_times.items():
                        percentage = (step_time / total_time) * 100
                        print(f"  {step_name:<15}: {step_time:.4f}秒 ({percentage:.2f}%)")
                    print(f"  总执行时间: {total_time:.4f}秒")
                    print(f"  平均每次迭代: {total_time/iteration_count:.4f}秒")
                    print("-" * 50)
            
            # 如果回测结束，则跳出循环
            if end_now:
                break


    # 打印最终性能统计
    total_time = sum(step_times.values())
    if total_time > 0:
        print("\n最终性能分析:")
        # 按时间降序排序
        sorted_steps = sorted(step_times.items(), key=lambda x: x[1], reverse=True)
        for step_name, step_time in sorted_steps:
            percentage = (step_time / total_time) * 100
            print(f"  {step_name:<15}: {step_time:.4f}秒 ({percentage:.2f}%)")
        print(f"  总执行时间: {total_time:.4f}秒")
        print(f"  总迭代次数: {iteration_count}")
        print(f"  平均每次迭代: {total_time/iteration_count:.4f}秒")

if __name__ == "__main__":
    cherries.run(main, profile="playground")
