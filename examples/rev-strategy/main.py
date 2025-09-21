import datetime
import math
import os
import sys
from typing import override

import attrs
from liblaf import cherries
from loguru import logger
from tqdm import tqdm
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import qoc 
import qoc.api as api # pyright: ignore

class Config(cherries.BaseConfig):
    db: str = qoc.data_dir("database").as_uri().replace("file://", "lmdb://")
    online: bool = False
    symbols: list[str] = ["BTCUSDT"]    
    past_threshold: float = -0.02  # 过去5小时跌幅超过2%
    past_window: int = 60*5        # 5 hours with 5-minute intervals
    # future_start: int = 0*5        # 0 hours ahead
    future_end: int = 360*5        # 30 hours ahead
    bullet_size: int = 1000      # 每次交易的金额
    max_holdings: int = 5        # 最大持仓数量


    transaction_fee: float = 0.0005  # 交易费用为 0.05%
    interval: str = "1m"
    start_date: str = "2025-02-01"
    end_date: str = "2025-03-22"
    output_dir: str = "/Users/lizeyu/Desktop/qoc/tmp/raw/1m_klines_raw"
    # limit_order_time: int = 3  # minutes

import math
import datetime
import attrs
from loguru import logger
from typing import Optional

import datetime
import attrs
import numpy as np
from collections import deque
from typing import Dict, List

@attrs.define
class StrategyRev(qoc.StrategySingleSymbol):
    # ===== Config =====
    past_window: int = 60 * 5                 # 过去窗口(分钟)
    future_end: int = 360 * 5                 # 持有期(分钟)
    bullet_size: float = 1000                 # 单次下单资金(USDT)
    past_threshold: float = -0.02             # 买入阈值(跌幅)
    max_holdings: int = 5                     # 单标最大持仓(单)

    # ===== State =====
    minute_count: int = 0                     # 计时器(分钟)
    current_value: float = 0.0                # 账户总价值
    current_holding: float = 0.0              # 持仓总市值

    order_count: Dict[str, int] = attrs.field(factory=dict, metadata={"dump": False})

    price_history: Dict[str, deque] = attrs.field(factory=dict, metadata={"dump": False})
    order_history: Dict[str, deque] = attrs.field(factory=dict, metadata={"dump": False})
    current_holdings: Dict[str, float] = attrs.field(factory=dict, metadata={"dump": False})

    value_array: List[float] = attrs.field(factory=list, metadata={"dump": False})
    holding_array: List[float] = attrs.field(factory=list, metadata={"dump": False})
    current_values: Dict[str, float] = attrs.field(factory=dict, metadata={"dump": False})

    # ===== 核心策略 =====
    def step(
        self, api: api.ApiBinance | api.ApiOffline,
        market: qoc.Market,
        now: datetime.datetime
    ) -> None:

        self.minute_count += 1

        # === 账户余额 ===
        try:
            account_balances = {
                b.asset: b.free + b.locked for b in api.account().balances
            }
        except Exception as e:
            print(f"[WARN] 获取账户余额失败: {e}")
            return

        usdt_balance = account_balances.get("USDT", 0.0)
        self.current_value = usdt_balance
        self.current_holding = 0.0

        # === 逐币处理 ===
        for symbol in self.symbols:
            # ---- 安全获取最新价格 ----
            data = market.tail(symbol, n=1)
            if data.empty or "close" not in data:
                continue
            price = float(data["close"].iloc[-1])

            # 初始化结构
            ph = self.price_history.setdefault(symbol, deque(maxlen=self.past_window + 1))
            oh = self.order_history.setdefault(symbol, deque(maxlen=self.future_end + 1))
            ch = self.current_holdings.setdefault(symbol, 0.0)

            ph.append(price)

            # 计算持仓市值
            exposure = sum(oh)
            self.current_value += exposure * price
            self.current_holding += exposure * price

            # ---- 买入信号 ----
            if len(ph) > self.past_window:
                past_price = ph[0]
                if past_price > 0:
                    past_growth = (price - past_price) / past_price
                else:
                    past_growth = np.nan

                print(f"{symbol} past growth: {past_growth:.4f}, now holdings: {ch}")
                order_count = self.order_count.get(symbol, 0)
                # 条件满足且时间到
                print(f"Order count for {symbol}: {order_count}")
                if (
                    not np.isnan(past_growth)
                    and past_growth < self.past_threshold
                    and ch < self.max_holdings
                    and self.minute_count % 5 == 0
                    and order_count < self.max_holdings

                ):
                    quantity = self.bullet_size / price
                    try:
                        api.order_market(symbol, qoc.api.OrderSide.BUY, quantity=quantity)
                        self.current_holdings[symbol] += quantity
                        order_count += 1
                        self.order_count[symbol] = order_count
                        oh.append(quantity)
                        print(f"Buy {symbol} qty={quantity:.4f} @ {price:.2f}")
                    except Exception as e:
                        print(f"[WARN] 买单失败 {symbol}: {e}")
                else:
                    oh.append(0.0)  # 无操作时也占位
            else:
                oh.append(0.0)

            # ---- 到期平仓 ----
            if len(oh) > self.future_end and oh[0] > 0:
                sell_qty = oh.popleft()
                try:
                    api.order_market(symbol, qoc.api.OrderSide.SELL, quantity=sell_qty)
                    order_count -= 1
                    self.order_count[symbol] = order_count
                    self.current_holdings[symbol] -= sell_qty
                    print(f"Sell {symbol} qty={sell_qty:.4f} @ {price:.2f}")
                except Exception as e:
                    print(f"[WARN] 卖单失败 {symbol}: {e}")
            elif len(oh) > self.future_end:
                oh.popleft()  # 清理无操作记录

        # === 记录净值 ===
        self.holding_array.append(self.current_holding)
        self.value_array.append(self.current_value)
        self.current_values["Total"] = self.current_value

        if self.minute_count % 60 == 0:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            ax1 = plt.gca()
            ax2 = ax1.twinx()

            ax1.plot(self.value_array, label='Portfolio Value', color='blue')
            ax2.plot(self.holding_array, label='Holding', color='orange')

            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Value (USDT)', color='blue')
            ax2.set_ylabel('Holding', color='orange')

            plt.title('Portfolio Value and Holding Over Time')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            plt.grid()
            plt.savefig('/Users/lizeyu/Desktop/Quant-on-Crypto/examples/rev-strategy/value.png')
            plt.close()


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
    elif interval == "5m":
        return datetime.timedelta(minutes=5)
    elif interval == "15m":
        return datetime.timedelta(minutes=15)
    elif interval == "30m":
        return datetime.timedelta(minutes=30)
    elif interval == "1h":
        return datetime.timedelta(hours=1)
    elif interval == "4h":
        return datetime.timedelta(hours=4)
    elif interval == "1d":
        return datetime.timedelta(days=1)
    else:
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
        offline_db_path = Path("examples/rev-strategy/data/database/")
        if offline_db_path.exists():
            # logger.info(f"删除现有数据库文件夹: {offline_db_path}")
            shutil.rmtree(offline_db_path)
        


        from offline_fetch import fetch_for_offline
        import arcticdb as adb

        uri = "lmdb://examples/rev-strategy/data/database_offline/"

        ac = adb.Arctic(uri)

        qoc_library = ac.get_library('market', create_if_missing=True)

        fetch_for_offline(cfg.symbols, cfg.interval, cfg.start_date, cfg.end_date, cfg.output_dir, qoc_library)

        api = qoc.ApiOffline.create(qoc_library, cfg.transaction_fee ,cfg.symbols, cfg.start_date, cfg.end_date)
        

    db = qoc.Database(uri=cfg.db)
    market = qoc.Market(
        library=db.get_library("market"), symbols=cfg.symbols, interval="1m"
    )
    balance = qoc.Balance(library=db.get_library("balance"), symbols=cfg.symbols)
    strategy = StrategyRev(
        library=db.get_library("strategy"),
        symbols=cfg.symbols,
        past_threshold=cfg.past_threshold,
        past_window=cfg.past_window,
        # future_start=cfg.future_start,
        future_end=cfg.future_end,
        bullet_size=cfg.bullet_size,
        max_holdings=cfg.max_holdings
    )

    # logger.debug(api.exchange_info(symbol=cfg.symbol))

    # 计算总步数
    total_steps = calculate_total_steps(cfg.start_date, cfg.end_date, cfg.interval)
    
    
    # 获取适当的时间增量
    time_delta = get_time_delta(cfg.interval) if not cfg.online else datetime.timedelta(seconds=0.1)
    
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
