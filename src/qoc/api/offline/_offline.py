from datetime import datetime
from typing import override
from typing import Self

import attrs
from polars.dataframe import DataFrame

import pandas as pd

import arcticdb as adb


from qoc.api._abc import TradingApi
from qoc.api.typing import Account, Interval, OrderResponseFull, OrderSideLike


@attrs.define
class ApiOffline(TradingApi):
    library: adb.library.Library = attrs.field()

    monitor_account: dict = attrs.field(factory=dict)

    timestamps: list[int] = attrs.field(factory=list)  # 修改为 list[int]
    now_index: int = attrs.field(default=0)
    interval: int = attrs.field(default=60 * 1000000) 

# cfg.symbols, cfg.interval, cfg.start_date, cfg.end_date, cfg.output_dir
    @classmethod
    def create(cls, library: adb.library.Library, symbols: list[str], start_date: str, end_date: str) -> Self:
        from datetime import datetime, timedelta
        import pandas as pd
        
        # 转换日期字符串为datetime对象
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # 初始化时间戳列表
        timestamps = []
        
        # 默认使用1分钟间隔
        interval = "1m"
        
        # 定义预期的间隔（微秒）
        expected_interval_micros = 60 * 1000000  # 1分钟 = 60秒 = 60,000,000微秒

        # 转换日期范围为时间戳（微秒）
        start_timestamp = int(start_dt.timestamp() * 1000000)
        end_timestamp = int(end_dt.timestamp() * 1000000)

        # 从所有交易对中收集时间戳
        all_timestamps = set()

        for symbol in symbols:
            symbol_key = f"{symbol}_klines_{interval}"
            
            if symbol_key in library.list_symbols():
                try:
                    # 读取数据
                    result = library.read(symbol_key).data
                    
                    df_pandas: pd.DataFrame = result 

                    # 检查数据是否为空
                    if not df_pandas.empty:
                        # 获取列和索引信息
                        columns = df_pandas.columns.tolist()
                        
                        # 情况1: "Close Time" 作为普通列
                        if "Close Time" in columns:
                            all_timestamps.update(
                                ts for ts in df_pandas["Close Time"] 
                                if start_timestamp <= ts <= end_timestamp
                            )
                        # 情况2: 索引可能是时间戳（检查索引名称或类型）
                        elif df_pandas.index.name == "Close Time" or isinstance(df_pandas.index, pd.DatetimeIndex):
                            # 如果索引是DatetimeIndex，需要转换为微秒时间戳
                            if isinstance(df_pandas.index, pd.DatetimeIndex):
                                index_timestamps = (df_pandas.index.astype(int) // 1000).tolist()  # 纳秒转微秒
                            else:
                                index_timestamps = df_pandas.index.tolist()
                            
                            all_timestamps.update(
                                ts for ts in index_timestamps
                                if start_timestamp <= ts <= end_timestamp
                            )
                        else:
                            print(f"警告: {symbol_key} 中没有找到 'Close Time' 列或时间索引")
                    
                except Exception as e:
                    print(f"处理交易对 {symbol} 时出错: {e}")

        # 将集合转换为列表并排序
        timestamps = sorted(all_timestamps)
        
        # 检查时间戳间隔
        if len(timestamps) > 1:
            # 计算所有相邻时间戳之间的间隔
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            
            # 统计分析
            min_interval = min(intervals)
            max_interval = max(intervals)
            avg_interval = sum(intervals) / len(intervals)
            
            print(f"时间戳间隔统计:")
            print(f"  最小间隔: {min_interval} 微秒 ({min_interval/1000000} 秒)")
            print(f"  最大间隔: {max_interval} 微秒 ({max_interval/1000000} 秒)")
            print(f"  平均间隔: {avg_interval} 微秒 ({avg_interval/1000000} 秒)")
            
            # 检查是否所有间隔都符合预期
            one_minute_micros = 60 * 1000000  # 1分钟的微秒数
            tolerance = 0  # 允许0误差
            
            all_one_minute = all(abs(interval - one_minute_micros) <= tolerance for interval in intervals)
            
            if all_one_minute:
                print(f"✓ 所有时间戳间隔都在预期的1分钟范围内（允许0秒误差）")
            else:
                # 计算不符合预期的间隔数量
                non_compliant = sum(1 for interval in intervals if abs(interval - one_minute_micros) > tolerance)
                print(f"⚠️ 发现 {non_compliant}/{len(intervals)} 个时间戳间隔不符合预期的1分钟")
                
                # 如果不符合的间隔太多，可以选择修复或给出警告
                if non_compliant > len(intervals) * 0.1:  # 如果超过10%的间隔不符合预期
                    print("⚠️ 警告: 大量时间戳间隔不符合预期，可能会影响回测准确性")
        
        print(f"初始化了 {len(timestamps)} 个时间戳，从 {start_date} 到 {end_date}")
        if timestamps:
            first_dt = datetime.fromtimestamp(timestamps[0]/1000000)
            last_dt = datetime.fromtimestamp(timestamps[-1]/1000000)
            print(f"时间戳范围: {first_dt.isoformat()} 到 {last_dt.isoformat()}")
        
        # 创建实例并设置时间戳
        instance = cls(library=library)


        instance.timestamps = timestamps
        instance.now_index = timestamps[0]
        instance.interval = expected_interval_micros
        print(f"起始时间戳: {instance.now_index} ({datetime.fromtimestamp(instance.now_index/1000000).isoformat()})")   

        # 初始化模拟账户数据
        instance.monitor_account = {
            "makerCommission": 15,
            "takerCommission": 15,
            "buyerCommission": 0,
            "sellerCommission": 0,
            "commissionRates": {
                "maker": "0.00150000",
                "taker": "0.00150000",
                "buyer": "0.00000000",
                "seller": "0.00000000"
            },
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True,
            "brokered": False,
            "requireSelfTradePrevention": False,
            "preventSor": False,
            "updateTime": 0,
            "accountType": "SPOT",
            "balances": [],
            "permissions": [
                "SPOT"
            ],
            "uid": 123456789
        }

        # 添加 USDT 余额
        instance.monitor_account["balances"].append({
            "asset": "USDT",
            "free": "10000.00000000",
            "locked": "0.00000000"
        })

        # 添加所有交易对的基础货币余额
        for symbol in symbols:
            base_asset = symbol.replace("USDT", "")
            instance.monitor_account["balances"].append({
                "asset": base_asset,
                "free": "0.00000000",
                "locked": "0.00000000"
            })

        return instance


    @override
    def account(self, **kwargs) -> Account:
        """返回当前模拟账户信息"""
        # 直接使用存储的账户数据结构
        from qoc.api.typing import Account
        
        # 返回验证过的账户对象
        return Account.model_validate(self.monitor_account)

    @override
    def klines(
        self,
        symbol: str,
        interval: Interval,
        *,
        start_time: datetime | None = None,
        **kwargs,
    ) -> DataFrame:
        import polars as pl
        
        symbol_key = f"{symbol}_klines_{interval}"
        if not self.library.has_symbol(symbol_key):
            raise ValueError(f"Symbol {symbol_key} not found in library")
        
        # 读取数据
        df = self.library.read(symbol_key).data
        
        # 获取当前时间戳的数据行
        klines_now = df[df.index == self.now_index]
        
        # 检查数据是否存在
        if klines_now.empty:
            print(f"警告: 时间戳 {self.now_index} 的数据不存在")
            return pl.DataFrame()
        
        # 列名映射 - 从 ArcticDB 格式转换为 Polars 格式
        column_mapping = {
            'Open Time': 'open_time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Close Time': 'close_time',  # 索引列也需要映射
            'Quote Asset Volume': 'quote_volume',
            'Number of Trades': 'count',
            'Taker Buy Base Volume': 'taker_buy_volume',
            'Taker Buy Quote Volume': 'taker_buy_quote_volume',
            'Ignore': 'ignore'
        }
        
        # 重置索引，使 Close Time 成为普通列
        klines_now = klines_now.reset_index()
        
        # 重命名列
        renamed_cols = {}
        for old_col, new_col in column_mapping.items():
            if old_col in klines_now.columns:
                renamed_cols[old_col] = new_col
        
        klines_now = klines_now.rename(columns=renamed_cols)
        
        # 转换为 Polars DataFrame
        pl_data = pl.from_pandas(klines_now)
        
        # 确保返回的 DataFrame 有正确的列类型
        schema = {
            'open_time': int,
            'open': pl.Float64,
            'high': pl.Float64,
            'low': pl.Float64,
            'close': pl.Float64,
            'volume': pl.Float64,
            'close_time': int,
            'quote_volume': pl.Float64,
            'count': pl.Int64,
            'taker_buy_volume': pl.Float64,
            'taker_buy_quote_volume': pl.Float64,
            'ignore': pl.Int64
        }
        
        # 应用类型转换
        for col, dtype in schema.items():
            if col in pl_data.columns:
                if dtype == pl.Datetime and pl_data[col].dtype != pl.Datetime:
                    # 如果是日期列且需要转换
                    if pl_data[col].dtype in (pl.Int64, pl.Float64):
                        # 假设时间戳是微秒级的
                        pl_data = pl_data.with_columns(
                            pl.col(col).cast(pl.Int64).cast(pl.Datetime(time_unit="us"))
                        )
                else:
                    # 其他列类型转换
                    pl_data = pl_data.with_columns(
                        pl.col(col).cast(dtype)
                    )

        
        return pl_data
        

    @override
    def order_market(
        self,
        symbol: str,
        side: OrderSideLike,
        *,
        quantity: float | None = None,
        quoteOrderQty: float | None = None,
        timestamp: datetime | None = None,
        lock_rate: float = 1.0,  # 保证金倍数参数，默认为1倍
        **kwargs,
    ) -> OrderResponseFull:
        now = self.now_index
        transaction_fee = 0.001  # 交易费用为0.1%
        
        try:
            # 获取当前时间点的价格数据
            symbol_key = f"{symbol}_klines_1m"
            
            if not self.library.has_symbol(symbol_key):
                raise ValueError(f"Symbol {symbol_key} not found in library")
            
            # 读取数据
            df = self.library.read(symbol_key).data
            
            # 获取当前时间戳对应的行
            price_row = df[df.index == now]
            
            if price_row.empty:
                raise ValueError(f"No price data found for timestamp {now}")
            
            # 获取收盘价并确保是float类型
            try:
                price = float(price_row['Close'].iloc[0])
            except (ValueError, TypeError):
                # 如果转换失败，尝试其他列名（大小写不同）
                try:
                    price = float(price_row['close'].iloc[0])
                except (ValueError, TypeError, KeyError):
                    raise ValueError(f"无法获取有效的收盘价格数据 (时间戳: {now})")

            print(f"执行市价单: {symbol}, 方向: {side}, 价格: {price}, 数量: {quantity}")
            
            # 交易的基础资产和报价资产
            base_asset = symbol.replace("USDT", "")
            quote_asset = "USDT"
            
            # 辅助函数：获取资产余额
            def get_asset_balance(asset, balance_type="free"):
                for balance in self.monitor_account["balances"]:
                    if balance["asset"] == asset:
                        return float(balance[balance_type])
                return 0.0
            
            # 辅助函数：更新资产余额
            def update_asset_balance(asset, amount, balance_type="free"):
                for balance in self.monitor_account["balances"]:
                    if balance["asset"] == asset:
                        current_amount = float(balance[balance_type])
                        balance[balance_type] = str(current_amount + amount)
                        return
                
                # 如果资产不存在，添加新资产
                new_balance = {
                    "asset": asset,
                    "free": "0.00000000",
                    "locked": "0.00000000"
                }
                new_balance[balance_type] = str(amount)
                self.monitor_account["balances"].append(new_balance)
            
            # 买入逻辑
            if side == "BUY":
                if quantity is None:
                    raise ValueError("Must provide quantity for BUY order")
                    
                # 计算所需的USDT金额（包含手续费）
                quote_amount = quantity * price
                fee_amount = quote_amount * transaction_fee
                total_cost = quote_amount + fee_amount
                
                # 检查余额是否足够
                usdt_free = get_asset_balance(quote_asset)
                if usdt_free < total_cost:
                    raise ValueError(f"Insufficient {quote_asset} balance: {usdt_free} < {total_cost}")
                
                # 直接扣除USDT余额
                update_asset_balance(quote_asset, -total_cost)
                # 增加基础资产
                update_asset_balance(base_asset, quantity)
                
                print(f"买入 {quantity} {base_asset} 花费 {total_cost} {quote_asset} (含手续费 {fee_amount})")
                    
            # 卖出逻辑
            elif side == "SELL":
                if quantity is None:
                    raise ValueError("Must provide quantity for SELL order")
                    
                # 检查基础资产余额
                base_free = get_asset_balance(base_asset)
                
                # 计算卖出的USDT金额
                quote_amount = quantity * price
                fee_amount = quote_amount * transaction_fee
                net_received = quote_amount - fee_amount
                
                # 情况1: 有足够余额直接卖出
                if base_free >= quantity:
                    update_asset_balance(base_asset, -quantity)
                    update_asset_balance(quote_asset, net_received)
                    print(f"卖出 {quantity} {base_asset} 获得 {net_received} {quote_asset} (扣除手续费 {fee_amount})")
                
                # 情况2: 余额不足，先卖出已有余额，再做空剩余部分
                else:
                    # 分两部分处理
                    # 1. 先卖出已有的余额
                    if base_free > 0:
                        partial_quote_amount = base_free * price
                        partial_fee = partial_quote_amount * transaction_fee
                        partial_net = partial_quote_amount - partial_fee
                        
                        update_asset_balance(base_asset, -base_free)
                        update_asset_balance(quote_asset, partial_net)
                        print(f"卖出可用余额 {base_free} {base_asset} 获得 {partial_net} {quote_asset}")
                    
                    # 2. 剩余部分使用保证金做空
                    remaining_quantity = quantity - base_free
                    short_quote_amount = remaining_quantity * price
                    short_fee = short_quote_amount * transaction_fee
                    
                    # 计算需要锁定的保证金 (使用lock_rate倍保证金)
                    required_margin = short_quote_amount / lock_rate
                    
                    # 检查USDT余额是否足够作为保证金
                    if get_asset_balance(quote_asset) < required_margin:
                        raise ValueError(f"Insufficient {quote_asset} for margin: {get_asset_balance(quote_asset)} < {required_margin}")
                    
                    # 锁定保证金 (从free移到locked)
                    update_asset_balance(quote_asset, -required_margin)
                    update_asset_balance(quote_asset, required_margin, "locked")
                    
                    # 创建做空头寸记录 (可以添加到监控账户的额外字段中)
                    # 这里简化处理，只在base_asset中记录负数余额
                    update_asset_balance(base_asset, -remaining_quantity)
                    update_asset_balance(quote_asset, short_quote_amount - short_fee)
                    
                    print(f"做空 {remaining_quantity} {base_asset} 使用保证金 {required_margin} {quote_asset} (杠杆率: {lock_rate})")
                    print(f"做空获得 {short_quote_amount - short_fee} {quote_asset} (扣除手续费 {short_fee})")
            
            # 创建订单响应对象
            from qoc.api.typing import OrderResponseFull, OrderResponseFill, OrderType, OrderSide
            import datetime
            import uuid
            
            # 创建一个唯一的客户端订单ID
            client_order_id = f"offline_{uuid.uuid4().hex[:16]}"
            
            # 创建fills列表 (成交明细)
            fills = [
                OrderResponseFill(
                    price=price,
                    qty=quantity,
                    commission=fee_amount,  # 手续费
                    commissionasset=quote_asset,  # 手续费资产
                    tradeid=int(now % 1000000)  # 使用时间戳部分作为交易ID
                )
            ]
            
            # 获取当前时间作为交易时间
            current_time = datetime.datetime.fromtimestamp(now / 1000000)
            
            # 转换side字符串为OrderSide枚举
            order_side = OrderSide(side)
            
            # 构建并返回OrderResponseFull对象
            return OrderResponseFull(
                symbol=symbol,
                clientorderid=client_order_id,
                transacttime=current_time,
                price=price,
                origqty=quantity if quantity is not None else 0.0,
                executedqty=quantity if quantity is not None else 0.0,
                origquoteorderqty=quoteOrderQty if quoteOrderQty is not None else 0.0,
                cummulativequoteqty=quote_amount if 'quote_amount' in locals() else 0.0,  # 总价值
                type=OrderType.MARKET,  # 市价单
                side=order_side,
                workingtime=current_time,
                fills=fills,
                # 添加扩展属性 (如果支持的话)
                # lockRate is not a valid parameter, removing it
                # isShort is not a valid parameter, removing it
            )
            
        except Exception as e:
            print(f"执行市价单出错: {e}")
            raise


    def step(self) -> bool:
        if self.now_index + self.interval <= self.timestamps[-1]:
            self.now_index += self.interval
            print(f"时间推进到: {self.now_index} ")
            return False
        else:
            return True
            


    def exchange_info(
        self,
        symbol: str | None = None,
        symbols: list[str] | None = None,
        permissions: list[str] | None = None,
    ) -> None:
        
        # TODO(liblaf): Implement.
        raise NotImplementedError