from typing import Union, Dict
import torch
import time
import os
import numpy as np
from collections import defaultdict, deque
from sklearn.linear_model import LinearRegression

class BaseTimeline:
    def __init__(self, starttime: Union[int, float], endtime: Union[int, float],
             feature_configs: Dict[str, Dict[str, Dict[str, int]]] = None,
             trading_fee: float = 0.001,
             initial_capital: float = 10000):
        """初始化时间轴"""
        self.starttime = starttime
        self.endtime = endtime
        self.current_timestamp = None
        self.trading_fee = trading_fee
        self.initial_capital = initial_capital
        self.inittime = None
        self.coins = []


        
        # Initialize data structures with empty tensors
        self.data = {}  # K线数据 [timestamp, open, high, low, close, ...]
        
        # Initialize other structures
        self.positions = {}          # 持仓信息 [timestamp, quantity, avg_price]
        self.trading_records = {}    # 交易记录 [timestamp, price, quantity, fee]
        self.signal_count = {}       # 信号计数 [coin: count]
        
        # Initialize capital and total value sequences
        self.capital = torch.tensor([[starttime, initial_capital]], dtype=torch.float32)
        self.total_value = torch.tensor([[starttime, initial_capital]], dtype=torch.float32)

        # Initialize feature configs
        self.feature_configs = feature_configs or {
            'coin_features': {},
            'global_features': {}
        }
        
        # Initialize features
        self.coin_features = {}
        self.global_features = {
            feature_name: {'data': deque(maxlen=config['window'])}
            for feature_name, config in self.feature_configs['global_features'].items()
        }
        


        self._capital = initial_capital
        self._total_value = initial_capital
        self._bond = 0 #空仓保证金
        self.tradeamount = 0 #总交易金额

        
         # 存储结构
        self.data = {'total': torch.empty((0, 12), dtype=torch.float32)}  # 原始K线数据
        self.positions = {'total': torch.empty((0, 3), dtype=torch.float32)}  # 持仓信息
        self.trading_records = {'total': torch.empty((0, 4), dtype=torch.float32)}  # 交易记录
        self.signal_count['total'] = 0


 
        # 特征配置和存储
        self.feature_configs = feature_configs or {
            'coin_features': {},
            'global_features': {}
        }
        
        # 特征
        self.coin_features = {}
        self.global_features = {
            feature_name: {'data': deque(maxlen=config['window'])}
            for feature_name, config in self.feature_configs['global_features'].items()
        }


    def feature_calc(self) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """
        基类特征计算框架
            
        返回:
            Dict: {
                'coin_features': {
                    feature_name: {coin: tensor},
                    ...
                },
                'global_features': {
                    feature_name: tensor,
                    ...
                }
            }
        """
        pass
    def _monitor_state(self) -> None:
        """每100个时间戳监控一次状态"""
        current_ts = int(self.get_current_timestamp())  # Convert to int to avoid floating point issues
        print(f"Raw timestamp difference: {current_ts-5790919}")
        print(f"Modulo check: {(current_ts-5790919) % (100)}")
        
        # Check if timestamp is at 100-interval mark
        if current_ts == 0 or (current_ts-5790919) % (100) != 0:
            return
        
        print(f"\n=== State Monitor at timestamp {current_ts} ===")
        print("Capital:", self.capital[-1].tolist())  # [timestamp, available_capital]
        print("Total Value:", self.total_value[-1].tolist())  # [timestamp, total_asset_value]
        
        print("\nTotal Records:")
        print("Position:", self.positions['total'][-1].tolist())  # [timestamp, quantity, avg_price]
        print("Trading Records:", len(self.trading_records['total']))  # Count of trades
        
        print("\nFeature Samples:")
        for feature_name, feature_data in self.coin_features.items():
            if 'BTC' in feature_data['data']:
                print(f"{feature_name}:", list(feature_data['data']['BTC'])[-1])
                
        for feature_name, feature_data in self.global_features.items():
            if len(feature_data['data']) > 0:
                print(f"{feature_name}:", feature_data['data'][-1])
        print("="*50)


    def _initialize_for_coin(self, coinname: str) -> None:
        """为新币种初始化数据结构"""
        if coinname not in self.coins:
            self.coins.append(coinname)
            
            # Initialize with empty tensors
            self.data[coinname] = torch.empty((0, 12), dtype=torch.float32)
            self.positions[coinname] = torch.empty((0, 3), dtype=torch.float32)
            self.trading_records[coinname] = torch.empty((0, 4), dtype=torch.float32)

            self.signal_count[coinname] = 0

            
            # Initialize features
            for feature_name, config in self.feature_configs['coin_features'].items():
                if feature_name not in self.coin_features:
                    self.coin_features[feature_name] = {
                        'data': defaultdict(lambda: deque(maxlen=config['window']))
                    }
                self.coin_features[feature_name]['data'][coinname] = deque(maxlen=config['window'])
    
    
    def time_pass(self, new_data: Dict[str, torch.Tensor]) -> None:

        """时间推进，更新数据并计算特征"""
        
        self.inittime = self.get_current_timestamp() if self.inittime is None or self.inittime==0 else self.inittime


        for coinname, new_row in new_data.items():
            self.data[coinname] = new_row

        for coinname in new_data:
            if coinname not in self.coins:
                self._initialize_for_coin(coinname)
                self.current_timestamp = new_data[coinname][0, 0].item()

        self.current_timestamp = self.get_current_timestamp()

        self.feature_calc()
        
        # Update recor ds
        totals = (0.0, 0.0)  # quantity, value
        for coin in self.coins:
            coin_results = self._update_records(coin)
            totals = tuple(t + r for t, r in zip(totals, coin_results))

        # Update total records and capital
        self._update_total_records(totals)
        self._update_capital_and_value(totals[1])
        

    def get_current_timestamp(self) -> int:
        """获取当前时间戳"""
        for coin in self.coins:
            if coin in self.data and self.data[coin].size(0) > 0:
                return int(self.data[coin][-1, 0].item())
        return 0

    
    def _update_records(self, coin: str) -> tuple[float, float, float, float]:
        """更新各类记录并返回统计值
        
        Returns:
            tuple: (quantity, value)
        """
        if self.data[coin].size(0) == 0:
            return 0.0, 0.0
            
        current_price = self.data[coin][-1, 4].item()
        
        # Calculate profits
        qty = self.positions[coin][-1, 1].item() if self.positions[coin].size(0) > 0 else 0.0
        avg_price = self.positions[coin][-1, 2].item() if self.positions[coin].size(0) > 0 else 0.0
        
        # Update total profit
        
        # Calculate position value
        value = qty * current_price    #❓关于做空
        
        return qty, value


    def _update_total_records(self, totals: tuple[float, float, float, float]) -> None:
        """更新汇总记录
        
        Args:
            totals: (total_quantity, total_value)
        """
        qty, value = totals
        
        
        # Update total position
        total_position = torch.tensor([[
            self.current_timestamp, qty, value
        ]], dtype=torch.float32)    #✅

        self.positions['total'] = torch.cat((self.positions['total'], total_position), dim=0)    #✅
        
       

    def _update_capital_and_value(self, total_position_value: float) -> None:
        """更新可用资金和总资产价值
        
        Args:
            total_position_value: 所有持仓的当前市值
        """

        capital_record = torch.tensor([[
            self.current_timestamp,
            self._capital
        ]], dtype=torch.float32)    #❓❓❓

        self._total_value = self._capital + total_position_value + 2*self._bond    #❓❓❓

        value_record = torch.tensor([[
            self.current_timestamp,
            self._total_value
        ]], dtype=torch.float32)    #❓❓❓

        
        self.capital = torch.cat((self.capital, capital_record), dim=0)    #✅
        self.total_value = torch.cat((self.total_value, value_record), dim=0)    #✅
        
    def trade(self, coinname: str, quantity: float) -> None:
        """执行交易操作，处理开仓、加仓、减仓及手续费"""
        if coinname not in self.coins:
            return
        
        if quantity != 0:
            self.signal_count[coinname] += 1
            self.signal_count['total'] += 1
            
        
        # 获取当前持仓状态
        if self.positions[coinname].size(0) == 0:
            current_qty = 0.0
            current_avg = 0.0
        else:
            current_qty = self.positions[coinname][-1, 1].item()  # quantity is in column 1
            current_avg = self.positions[coinname][-1, 2].item()  # avg_price is in column 2
        
        # 计算交易金额和手续费
        trade_price = self.data[coinname][-1, 4].item()  # Use closing price
        transaction_amount = trade_price * abs(quantity)
        fee_cost = transaction_amount * self.trading_fee
        
        self._capital -= fee_cost    #✅
        self.tradeamount += transaction_amount
        
        # Record trade with fee
        record = torch.tensor([[
            self.current_timestamp,
            trade_price,
            quantity,
            fee_cost
        ]], dtype=torch.float32)
        self.trading_records[coinname] = torch.cat((self.trading_records[coinname], record), dim=0)

        aver_price = current_avg
        
        # 新开仓
        if current_qty == 0:
            aver_price = trade_price
            self._capital -= transaction_amount   #✅
  
            if quantity < 0:
                self._bond += transaction_amount    #✅空仓考虑1x保证金
            

        # 加仓
        elif current_qty * quantity > 0:
            aver_price = (current_avg * abs(current_qty) + trade_price * abs(quantity)) / (abs(current_qty) + abs(quantity))
            self._capital -= transaction_amount   #✅
            if quantity < 0:
                self._bond += transaction_amount    #✅空仓考虑1x保证金

        # 减仓
        elif current_qty * quantity < 0:
            remaining_qty = abs(quantity) - abs(current_qty)
            
            #多空反转
            if remaining_qty > 0:
                if quantity < 0:
                    aver_price = trade_price
                    self._capital += trade_price * abs(current_qty)
                    self._capital -= trade_price * remaining_qty
                    self._bond += trade_price * abs(remaining_qty) #空仓保证金
                    
                if quantity > 0:
                    aver_price = trade_price
                    self._capital -= trade_price * abs(remaining_qty)    #✅

                    self._capital += (2*current_avg-trade_price) * abs(current_qty)    #✅
                    self._bond -= current_avg * abs(current_qty) #空仓保证金
            
            # 多空不反转
            else:
                if quantity < 0:
                    aver_price =  current_avg
                    self._capital += trade_price * abs(quantity)    #✅

                if quantity > 0:
                    aver_price = current_avg
                    self._capital -= trade_price * abs(quantity)    #✅
                    self._capital += 2*current_avg * abs(quantity)
                    self._bond -= current_avg * abs(quantity) #空仓保证金
            
        position_record = torch.tensor([[
            self.current_timestamp,
            current_qty+quantity,
            aver_price
        ]], dtype=torch.float32)
        self.positions[coinname] = torch.cat((self.positions[coinname], position_record), dim=0)
            



class StatArbitrageTimeline(BaseTimeline):
    def __init__(self, starttime: Union[int, float], endtime: Union[int, float], 
                feature_configs: Dict[str, Dict[str, Dict[str, int]]], trading_fee: float = 0.001,
                 initial_capital: float = 10000):
        super().__init__(starttime, endtime, feature_configs, trading_fee, initial_capital)

        # Initialize feature calculation parameters
        self.regression_window = feature_configs['coin_features']['beta']['window']
        self.lookback_period = feature_configs['global_features']['correlation_ma']['window']
        self.correlation_window = feature_configs['global_features']['correlation_mean']['window']
        self.std_rate = feature_configs['coin_features']['residual_std']['std_rate']

        # Initialize feature records dictionary with defaultdict
        self.feature_records = {
            'price': defaultdict(lambda: torch.empty((0, 2), dtype=torch.float32)),
            'beta': defaultdict(lambda: torch.empty((0, 2), dtype=torch.float32)),
            'residual': defaultdict(lambda: torch.empty((0, 2), dtype=torch.float32)),
            'std': defaultdict(lambda: torch.empty((0, 2), dtype=torch.float32))
        }
    
    def feature_calc(self) -> None:
        """计算所有特征"""

        feature_times = {}

        # Price特征计算时间
        start = time.time()
        self._calc_price_feature()
        feature_times['price'] = time.time() - start
        
        # Beta特征计算时间
        start = time.time()
        self._calc_beta_feature()
        feature_times['beta'] = time.time() - start
        
        # 残差特征计算时间
        start = time.time()
        self._calc_residual_feature()
        feature_times['residual'] = time.time() - start
        
        # 残差标准差特征计算时间
        start = time.time()
        self._calc_residual_std_feature()
        feature_times['residual_std'] = time.time() - start
        
        # 仓位特征计算时间
        self._calc_position_feature() 
        self._calc_position_avg_feature()
        feature_times['position'] = time.time() - start

        # 相关性均值特征计算时间
        start = time.time()
        self._calc_correlation_mean_feature()
        self._calc_correlation_ma_feature()
        feature_times['correlation'] = time.time() - start


        # 如果数据量超过1000条，记录详细信息
        if hasattr(self, '_feature_calc_count'):
            self._feature_calc_count += 1
        else:
            self._feature_calc_count = 1
            
        if self._feature_calc_count % 1000 == 0:
            total_time = sum(feature_times.values())
            print("\n=== 特征计算详细耗时 ===")
            print(f"计数: {self._feature_calc_count}")
            print(f"数据大小: {len(self.data[list(self.data.keys())[0]])}")
            for feature, t in feature_times.items():
                print(f"{feature}: {t:.4f}s ({t/total_time*100:.1f}%)")
            print("======================\n")
            
            # 记录数据大小与计算时间的关系
            if not hasattr(self, '_performance_log'):
                self._performance_log = []
            self._performance_log.append({
                'count': self._feature_calc_count,
                'data_size': len(self.data[list(self.data.keys())[0]]),
                'times': feature_times.copy()
            })

        # Calculate price and market features

        
        # Calculate global features
    def _record_feature(self, feature_name: str, coin: str, value: float) -> None:
        """Record feature value with timestamp"""
        record = torch.tensor([[
            self.current_timestamp,
            value
        ]], dtype=torch.float32)
        self.feature_records[feature_name][coin] = torch.cat(
            (self.feature_records[feature_name][coin], record), dim=0
        )

    def _calc_price_feature(self) -> None:
        """计算价格特征"""
        for coin in self.coins:
            current_price = self.data[coin][-1, 4]
            self.coin_features['price']['data'][coin].append(current_price)
            self._record_feature('price', coin, current_price)

#            print(self.coin_features['price']['data'][coin])

        
    def _calc_beta_feature(self) -> None:
        """使用滑动窗口增量计算优化的beta计算方法"""
        
        # 初始化存储结构，用于保存每个币种的求和结果
        if not hasattr(self, '_beta_sums'):
            self._beta_sums = {}
        
        for coin in self.coins:
            if coin == 'BTC' or coin not in self.data or 'BTC' not in self.data:
                continue
            
            # 获取价格数据
            price_deque = self.coin_features['price']['data'][coin]
            btc_price_deque = self.coin_features['price']['data']['BTC']
            
            # 检查数据是否足够
            if len(price_deque) < self.regression_window+self.lookback_period or len(btc_price_deque) < self.regression_window+self.lookback_period:
                continue
            
            x = np.array(list(price_deque)[0:self.regression_window]).reshape(-1, 1)
            y = np.array(list(btc_price_deque)[0:self.regression_window]).reshape(-1, 1)
            
            # 检查数据有效性
            if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                continue
                
            # 使用无截距项的线性回归
            reg = LinearRegression(fit_intercept=False).fit(x, y)
            beta = reg.coef_[0][0]  # 获取beta值
            
            # 存储beta值
            self.coin_features['beta']['data'][coin].append(beta)
            self._record_feature('beta', coin, beta)

    def _calc_residual_feature(self) -> None:
        for coin in self.coins:
            if coin == 'BTC':
                continue
                
            # Check if we have beta value
            beta_deque = self.coin_features['beta']['data'][coin]
            if len(beta_deque) == 0:
                continue
                
            # Get current prices
            current_coin_price = self.data[coin][-1, 4].item()
            current_btc_price = self.data['BTC'][-1, 4].item()
            
            # Get most recent beta
            beta = beta_deque[-1]
            
            # Calculate residual
            pred = beta * current_coin_price
            residual = current_btc_price - pred
            
            # Store residual
            self.coin_features['residual']['data'][coin].append(residual)
            self._record_feature('residual', coin, residual)

    
    def _calc_residual_std_feature(self) -> None:
        for coin in self.coins:
            if coin == 'BTC':
                continue
                
            # Check if we have residual value
            residual_deque = self.coin_features['residual']['data'][coin]
            if len(residual_deque) == 0:
                continue
                
            # Get latest residual
            residual = residual_deque[-1]
            
            # Get previous std or initialize
            std_deque = self.coin_features['residual_std']['data'][coin]
            if len(std_deque) == 0:
                std_temp = 0.02  # Initial std value
            else:
                std_temp = std_deque[-1]
            
            # Update std using EWMA
            new_std = np.sqrt(std_temp**2 * (0.99) + residual**2 * 0.01)
            
            # Store new std
            self.coin_features['residual_std']['data'][coin].append(new_std)
            self._record_feature('std', coin, new_std)
        
    def _calc_position_feature(self) -> None:
        """计算持仓量特征"""
        for coin in self.coins:
            if self.positions[coin].size(0) == 0:
                # No positions yet, use default values
                position_size = 0.0
            else:
                # Get latest position
                position_size = self.positions[coin][-1, 1].item() 
            
            # Store in features
            if coin not in self.coin_features['position_size']['data']:
                self.coin_features['position_size']['data'][coin] = deque(maxlen=1)
            self.coin_features['position_size']['data'][coin].append(position_size)

    def _calc_position_avg_feature(self) -> None:
        """计算持仓均价特征"""
        for coin in self.coins:
            # Check if we have any positions
            if self.positions[coin].size(0) == 0:
                # No positions yet, use default values
                avg_price = 0.0
            else:
                # Get latest average price
                avg_price = self.positions[coin][-1, 2].item()  # Using -1 to get last row, 2 for avg_price
            
            # Store in features
            if coin not in self.coin_features['position_avg_price']['data']:
                self.coin_features['position_avg_price']['data'][coin] = deque(maxlen=1)
            self.coin_features['position_avg_price']['data'][coin].append(avg_price)

    def _calc_correlation_mean_feature(self) -> None:
        # 收集满足条件的币种的residual数据
        # valid_residuals = {}
        # for coin in self.coins:
        #     if coin == 'BTC':
        #         continue
                
        #     residual_deque = self.coin_features['residual']['data'][coin]
        #     if len(residual_deque) < residual_deque.maxlen:
        #         return  # 如果有任何币种的residual未满，直接返回
                
        #     valid_residuals[coin] = np.array(list(residual_deque))
        
        # # 至少需要两个币种才能计算相关性
        # if len(valid_residuals) < 2:
        #     return
            
        # # 构建残差矩阵进行相关性计算
        # coins = list(valid_residuals.keys())
        # n_coins = len(coins)
        # correlations = []
        
        # # 计算两两相关性
        # for i in range(n_coins):
        #     for j in range(i+1, n_coins):
        #         corr = np.corrcoef(valid_residuals[coins[i]], 
        #                         valid_residuals[coins[j]])[0, 1]
        #         correlations.append(corr)
        
        # # 计算平均相关性
        # corr_mean = np.mean(correlations)
        
        # # 存储结果
        # self.global_features['correlation_mean']['data'].append(corr_mean)
        pass
        
    def _calc_correlation_ma_feature(self) -> None:
        pass

