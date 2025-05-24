from typing import Union, Dict
import torch
import time
import os
import numpy as np
from collections import defaultdict, deque
from sklearn.linear_model import LinearRegression
import logging
logger = logging.getLogger(__name__)

class BaseTimeline:
    def __init__(self, starttime: Union[int, float], endtime: Union[int, float],
                feature_configs: Dict[str, Dict[str, Dict[str, int]]] = None,
                trading_fee: float = 0.001,
                initial_capital: float = 10000):
        """初始化时间轴"""
        
        # 1. 基础时间参数
        self.starttime = starttime          # 回测开始时间
        self.endtime = endtime             # 回测结束时间
        self.current_timestamp = None      # 当前时间戳
        self.inittime = None               # 初始化时间
        
        # 2. 账户参数
        self.initial_capital = initial_capital  # 初始资金
        self.trading_fee = trading_fee         # 交易手续费率
        self._capital = initial_capital        # 当前可用资金
        self._total_value = initial_capital    # 当前总资产
        self._bond = 0                         # 空仓保证金
        self.tradeamount = 0                   # 总交易金额
        
        # 3. 交易品种
        self.coins = []                        # 交易币种列表
        
        # 4. 市场数据存储
        self.data = {
            'total': torch.empty((0, 12), dtype=torch.float32)  # K线数据 [timestamp, open, high, low, close, ...]
        }
        
        # 5. 仓位和交易记录
        self.positions = {
            'total': torch.empty((0, 3), dtype=torch.float32)   # 持仓信息 [timestamp, quantity, avg_price]
        }
        self.trading_records = {
            'total': torch.empty((0, 4), dtype=torch.float32)   # 交易记录 [timestamp, price, quantity, fee]
        }
        
        # 6. 资金记录
        self.capital = torch.tensor([[starttime, initial_capital]], dtype=torch.float32)      # 资金曲线
        self.total_value = torch.tensor([[starttime, initial_capital]], dtype=torch.float32)  # 总资产曲线
        
        # 7. 交易统计
        self.signal_count = {'total': 0}       # 信号计数 [coin: count]
        
        # 8. 特征配置和存储
        self.feature_configs = feature_configs or {
            'coin_features': {},               # 币种特征配置
            'global_features': {}              # 全局特征配置
        }
        
        # 9. 特征数据存储
        self.coin_features = {}                # 币种特征数据
        self.global_features = {               # 全局特征数据
            feature_name: {'data': deque(maxlen=config['window'])}
            for feature_name, config in self.feature_configs['global_features'].items()
        }

        self.cor_r = 1
        self.cor_p = 1


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
        ]], dtype=torch.float32) 

        self.positions['total'] = torch.cat((self.positions['total'], total_position), dim=0)
        
       

    def _update_capital_and_value(self, total_position_value: float) -> None:
        """更新可用资金和总资产价值
        
        Args:
            total_position_value: 所有持仓的当前市值
        """

        capital_record = torch.tensor([[
            self.current_timestamp,
            self._capital
        ]], dtype=torch.float32)

        self._total_value = self._capital + total_position_value + 2*self._bond

        value_record = torch.tensor([[
            self.current_timestamp,
            self._total_value
        ]], dtype=torch.float32)

        
        self.capital = torch.cat((self.capital, capital_record), dim=0)
        self.total_value = torch.cat((self.total_value, value_record), dim=0) 
        
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
        
        self._capital -= fee_cost    #
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
            self._capital -= transaction_amount   #
  
            if quantity < 0:
                self._bond += transaction_amount    #空仓考虑1x保证金
            

        # 加仓
        elif current_qty * quantity > 0:
            aver_price = (current_avg * abs(current_qty) + trade_price * abs(quantity)) / (abs(current_qty) + abs(quantity))
            self._capital -= transaction_amount   #
            if quantity < 0:
                self._bond += transaction_amount    #空仓考虑1x保证金

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
                    self._capital -= trade_price * abs(remaining_qty)    #

                    self._capital += (2*current_avg-trade_price) * abs(current_qty)    #
                    self._bond -= current_avg * abs(current_qty) #空仓保证金
            
            # 多空不反转
            else:
                if quantity < 0:
                    aver_price =  current_avg
                    self._capital += trade_price * abs(quantity)    #

                if quantity > 0:
                    aver_price = current_avg
                    self._capital -= trade_price * abs(quantity)    #
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
        self.window_all = feature_configs['coin_features']['price']['window']
        self.window_regression = feature_configs['coin_features']['price']['window_calc']
        self.residual_window = feature_configs['coin_features']['residual']['window']
        self.std_rate = feature_configs['coin_features']['residual_std']['std_rate']

        # Add Bollinger Band parameters
        corr_config = feature_configs['global_features']['correlation']

        self.corr_window = corr_config['window']
        self.corr_threshold_u = corr_config['threshold_u']
        self.freeze_history = []
        self.correlation_residual = []
        self.correlation_price = []
        self.correlation_residual_deque = deque(maxlen=1440)

        
        # Add freeze state
        self.freeze = False
        self.freeze_countdown = 0   
        self.freeze_days = 0 
        self.all_days = 0
        self.freeze_rate = 0
        self.btc_prices = deque(maxlen=self.corr_window)
        self.freeze_period = corr_config['freeze_period']
        


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

   
        # Correlation特征计算时间
        start = time.time()
        self._calc_correlation_residual()
        feature_times['correlation_r'] = time.time() - start   
        # Correlation特征计算时间
        start = time.time()
        self._calc_correlation_price()
        feature_times['correlation_p'] = time.time() - start     
     
        # 仓位特征计算时间
        self._calc_position_feature() 
        feature_times['position'] = time.time() - start


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


    def _calc_correlation_residual(self) -> None:
        """Calculate correlations between residuals"""
        
        # Skip if insufficient coins
        if len(self.coins) < 2:
            return
        weighted_avg = 1
        # Initialize sliding window sums if not exists
        if not hasattr(self, '_corr_sums'):
            self._corr_sums = defaultdict(lambda: {
                'sum_x': 0.0,
                'sum_y': 0.0,
                'sum_xy': 0.0,
                'sum_x2': 0.0,
                'sum_y2': 0.0,
                'window_values_x': deque(maxlen=self.residual_window),
                'window_values_y': deque(maxlen=self.residual_window)
            })
        
        valid_pairs = []
        
        # Calculate correlations for each pair
        for i, coin1 in enumerate(self.coins):
            if coin1 == 'BTC':
                continue
                
            residuals1 = self.coin_features['residual']['data'][coin1]
            if len(residuals1) < self.residual_window:
                continue
                
            for coin2 in self.coins[i+1:]:
                if coin2 == 'BTC':
                    continue
                    
                residuals2 = self.coin_features['residual']['data'][coin2]
                if len(residuals2) < self.residual_window:
                    continue
                    
                pair_key = f"{coin1}_{coin2}"
                
                # Get current residual values
                new_residual1 = residuals1[-1]
                new_residual2 = residuals2[-1]
                
                sums = self._corr_sums[pair_key]
                
                # Update sliding window
                if len(sums['window_values_x']) == self.residual_window:
                    # Remove oldest values from sums
                    old_x = sums['window_values_x'][0]
                    old_y = sums['window_values_y'][0]
                    sums['sum_x'] -= old_x
                    sums['sum_y'] -= old_y
                    sums['sum_xy'] -= old_x * old_y
                    sums['sum_x2'] -= old_x * old_x
                    sums['sum_y2'] -= old_y * old_y

                # Add new values
                sums['window_values_x'].append(new_residual1)
                sums['window_values_y'].append(new_residual2)
                sums['sum_x'] += new_residual1
                sums['sum_y'] += new_residual2
                sums['sum_xy'] += new_residual1 * new_residual2
                sums['sum_x2'] += new_residual1 * new_residual1
                sums['sum_y2'] += new_residual2 * new_residual2
                
                # Calculate correlation if we have enough data
                n = len(sums['window_values_x'])
                if n >= self.residual_window:
                    try:
                        numerator = n * sums['sum_xy'] - sums['sum_x'] * sums['sum_y']
                        denominator = np.sqrt((n * sums['sum_x2'] - sums['sum_x']**2) * 
                                            (n * sums['sum_y2'] - sums['sum_y']**2))
                        
                        
                        if denominator != 0:
                            corr = numerator / denominator
                            # # Calculate volatility for weighting using residual std
                            # std1 = self.coin_features['residual_std']['data'][coin1][-1]
                            # std2 = self.coin_features['residual_std']['data'][coin2][-1]
                            # weight = 1 / (std1 * std2 + 1e-8)                            

                            weight = 1
                            valid_pairs.append((corr, weight))
                    except:
                        continue

        
        # Calculate weighted average correlation
        if valid_pairs:
            correlations, weights = zip(*valid_pairs)
            weights = np.array(weights)
            weighted_avg = np.average(correlations, weights=weights)
            self.cor_r = self.std_rate * weighted_avg + (1-self.std_rate)*self.cor_r

            self.correlation_residual.append((self.current_timestamp, self.cor_r))
            self.correlation_residual_deque.append(self.cor_r)
            # Update freeze state based on price correlation
            self.update_freeze_state(self.cor_r)
        


    def _calc_correlation_price(self) -> None:
        """Calculate correlations between price returns using sliding window"""
        
        # Skip if insufficient coins
        if len(self.coins) < 2:
            return
        
        # Initialize sliding window sums if not exists
        if not hasattr(self, '_price_corr_sums'):
            self._price_corr_sums = defaultdict(lambda: {
                'sum_x': 0.0,
                'sum_y': 0.0,
                'sum_xy': 0.0,
                'sum_x2': 0.0,
                'sum_y2': 0.0,
                'window_values_x': deque(maxlen=self.window_all),
                'window_values_y': deque(maxlen=self.window_all)
            })
        
        valid_pairs = []
        weighted_avg = 1
        
        # Calculate correlations for each pair
        for i, coin1 in enumerate(self.coins):
            if coin1 == 'BTC':
                continue
                
            prices1 = self.coin_features['price']['data'][coin1]
            if len(prices1) < 2:  # Need at least 2 prices for returns
                continue
                
            # Calculate return
            new_return1 = (prices1[-1] - prices1[-2]) / prices1[-2]
                
            for coin2 in self.coins[i+1:]:
                if coin2 == 'BTC':
                    continue
                    
                prices2 = self.coin_features['price']['data'][coin2]
                if len(prices2) < 2:
                    continue
                    
                # Calculate return
                new_return2 = (prices2[-1] - prices2[-2]) / prices2[-2]
                
                pair_key = f"{coin1}_{coin2}"
                sums = self._price_corr_sums[pair_key]
                
                # Update sliding window
                if len(sums['window_values_x']) == self.window_all:
                    # Remove oldest values from sums
                    old_x = sums['window_values_x'][0]
                    old_y = sums['window_values_y'][0]
                    sums['sum_x'] -= old_x
                    sums['sum_y'] -= old_y
                    sums['sum_xy'] -= old_x * old_y
                    sums['sum_x2'] -= old_x * old_x
                    sums['sum_y2'] -= old_y * old_y
                
                # Add new values
                sums['window_values_x'].append(new_return1)
                sums['window_values_y'].append(new_return2)
                sums['sum_x'] += new_return1
                sums['sum_y'] += new_return2
                sums['sum_xy'] += new_return1 * new_return2
                sums['sum_x2'] += new_return1 * new_return1
                sums['sum_y2'] += new_return2 * new_return2
                
                # Calculate correlation if we have enough data
                n = len(sums['window_values_x'])
                if n >= self.window_all:
                    try:
                        numerator = n * sums['sum_xy'] - sums['sum_x'] * sums['sum_y']
                        denominator = np.sqrt((n * sums['sum_x2'] - sums['sum_x']**2) * 
                                            (n * sums['sum_y2'] - sums['sum_y']**2))
                        
                        if denominator != 0:
                            corr = numerator / denominator
                            # Weight by inverse volatility
                            # weight = 1 / (np.std(list(sums['window_values_x'])) * 
                            #             np.std(list(sums['window_values_y'])) + 1e-8)                           
                            weight = 1 
                            valid_pairs.append((corr, weight))
                    except:
                        continue
        
        # Calculate weighted average correlation
       
        if valid_pairs:
            correlations, weights = zip(*valid_pairs)
            weights = np.array(weights)
            weighted_avg = np.average(correlations, weights=weights)
            self.cor_p = self.std_rate*weighted_avg + (1-self.std_rate)*self.cor_p
            self.correlation_price.append((self.current_timestamp, self.cor_p))
            
            # # Update freeze state based on price correlation
            # self.update_freeze_state(self.cor_p)


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
            if len(price_deque) < self.window_all or len(btc_price_deque) < self.window_all:
                continue
            
            x = np.array(list(price_deque)[0:self.window_regression]).reshape(-1, 1)
            y = np.array(list(btc_price_deque)[0:self.window_regression]).reshape(-1, 1)
            
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
            new_std = np.sqrt(std_temp**2 * (1-self.std_rate) + residual**2 * self.std_rate)
            
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
        


    def update_freeze_state(self, correlation: float) -> None:
        """Update market freeze state based on correlation value"""
        # Check for breakout and update freeze state
        if correlation > self.corr_threshold_u:
            self.freeze = True
            self.freeze_countdown = self.freeze_period
        elif self.freeze_countdown > 0:
            self.freeze_countdown -= 1
            if self.freeze_countdown == 0:
                self.freeze = False
                
        # Update freeze statistics
        self.all_days += 1
        if self.freeze:
            self.freeze_days += 1
            if self.all_days > 0:
                self.freeze_rate = self.freeze_days / self.all_days
                
        # Record freeze state
        self.freeze_history.append((self.current_timestamp, self.freeze))