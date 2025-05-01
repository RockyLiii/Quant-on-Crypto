from abc import ABC, abstractmethod
from typing import Dict
import torch
from timeline import BaseTimeline, StatArbitrageTimeline
from collections import defaultdict, deque

class BaseStrategy(ABC):
    def __init__(self, params: dict = None):
        """
        策略基类初始化
        
        参数:
            params: 策略参数字典
        """
        self.params = params or {}
        self.timeline = None
    
    @abstractmethod
    def get_feature_configs(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """返回特征配置"""
        pass

    def analyze_features(self) -> float:
        """
        分析特征并生成交易信号
        参数:
            features: 特征字典，格式为 {feature_name: {coin: feature_tensor}}
            coin: 当前分析的币种    
        返回:
            float: 交易信号
        """
        pass



class StatArbitrageStrategy(BaseStrategy):

    
    def __init__(self, params: dict = None):
        self.params = params or {}
        super().__init__(params)

        # Get feature parameters from config
        feature_params = self.params.get('strategy', {}).get('feature_params', {})
        
        # Initialize feature calculation parameters
        self.regression_window = feature_params.get('regression_window', 50)
        self.lookback_period = feature_params.get('lookback_period', 288)
        self.correlation_window = feature_params.get('correlation_window', 1728)
        self.std_rate = feature_params.get('std_rate', 0.01)
    
            
        # 从配置中读取窗口参数，注意访问路径
        trading_params = self.params.get('strategy', {}).get('trading_params', {})
        
        self.correlation_thres = trading_params.get('correlation_thres', 0.5)
        self.max_std = trading_params.get('max_std', 0.1)
        self.sigma_threshold = trading_params.get('sigma_threshold', 2.0)
        self.stop_loss = trading_params.get('lc_threshold', -0.03)
        self.max_holding_period = trading_params.get('holding_period', 30)
        self.max_drawback = trading_params.get('max_drawback', 0.05)
        self.trading_fee = trading_params.get('trading_fee', 0.001)
        self.bullet = trading_params.get('bullet', 0.02)

        self.count = 0

        self.signal_history = defaultdict(lambda: deque(maxlen=self.max_holding_period))




    def get_feature_configs(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """实现特征配置"""
        return {
            'coin_features': {
                "price": {"window": self.lookback_period+self.regression_window},
                "beta": {"window": self.regression_window},
                "residual": { "window": self.correlation_window},
                "residual_std": { 
                    "window": 1,
                    "std_rate": self.std_rate
                },
                "position_size": {"window": 1},
                "position_avg_price": {"window": 1}
            },
            'global_features': {
                "correlation_mean": {"window": None},
                "correlation_ma": {"window": 1}
            }
        }
    
    def analyze_features(self) -> Dict[str, float]:
        """
        分析特征并生成所有币种的交易信号
        
        返回:
            Dict[str, float]: {coin: signal} 每个币种的交易信号
            signal: 1.0表示做多一个单位, -1.0表示做空一个单位, 0.0表示不操作
        """
        signals = {}
        for coin in self.timeline.coins:
            signals[coin] = 0.0
            
        try:  
            for coin in self.timeline.coins:
                current_timestamp = self.timeline.current_timestamp
                t_re = current_timestamp - self.timeline.inittime
                

                if coin == 'BTC' or t_re < 2000:
                    continue
    
                beta = self.timeline.coin_features['beta']['data'][coin][-1]
                residual = self.timeline.coin_features['residual']['data'][coin][-1]
                std = self.timeline.coin_features['residual_std']['data'][coin][-1]
                position = self.timeline.coin_features['position_size']['data'][coin][-1]
                position_avg_price = self.timeline.coin_features['position_avg_price']['data'][coin][-1]
                current_price_COIN = self.timeline.coin_features['price']['data'][coin][-1]
                current_price_BTC = self.timeline.coin_features['price']['data']['BTC'][-1]

                available_capital = self.timeline.capital[-1]
                total_value = self.timeline.total_value[-1]

                # print(available_capital, total_value)


                # draw_back = 0.0
                # if position != 0 and position_avg_price != 0:
                #     # 计算当前仓位的浮动盈亏率
                #     if position > 0:  # 多仓
                #         draw_back = (current_price_COIN / position_avg_price - 1)
                #     else:  # 空仓
                #         draw_back = (1 - current_price_COIN / position_avg_price)

                # correlation = correlation_deque[-1]
                # print(f"Coin: {coin}, Beta: {beta}, Residual: {residual}, Std: {std}, Position: {position}")
                
                # 交易信号生成逻辑



                # 新开仓
                if (abs(residual) > self.sigma_threshold * std):
                    self.count += 1
                
                    # and correlation < self.correlation_thres
                    amount = 1/(beta*current_price_COIN+current_price_BTC)
                    if residual > 0:
                        signals[coin] += beta * amount
                        signals['BTC'] -= amount
                        self.signal_history[coin].append((current_timestamp, beta * amount, -amount, current_price_COIN))

                    else:
                        signals[coin] -= beta * amount
                        signals['BTC'] += amount
                        self.signal_history[coin].append((current_timestamp, -beta * amount, amount, current_price_COIN))

                

                    

                #平旧仓
                if self.signal_history[coin]:
                    oldest_signal = self.signal_history[coin][0]
                    oldest_time, signal_coin, signal_BTC, entry_price = oldest_signal
                    
                    # 如果最老的信号超过了最大持有期
                    if current_timestamp - oldest_time >= self.max_holding_period:
                        # 生成反向平仓信号
                        signals[coin] -= signal_coin
                        signals['BTC'] -= signal_BTC
                        self.signal_history[coin].popleft()  # 移除最老的信号
                            
        except (KeyError, IndexError):
            signals[coin] = 0.0
            

    
        return signals
