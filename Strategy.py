from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from collections import defaultdict, deque
import logging
import numpy as np

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for trading strategies"""
    
    def __init__(self, params: dict = None):
        """Initialize strategy base class"""
        self.params = params or {}
        self.timeline = None

        # State tracking
        self.signal_history = defaultdict(lambda: deque(maxlen=self.max_holding_period))
        
    @abstractmethod
    def get_feature_configs(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Return feature configuration dictionary"""
        pass

    def analyze_features(self) -> Dict[str, float]:
        """
        Main feature analysis framework that processes positions and generates signals
        
        Returns:
            Dict[str, float]: Dictionary of trading signals for each coin
        """
        signals = {coin: 0.0 for coin in self.timeline.coins}
        
        try:
            for coin in self.timeline.coins:
                try:
                    # Skip coins that don't meet analysis criteria
                    if not self._should_analyze_coin(coin):
                        continue
                    
                    # Get current market state
                    market_state = self._get_market_state(coin)
                    
                    # Handle existing positions
                    revenue = self._close_positions(coin, market_state, signals)
                    
                    # Open new positions if conditions are met
                    if self._can_open_position(market_state):
                        self._generate_signals(coin, market_state, signals)
                    
                    # Update performance metrics
                    self._update_metrics(coin, market_state, revenue)
                    
                except Exception as e:
                    logger.error(f"Error processing coin {coin}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in feature analysis: {e}")
            
        return signals

    @abstractmethod
    def _should_analyze_coin(self, coin: str) -> bool:
        """Check if coin should be analyzed"""
        pass

    @abstractmethod
    def _get_market_state(self, coin: str) -> dict:
        """Get current market state"""
        pass

    @abstractmethod
    def _can_open_position(self, state: dict) -> bool:
        """Check if new position can be opened"""
        pass

    @abstractmethod
    def _generate_signals(self, coin: str, state: dict, signals: dict) -> None:
        """Generate trading signals"""
        pass

    def _close_positions(self, coin: str, state: dict, signals: dict) -> float:
        """Handle position closing"""
        if not self.signal_history[coin]:
            return 0.0
        
            
        if self.timeline.freeze:
            return self._force_close_positions(coin, state, signals)
        
        return self._normal_close_positions(coin, state, signals)

    def _force_close_positions(self, coin: str, state: dict, signals: dict) -> float:
        """
        Force close all positions and calculate total revenue
        
        Args:
            coin: Trading coin symbol
            signals: Signal dictionary to update
            
        Returns:
            float: Total revenue from closed positions
        """
        total_revenue = 0.0
        
        while self.signal_history[coin]:
            oldest_signal = self.signal_history[coin][0]
            
            # Calculate revenue before closing
            try:
                # Get market state for revenue calculation
                state = self._get_market_state(coin)
                position_revenue = self._calculate_revenue(oldest_signal, state)
                total_revenue += position_revenue
                
                # Record revenue rate
                self._record_revenue_rate(position_revenue, oldest_signal)
                
            except Exception as e:
                logger.error(f"Error calculating revenue for forced close of {coin}: {e}")
                position_revenue = 0.0
                
            # Close position
            _, signal_coin, signal_BTC, _, _ = oldest_signal
            signals[coin] -= signal_coin
            signals['BTC'] -= signal_BTC
            self.signal_history[coin].popleft()
            
        return total_revenue

    def _normal_close_positions(self, coin: str, state: dict, signals: dict) -> float:
        """Handle normal position closing"""
        revenue = 0.0
        try:
            if not self.signal_history[coin]:
                return 0.0
                
            # oldest_signal = self.signal_history[coin][0]
            
            # ###### REVENUE HERE?

            
            # if self._should_close_position(oldest_signal, revenue, state):
            #     revenue += self._calculate_revenue(oldest_signal, state)
            #     self._close_position(coin, oldest_signal, signals)
            #     self._record_revenue_rate(revenue, oldest_signal)
            
            positions_to_close = self._should_close_position_1(self.signal_history[coin], state)

            if positions_to_close:
                for idx in sorted(positions_to_close, reverse=True):
                    signal_1 = self.signal_history[coin][idx]
                    revenue_i = self._calculate_revenue(signal_1, state)
                    revenue += revenue_i
                    self._close_position_1(coin, idx, signals)
                    self._record_revenue_rate(revenue_i, signal_1)
            
                
            return revenue
            
        except IndexError:
            # Silently handle empty deque without logging error
            return 0.0
        except Exception as e:
            # Log other unexpected errors
            logger.error(f"Unexpected error closing positions for {coin}: {e}")
            return 0.0

    @abstractmethod
    def _calculate_revenue(self, signal_data: tuple, state: dict) -> float:
        """Calculate position revenue"""
        pass

    @abstractmethod
    def _should_close_position(self, oldest_signal, revenue,  state: dict) -> bool:
        """Check if position should be closed"""
        pass

    def _close_position(self, coin: str, signal_data: tuple, signals: dict) -> None:
        """Close position and update signals"""
        _, signal_coin, signal_BTC, _, _ = signal_data
        signals[coin] -= signal_coin
        signals['BTC'] -= signal_BTC
        self.signal_history[coin].popleft()    
    
    def _close_position_1(self, coin: str, i: int, signals: dict) -> None:
        """Close position and update signals"""
        signal_history = list(self.signal_history[coin])
        signal = signal_history[i]
        _, signal_coin, signal_BTC, _, _ = signal
        
        # Update signals
        signals[coin] -= signal_coin
        signals['BTC'] -= signal_BTC
        
        # Clear and rebuild deque without the closed position
        self.signal_history[coin].clear()
        for j, sig in enumerate(signal_history):
            if j != i:
                self.signal_history[coin].append(sig)

    def _record_revenue_rate(self, revenue: float, signal_data: tuple) -> None:
        """Record revenue rate"""
        _, signal_coin, signal_BTC, coin_price, BTC_price = signal_data
        position_value = abs(signal_coin) * coin_price + abs(signal_BTC) * BTC_price
        revenue_rate = revenue / position_value if position_value != 0 else 0
        self.revenue_rates.append(revenue_rate)

    def _update_metrics(self, coin: str, state: dict, revenue: float) -> None:
        """Update performance metrics"""
        self.coin_revenues[coin] += revenue
        self.coin_revenues_path[coin].append(
            (state['timestamp'], self.coin_revenues[coin])
        )


class StatArbitrageStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        self.params = params or {}
        super().__init__(params)

        # 1. Feature Calculation Parameters
        feature_params = self.params.get('strategy', {}).get('feature_params', {})
        self.regression_window = feature_params.get('regression_window', 50)     # 回归窗口
        self.lookback_period = feature_params.get('lookback_period', 338)       # 回看周期
        self.std_rate = feature_params.get('std_rate', 0.01)                    # 标准差更新率

        # 2. Trading Control Parameters
        trading_params = self.params.get('strategy', {}).get('trading_params', {})
        # 2.1 Signal Generation Parameters
        self.max_std = trading_params.get('max_std', 0.1)                      # 最大标准差
        self.sigma_threshold = trading_params.get('sigma_threshold', 2.0)       # 信号触发阈值
        self.sigma_threshold_u = trading_params.get('sigma_threshold_u', 4.0)   # 信号上限阈值
        # 2.2 Risk Management Parameters
        self.stop_loss = trading_params.get('lc_threshold', -0.03)             # 止损线
        self.max_holding_period = trading_params.get('holding_period', 30)      # 最大持仓期
        self.trading_fee = trading_params.get('trading_fee', 0.001)            # 交易费率
        self.bullet = trading_params.get('bullet', 0.02)                       # 仓位系数
        self.correlation_threshold = trading_params.get('correlation_threshold', 0.7)           # 相关性阈值
        self.correlation_threshold_u = trading_params.get('correlation_threshold_u', 0.5)           # 相关性阈值

        # 2.3 Bollinger Band Parameters
        self.bollinger_window = trading_params.get('bollinger_window', 60)      # 布林带窗口
        self.bollinger_threshold = trading_params.get('bollinger_threshold', 1.5)# 布林带阈值
        self.freeze_period = trading_params.get('freeze_period', 120)           # 冻结期

        # 3. Strategy State Variables
        self.signal_history = defaultdict(lambda: deque(maxlen=self.max_holding_period))  # 信号历史
        self.residual_deviate_revenue_temp = defaultdict(lambda: deque(maxlen=self.max_holding_period))  # 信号历史

        # 4. Performance Tracking
        self.revenue_rates = []                                                 # 收益率列表
        self.coin_revenues = defaultdict(float)                                # 币种累计收益
        self.coin_revenues_path = defaultdict(lambda: [])                      # 币种收益路径
        self.residual_deviate = defaultdict(lambda: [])                      # 币种收益路径
        self.residual_deviate_revenue = defaultdict(lambda: [])                      # 币种收益路径

    def get_feature_configs(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """实现特征配置"""
        return {
            'coin_features': {
                "price": {"window": self.lookback_period, "window_calc": self.regression_window },
                "beta": {"window": 1},
                "residual": { "window": self.lookback_period},
                "residual_std": { 
                    "window": 1,
                    "std_rate": self.std_rate
                },
                "position_size": {"window": 1},
                "position_avg_price": {"window": 1}
            },
            'global_features': {
                "correlation": {
                    "window": self.lookback_period,
                    "threshold_u": self.correlation_threshold_u,
                    "freeze_period": self.freeze_period
                }
            }
        }
    
    def analyze_features(self) -> Dict[str, float]:
        """
        Main feature analysis framework that processes positions and generates signals
        
        Returns:
            Dict[str, float]: Dictionary of trading signals for each coin
        """
        signals = {coin: 0.0 for coin in self.timeline.coins}
        
        try:
            for coin in self.timeline.coins:
                try:
                    # Skip coins that don't meet analysis criteria
                    if not self._should_analyze_coin(coin):
                        continue
                    
                    # Get current market state
                    market_state = self._get_market_state(coin)
                    
                    # Handle existing positions
                    revenue = self._close_positions(coin, market_state, signals)
                    
                    # Open new positions if conditions are met
                    if self._can_open_position(market_state):
                        self._generate_signals(coin, market_state, signals)
                    
                    # Update performance metrics
                    self._update_metrics(coin, market_state, revenue)
                    
                except Exception as e:
                    logger.error(f"Error processing coin {coin}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in feature analysis: {e}")
            
        return signals
    

    def _should_analyze_coin(self, coin: str) -> bool:
        """Check if coin should be analyzed"""
        current_timestamp = self.timeline.current_timestamp
        t_re = current_timestamp - self.timeline.inittime
        return coin != 'BTC' and t_re >= 2000

    def _get_market_state(self, coin: str) -> dict:
        """Get current market state for analysis"""
        return {
            'timestamp': self.timeline.current_timestamp,
            'beta': self.timeline.coin_features['beta']['data'][coin][-1],
            'residual': self.timeline.coin_features['residual']['data'][coin][-1],
            'std': self.timeline.coin_features['residual_std']['data'][coin][-1],
            'position': self.timeline.coin_features['position_size']['data'][coin][-1],
            'position_avg_price': self.timeline.coin_features['position_avg_price']['data'][coin][-1],
            'current_price_COIN': self.timeline.coin_features['price']['data'][coin][-1],
            'current_price_BTC': self.timeline.coin_features['price']['data']['BTC'][-1],
            'available_capital': float(self.timeline.capital[-1][1]),
            'total_value': float(self.timeline.total_value[-1][1]),
            'correlation_residual_deque': self.timeline.correlation_residual_deque
        }

    def _can_open_position(self, state: dict) -> bool:
        """检查是否满足开仓条件"""
        residual, std = state['residual'], state['std']
        residual_condition = (self.sigma_threshold * std < abs(residual) < 
                            self.sigma_threshold_u * std)
        return (residual_condition and 
                not self.timeline.freeze and 
                std < self.max_std and 
                state['available_capital'] > 5000)
    
    def _close_positions(self, coin: str, state: dict, signals: dict) -> float:
        """Handle position closing"""
        if not self.signal_history[coin]:
            return 0.0
            
        # return self._force_close_positions(coin, signals)
        
        return self._normal_close_positions(coin, state, signals)

    
    def _should_close_position(self, oldest_signal, revenue, state: dict) -> bool:
        """
        Check if position should be closed
        
        Args:
            open_time: Position opening timestamp
            state: Current market state
            
        Returns:
            bool: True if position should be closed
        """
        open_time, signal_coin, signal_BTC, coin_price, BTC_price = oldest_signal
        revenue_rate = revenue / (abs(signal_coin)*coin_price + abs(signal_BTC)*BTC_price)
        if revenue_rate <= self.stop_loss:
            return True

        # Check holding period
        holding_time = state['timestamp'] - open_time
        if holding_time >= self.max_holding_period - 1:
            return True
            
        # Check residual deviation
        # residual_condition = abs(state['residual']) > self.sigma_threshold_u * state['std']
        # if residual_condition:
        #     return True
            
        return False    
    

    def _should_close_position_1(self, signal_history, state: dict) -> List[int]:
        """
        Check which positions should be closed
        
        Args:
            signal_history: Deque of signals to check
            state: Current market state
            
        Returns:
            List[int]: List of indices of positions that should be closed
        """
        positions_to_close = []
        
        for i, signal in enumerate(signal_history):
            open_time, signal_coin, signal_BTC, coin_price, BTC_price = signal
            
            # Calculate revenue and revenue rate
            revenue = self._calculate_revenue(signal, state)
            position_value = abs(signal_coin)*coin_price + abs(signal_BTC)*BTC_price
            revenue_rate = revenue / position_value if position_value != 0 else 0
            
            # Check closing conditions
            holding_time = state['timestamp'] - open_time
            # residual_condition = abs(state['residual']) > self.sigma_threshold_u * state['std']
            
            if (revenue_rate <= self.stop_loss or 
                holding_time >= self.max_holding_period - 1 ):
                positions_to_close.append(i)
        
        return positions_to_close
    
    def _generate_signals(self, coin: str, state: dict, signals: dict) -> None:
        """Generate trading signals based on market state"""
        amount = self._calculate_position_size(state)
        
        if state['residual'] > 0:
            self._generate_long_signals(coin, amount, state, signals)
        else:
            self._generate_short_signals(coin, amount, state, signals)

    def _calculate_position_size(self, state: dict) -> float:
        """Calculate position size based on market conditions"""
        # Convert deque to numpy array for mean calculation
        correlation_data = np.array(state['correlation_residual_deque'])
        
        # Calculate correlation if we have enough data
        if len(correlation_data) > 100:
            corr_r = np.mean(correlation_data)
            corr_r = max(corr_r, 0.1)
            amount = 200 / corr_r
        else:
            amount = 0  # Default amount if not enough correlation data
            
        # Calculate final position size with price normalization
        return amount / (state['beta'] * state['current_price_COIN'] + 
                        state['current_price_BTC'])

    def _generate_long_signals(self, coin: str, amount: float, 
                            state: dict, signals: dict) -> None:
        """Generate long position signals"""
        beta = state['beta']
        signals[coin] += beta * amount
        signals['BTC'] += -amount
        
        self._record_signal(coin, state['timestamp'], beta * amount, -amount,
                        state['current_price_COIN'], state['current_price_BTC'])

    def _generate_short_signals(self, coin: str, amount: float, 
                            state: dict, signals: dict) -> None:
        """Generate short position signals"""
        beta = state['beta']
        signals[coin] += -beta * amount
        signals['BTC'] += amount
        
        self._record_signal(coin, state['timestamp'], -beta * amount, amount,
                        state['current_price_COIN'], state['current_price_BTC'])

    def _record_signal(self, coin: str, timestamp: int, signal_coin: float,
                    signal_BTC: float, coin_price: float, BTC_price: float) -> None:
        """Record signal in history"""
        self.signal_history[coin].append(
            (timestamp, signal_coin, signal_BTC, coin_price, BTC_price)
        )


    def _calculate_revenue(self, signal_data: tuple, state: dict) -> float:
        """Calculate position revenue"""
        _, signal_coin, signal_BTC, coin_price, BTC_price = signal_data
        return float(signal_coin * (state['current_price_COIN'] - coin_price) + 
                    signal_BTC * (state['current_price_BTC'] - BTC_price) - self.trading_fee*(abs(signal_coin) * (state['current_price_COIN'] + coin_price) + 
                    abs(signal_BTC) * (state['current_price_BTC'] + BTC_price)))

    def _update_metrics(self, coin: str, state: dict, revenue: float) -> None:
        """Update all metrics"""
        self._update_revenue_metrics(coin, state, revenue)
        self._update_deviation_metrics(coin, state)
        self._update_potential_revenue(coin, state)

    def _update_revenue_metrics(self, coin: str, state: dict, revenue: float) -> None:
        """Update revenue related metrics"""
        self.coin_revenues[coin] += revenue
        self.coin_revenues_path[coin].append(
            (state['timestamp'], self.coin_revenues[coin])
        )

    def _update_deviation_metrics(self, coin: str, state: dict) -> None:
        """Update deviation metrics"""
        deviation_ratio = abs(state['residual']) / state['std']
        self.residual_deviate[coin].append(
            (state['timestamp'], deviation_ratio)
        )

    def _update_potential_revenue(self, coin: str, state: dict) -> None:
        """
        Update and track potential revenue metrics
        
        Args:
            coin: Trading coin symbol
            state: Current market state dictionary containing:
                - current_price_BTC: Current BTC price
                - current_price_COIN: Current coin price
                - beta: Current beta value
                - residual: Current residual value
                - std: Current standard deviation
                - timestamp: Current timestamp
        """
        # Calculate trading direction based on price relationship
        direction = -1 if state['current_price_BTC'] < state['beta'] * state['current_price_COIN'] else 1
        
        # Record current state
        self.residual_deviate_revenue_temp[coin].append((
            state['timestamp'],
            abs(state['residual']) / state['std'],
            state['beta'],
            state['current_price_COIN'],
            state['current_price_BTC'],
            direction
        ))
        
        # Process potential revenue if we have enough history
        if len(self.residual_deviate_revenue_temp[coin]) == self.max_holding_period:
            # Get oldest state
            t, ratio, beta, coin_price, btc_price, dir = self.residual_deviate_revenue_temp[coin][0]
            
            # Calculate normalized revenue
            revenue = dir * (
                (state['current_price_BTC'] - btc_price) - 
                beta * (state['current_price_COIN'] - coin_price)
            ) / (abs(beta * coin_price) + abs(btc_price))
            
            # Record revenue data and remove oldest state
            self.residual_deviate_revenue[coin].append((t, ratio, revenue))
            self.residual_deviate_revenue_temp[coin].popleft()


