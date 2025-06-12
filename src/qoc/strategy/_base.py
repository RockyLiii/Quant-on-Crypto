import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque

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
    def get_feature_configs(self) -> dict[str, dict[str, dict[str, int]]]:
        """Return feature configuration dictionary"""

    def analyze_features(self) -> dict[str, float]:
        """Main feature analysis framework that processes positions and generates signals

        Returns:
            Dict[str, float]: Dictionary of trading signals for each coin
        """
        signals = dict.fromkeys(self.timeline.coins, 0.0)

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

    @abstractmethod
    def _get_market_state(self, coin: str) -> dict:
        """Get current market state"""

    @abstractmethod
    def _can_open_position(self, state: dict) -> bool:
        """Check if new position can be opened"""

    @abstractmethod
    def _generate_signals(self, coin: str, state: dict, signals: dict) -> None:
        """Generate trading signals"""

    def _close_positions(self, coin: str, state: dict, signals: dict) -> float:
        """Handle position closing"""
        if not self.signal_history[coin]:
            return 0.0

        return self._normal_close_positions(coin, state, signals)

    def _force_close_positions(self, coin: str, state: dict, signals: dict) -> float:
        """Force close all positions and calculate total revenue

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
                logger.error(
                    f"Error calculating revenue for forced close of {coin}: {e}"
                )
                position_revenue = 0.0

            # Close position
            _, signal_coin, signal_BTC, _, _ = oldest_signal
            signals[coin] -= signal_coin
            signals["BTC"] -= signal_BTC
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

            positions_to_close = self._should_close_position_1(
                self.signal_history[coin], state
            )

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

    @abstractmethod
    def _should_close_position(self, oldest_signal, revenue, state: dict) -> bool:
        """Check if position should be closed"""

    def _close_position(self, coin: str, signal_data: tuple, signals: dict) -> None:
        """Close position and update signals"""
        _, signal_coin, signal_BTC, _, _ = signal_data
        signals[coin] -= signal_coin
        signals["BTC"] -= signal_BTC
        self.signal_history[coin].popleft()

    def _close_position_1(self, coin: str, i: int, signals: dict) -> None:
        """Close position and update signals"""
        signal_history = list(self.signal_history[coin])
        signal = signal_history[i]
        _, signal_coin, signal_BTC, _, _ = signal

        # Update signals
        signals[coin] -= signal_coin
        signals["BTC"] -= signal_BTC

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
            (state["timestamp"], self.coin_revenues[coin])
        )
