# System Design

## Overview

This document outlines the architecture for a quantitative trading system for cryptocurrency markets. The system supports both backtesting and live trading with a modular feature-based approach.

## Core Components

### State Management

```python
class State:
    raw: ...
    trade: ...
    features: ...
```

The `State` class maintains the current system state including:

- **raw**: Market data and historical information
- **trade**: Current trading positions
- **features**: Computed technical indicators and features

### Feature System

#### Base Feature Interface

```python
class Feature:
    def compute(self, state: State) -> State: ...
```

All features inherit from `Feature` and implement the `compute` method to compute indicators from timeline data.

#### Feature Implementations

```python
class Feature1(Feature):
    window: int = 100

    def compute(self, state: State) -> State: ...

class Feature2(Feature):
    param: float = 1.0

    def compute(self, state: State) -> State: ...
```

Features are configurable with parameters:

- **Feature1**: Window-based indicator (e.g., moving average)
- **Feature2**: Parameter-driven indicator

#### Feature Store

```python
class FeatureStore:
    def compute(self, state: State) -> State: ...

features = FeatureStore(features=[Feature1(window=100), Feature2(param=1.0)])
```

The `FeatureStore` manages feature computation and updates the system state with calculated indicators.

### Trading Strategy

```python
class Strategy:
    stats: Statistics
    timeline: Timeline

    def needs_update(self, timestamp: Timestamp) -> bool: ...
    def gen_signal(self): ...
```

The `Strategy` class:

- Determines when updates are needed
- Generates trading signals based on computed features
- Maintains statistics and timeline data

### Trading Systems

#### Trade System Interface

```python
class TradeSystemAPI:
    def place_order(self, coin: CoinName, amount: float, price: float) -> None: ...
    def cancel_order(self, coin: CoinName) -> None: ...
    def query_order(self, coin: CoinName) -> OrderStatus: ...
```

#### Implementations

- **LocalTradeSystem**: For backtesting and simulation
- **BinanceTradeSystem**: For live trading on Binance exchange

## System Workflows

### Main Trading Loop

```python
def main() -> None:
    timeline = Timeline()
    stats = Statistics(
        timeline=timeline,
        feature_configs=[Feature1(window=100), Feature2(param=1.0)]
    )
    strategy = Strategy(stats=stats, timeline=timeline)
```

### Backtesting Workflow

```python
def backtest():
    # Initialize timelines
    raw_timeline = Timeline()
    trade_timeline = Timeline()
    strategy_timeline = Timeline()

    # Setup strategy and trade system
    strategy = Strategy(
        stats=Statistics(
            timeline=strategy_timeline,
            feature_configs=[Feature1(window=100), Feature2(param=1.0)],
        )
    )
    trade = TradeSystem(timeline=trade_timeline, strategy=strategy)

    # Main backtesting loop
    for timestamp in timestamps:
        # 1. Update raw market data
        # 2. Update trade system data
        # 3. Update strategy data (features, positions, capital)
        # 4. Generate and apply trading signals

        trade.update(raw_data.latest)
        strategy.time_pass(trade.latest)
        strategy.stats.feature_calc()
        signals = strategy.gen_signal()

        # Execute trades based on signals
        for coin, signal in signals.items():
            trade.trade(coin, signal)
```

## Data Flow

1. **Raw Data Ingestion**: Market data flows into the raw timeline
2. **Trade Data Update**: Trading system processes and updates trade timeline
3. **Feature Computation**: Features are calculated from available data
4. **Signal Generation**: Strategy generates trading signals based on features
5. **Trade Execution**: Signals are executed through the trading system API
