import collections
import os
from collections import deque
from pathlib import Path

import attrs
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pendulum
import polars as pl
from liblaf import cherries
from loguru import logger
from pendulum import DateTime, Duration

import qoc
from qoc.api.typing import Balance, OrderResponseFull, OrderSide, Symbol


class Config(cherries.BaseConfig):
    online: bool = False


@attrs.frozen
class Order:
    quantity: float
    symbol: Symbol
    time: DateTime


@attrs.define
class TimeSeriesData:
    """Flexible time series storage structure"""

    data: dict[str, list[tuple[DateTime, float]]] = attrs.field(factory=dict)

    def add_metric(self, metric_name: str, timestamp: DateTime, value: float) -> None:
        """Add a metric value at a specific timestamp"""
        if metric_name not in self.data:
            self.data[metric_name] = []
        self.data[metric_name].append((timestamp, value))

    def get_metric(self, metric_name: str) -> list[tuple[DateTime, float]]:
        """Get time series data for a specific metric"""
        return self.data.get(metric_name, [])

    def to_dataframe(self, metric_name: str) -> pl.DataFrame:
        """Convert a metric's time series to Polars DataFrame"""
        if metric_name not in self.data:
            return pl.DataFrame()

        timestamps, values = zip(*self.data[metric_name], strict=False)
        return pl.DataFrame({"timestamp": timestamps, "value": values})

    def get_all_metrics(self) -> list[str]:
        """Get list of all available metrics"""
        return list(self.data.keys())


@attrs.define
class StrategyRev:
    symbols: list[Symbol] = attrs.field(factory=lambda: ["BTCUSDT"])

    # -------------------------------- Config -------------------------------- #

    past_window: Duration = attrs.field(factory=lambda: pendulum.duration(hours=5))
    """过去窗口"""

    future_end: Duration = attrs.field(factory=lambda: pendulum.duration(hours=30))
    """持有期"""

    bullet_size: float = 25
    """单次下单资金 (USDT)"""

    past_threshold: float = -0.02
    """买入阈值 (跌幅)"""

    max_holdings: int = 5
    """单标最大持仓 (单)"""

    # --------------------------------- State -------------------------------- #
    orders: collections.defaultdict[str, deque[Order]] = attrs.field(
        factory=lambda: collections.defaultdict(deque)
    )

    # Time series storage
    time_series: TimeSeriesData = attrs.field(factory=TimeSeriesData)
    """Flexible time series storage for all metrics"""

    # Track last logged date for daily logging
    last_logged_date: str = attrs.field(default="")
    """Last logged date for daily progress tracking"""

    def get_time_series(self) -> TimeSeriesData:
        """Return the complete time series data"""
        return self.time_series

    def step(self, api: qoc.Api) -> None:
        now: DateTime = qoc.now()

        # Check if we've entered a new day and log progress
        current_date = now.format("YYYY-MM-DD")
        if current_date != self.last_logged_date:
            logger.info(f"Trading Date: {current_date}")
            self.last_logged_date = current_date

        orders: deque[Order] = deque()  # Initialize with a default value
        for symbol in self.symbols:
            orders = self.orders[symbol]
            klines: pl.DataFrame = api.klines(
                symbol, "1m", endTime=now - self.past_window, limit=1
            )
            past_price: float = klines["close"].last()  # pyright: ignore[reportAssignmentType]
            price: float = api.price(symbol, interval="1m")
            past_growth: float = (price - past_price) / past_price

            # logger.debug(
            #     "{} past growth: {:.4f}, now holdings: {}",
            #     symbol,
            #     past_growth,
            #     len(orders),
            # )
            if past_growth < self.past_threshold and len(orders) < self.max_holdings:
                quantity: float = self.bullet_size / price
                response: OrderResponseFull = api.order_market(
                    symbol, OrderSide.BUY, quantity=quantity
                )
                self.orders[symbol].append(
                    Order(
                        quantity=response.executed_qty,
                        symbol=symbol,
                        time=response.transact_time,
                    )
                )
            while orders and orders[0].time + self.future_end < now:
                order: Order = orders.popleft()
                api.order_market(order.symbol, OrderSide.SELL, quantity=order.quantity)
            self.orders[symbol] = orders

        self.time_series.add_metric("now holding", now, len(orders))

    def dump(self, api: qoc.Api) -> None:
        clock: qoc.Clock = qoc.get_clock()
        cherries.log_metric("time", clock.now.timestamp(), step=clock.step)

        for symbol in self.symbols:
            orders: deque[Order] = self.orders[symbol]
            cherries.log_metric(f"holdings/{symbol}", len(orders), step=clock.step)

        total_value: float = 0.0
        balances: list[Balance] = api.account(omitZeroBalances=True).balances
        for balance in balances:
            if balance.asset == "USDT":
                total_value += balance.free + balance.locked
            else:
                symbol: Symbol = balance.asset + "USDT"
                price: float = api.price(symbol, interval="1m")
                total_value += (balance.free + balance.locked) * price

        # Store total value with datetime index
        self.time_series.add_metric("total_value", clock.now, total_value)

        cherries.log_metric("Total Value (USDT)", total_value, step=clock.step)
        cherries.log_metric(
            "price", api.price("BTCUSDT", interval="1m"), step=clock.step
        )


def plot_time_series(tss: TimeSeriesData, output_dir: str = "./plots") -> None:
    """Plot time series data and save to specified directory

    Args:
        tss: TimeSeriesData object containing the time series data
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all available metrics
    metrics = tss.get_all_metrics()

    if not metrics:
        logger.warning("No metrics found in time series data")
        return

    logger.info(f"Plotting {len(metrics)} metrics: {metrics}")

    # Plot each metric separately
    for metric in metrics:
        data = tss.get_metric(metric)

        if not data:
            logger.warning(f"No data found for metric: {metric}")
            continue

        # Extract timestamps and values
        timestamps, values = zip(*data, strict=False)

        # Convert pendulum DateTime to python datetime for matplotlib
        datetime_list = []
        for ts in timestamps:
            if hasattr(ts, "to_datetime"):
                # For pendulum DateTime objects
                datetime_list.append(ts.to_datetime())
            elif hasattr(ts, "in_timezone"):
                # Alternative method for pendulum DateTime
                datetime_list.append(ts.in_timezone("UTC").replace(tzinfo=None))
            else:
                # For regular datetime objects
                datetime_list.append(ts)

        # Create the plot
        plt.figure(figsize=(14, 8))
        plt.plot(datetime_list, values, linewidth=2, marker="o", markersize=2)

        # Format the plot
        plt.title(f"Time Series: {metric}", fontsize=16, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Calculate time span to set appropriate tick interval
        time_span = datetime_list[-1] - datetime_list[0]
        days = time_span.days

        # Set appropriate time locator based on data span
        if days <= 1:
            # For data spanning 1 day or less, show every 2 hours
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        elif days <= 7:
            # For data spanning up to a week, show every 12 hours
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        elif days <= 30:
            # For data spanning up to a month, show daily
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        else:
            # For longer periods, show weekly
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        # Limit maximum number of ticks to prevent the warning
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

        plt.xticks(rotation=45)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        safe_filename = metric.replace("/", "_").replace(" ", "_")
        output_path = os.path.join(output_dir, f"{safe_filename}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot for {metric} to: {output_path}")


def main(cfg: Config) -> None:
    if cfg.online:
        qoc.set_clock(qoc.ClockOnline(interval="5m"))
    else:
        qoc.set_clock(
            qoc.ClockOffline(interval="5m", start="2024-01-01", end="2025-01-01")
        )
    api: qoc.Api = qoc.ApiBinanceSpot() if cfg.online else qoc.ApiOfflineSpot()
    strategy = StrategyRev()

    logger.info("Starting strategy execution...")

    for _ in qoc.loop():
        try:
            strategy.step(api=api)
            strategy.dump(api=api)
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logger.exception("error")

    logger.info("Strategy execution completed, generating plots...")
    tss = strategy.get_time_series()
    plot_time_series(tss, output_dir="deploy/rev/plots")


if __name__ == "__main__":
    cherries.main(main)


# BINANCE_BASE_URL='https://api.binance.us' python /Users/lizeyu/Desktop/Quant-on-Crypto/deploy/rev/main.py
