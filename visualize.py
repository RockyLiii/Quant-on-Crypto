







import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

    
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def plot_time_series(time_series_dict, output_path="output", filename="chart.png", 
                    title="Time Series Plot", ylabel="Value", time_unit='ms', d_t=None,start_time=None, end_time=None):
    """
    简洁地绘制多个时间序列
    Args:
        time_series_dict: 字典 {数据名称: 时间序列tensor}，每个tensor应为 [[timestamp, value], ...]
        output_path: 输出目录
        filename: 输出文件名
        title: 图表标题
        ylabel: Y轴标签
        time_unit: 时间戳单位 ('ms', 's', 等)
    """
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 创建各个序列的DataFrame
    dfs = {}
    for name, series in time_series_dict.items():
        series_data = series.numpy()
        # 为每个序列创建单独的DataFrame
        df = pd.DataFrame({
            'timestamp': series_data[:, 0],
            'value': series_data[:, 1]
        })
        if d_t is not None:
            df['timestamp'] = df['timestamp'] * d_t
        
        # 过滤时间范围
        if start_time is not None and end_time is not None:
            df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
            
        if len(df) > 0:  # 只添加非空的数据
            dfs[name] = df
    
    
    # 找到所有时间戳的交集
    timestamps = set(dfs[list(dfs.keys())[0]]['timestamp'])
    for df in dfs.values():
        timestamps = timestamps.intersection(set(df['timestamp']))
    timestamps = sorted(list(timestamps))
    
    # 创建最终的DataFrame，只包含所有序列共有的时间戳
    final_df = pd.DataFrame({'timestamp': timestamps})
    
    # 将每个序列的值添加到最终DataFrame
    for name, df in dfs.items():
        merged = pd.merge(final_df, df, on='timestamp', how='left')
        final_df[name] = merged['value']
    
    # 尝试将时间戳转换为日期时间
    try:
        final_df['datetime'] = pd.to_datetime(final_df['timestamp'], unit=time_unit)
        x_column = 'datetime'
    except Exception as e:
        print(f"时间戳转换失败: {e}")
        final_df['index'] = np.arange(len(final_df))
        x_column = 'index'
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制每个系列
    for name in time_series_dict.keys():
        plt.plot(final_df[x_column], final_df[name], label=name)
    
    # 设置图表属性
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图表
    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    # 保存数据
    data_file = os.path.join(output_path, filename.replace('.png', '.csv'))
    final_df.to_csv(data_file, index=False)
    
    return True
# 为了向后兼容，保留原来的函数
def plot_value_curve(timeline, output_path="output", d_t=None, start_time=None, end_time=None
):
    """绘制资产曲线图"""
    return plot_time_series(
        time_series_dict={
            'Total Value': timeline.total_value
        },
        output_path=output_path,
        filename="values.png",
        title="Values Over Time",
        ylabel="Value",
        d_t=d_t,
        start_time=start_time, end_time=end_time
    )
def plot_capital_curve(timeline, output_path="output", d_t=None, start_time=None, end_time=None
):
    """绘制资产曲线图"""
    return plot_time_series(
        time_series_dict={
            'Available Capital': timeline.capital
        },
        output_path=output_path,
        filename="capital.png",
        title="Capital Over Time",
        ylabel="Value",
        d_t=d_t,
        start_time=start_time, end_time=end_time
    )

def plot_position_curve(timeline, output_path="output", d_t=None, start_time=None, end_time=None
):
    """绘制资产曲线图"""
    return plot_time_series(
        time_series_dict={
            'Position Value': timeline.positions['total'][:, [0, 2]]  # 时间戳和价值
        },
        output_path=output_path,
        filename="position_values.png",
        title="Position Values Over Time",
        ylabel="Value",
        d_t=d_t,
        start_time=start_time, end_time=end_time
    )


def plot_all_coin_features(timeline, feature_name, output_path="output", d_t=None, start_time=None, end_time=None):
    """Plot one feature for all coins in the same graph"""
    # Skip if no feature records
    if feature_name not in timeline.feature_records:
        return
    
    # Prepare data for plotting
    time_series_dict = {}
    for coin in timeline.coins:
        if coin == 'BTC':  # Skip BTC for beta and residual
            if feature_name in ['beta', 'residual', 'std']:
                continue
        data = timeline.feature_records[feature_name][coin]
        if data.size(0) > 0:  # Only include coins with data
            time_series_dict[coin] = data
    
    if not time_series_dict:  # Skip if no data
        return
    
    # Plot the feature
    plot_time_series(
        time_series_dict=time_series_dict,
        output_path=output_path,
        filename=f"all_coins_{feature_name}.png",
        title=f"{feature_name.capitalize()} Values for All Coins",
        ylabel=feature_name.capitalize(),
        d_t=d_t,
        start_time=start_time, end_time=end_time

    )

def plot_all_features(timeline, strategy, output_path="output", d_t=None, start_time=None, end_time=None):
    """Plot all available features from the timeline"""
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    feature_path = os.path.join(output_path, 'features')
    os.makedirs(feature_path, exist_ok=True)
    
    # Get available features from timeline's feature records
    features = list(timeline.feature_records.keys())
    
    # Plot each available feature
    for feature in features:
        plot_all_coin_features(timeline, feature, feature_path, d_t, start_time, end_time)
    
    # Plot revenue paths if available
    if hasattr(strategy, 'coin_revenues_path'):
        plot_coin_revenues(strategy, output_path, d_t, start_time, end_time)
    # Plot residual deviation distribution
    if hasattr(strategy, 'residual_deviate'):
        plot_residual_deviation_distribution(strategy, output_path)
    
    if hasattr(strategy, 'residual_deviate_revenue'):
        plot_ratio_revenue_analysis(strategy, output_path)

    # Plot correlations if either type exists
    if (hasattr(timeline, 'correlation_price') or 
        hasattr(timeline, 'correlation_residual')):
        plot_correlation_timeline(timeline, output_path, d_t, start_time, end_time)

def plot_coin_revenues(strategy, output_path="output", d_t=None, start_time=None, end_time=None):
    """Plot cumulative revenue paths for all coins"""
    plt.figure(figsize=(12, 6))
    
    for coin, revenue_path in strategy.coin_revenues_path.items():
        if coin != 'BTC' and revenue_path:  # Skip BTC and empty paths
            # Unzip the path into timestamps and values
            timestamps, values = zip(*revenue_path)
            
            # Convert timestamps if d_t is provided
            if d_t is not None:
                timestamps = [ts * d_t for ts in timestamps]
            
            # Convert timestamps to datetime
            dates = [pd.to_datetime(ts, unit='ms') for ts in timestamps]
            
            # Plot this coin's revenue path
            plt.plot(dates, values, label=coin)
    
    plt.title('Cumulative Revenue by Coin')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Revenue')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_path, 'coin_revenues.png'))
    plt.close()

def plot_residual_deviation_distribution(strategy, output_path="output"):
    """
    Plot residual deviation distribution for all coins
    
    Args:
        strategy: Strategy instance containing residual_deviate data
        output_path: Output directory for saving the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Define bins with 0.2 intervals
    bins = np.arange(0, 10.2, 0.2)  # From 0 to 10 with 0.2 intervals
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Process each coin's data
    for coin, data in strategy.residual_deviate.items():
        if coin == 'BTC' or not data:
            continue
            
        # Extract deviation values
        timestamps, deviations = zip(*data)
        deviations = np.array(deviations)
        
        # Calculate histogram
        hist, _ = np.histogram(deviations, bins=bins, density=True)
        
        # Plot frequency distribution as line
        plt.plot(bin_centers, hist, label=coin, alpha=0.7)
    
    plt.title('Residual Deviation Distribution by Coin')
    plt.xlabel('Residual Deviation (|residual|/std)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, 'residual_deviation_dist.png'))
    plt.close()

def plot_ratio_revenue_analysis(strategy, output_path="output"):
    """
    Plot average revenue for different deviation ratios across all coins
    
    Args:
        strategy: Strategy instance containing residual_deviate_revenue data
        output_path: Output directory for saving the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Define ratio bins with 0.1 intervals
    bins = np.arange(0, 10.1, 0.1)  # From 0 to 10 with 0.1 intervals
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Process each coin's data
    for coin in strategy.residual_deviate_revenue.keys():
        if coin == 'BTC' or not strategy.residual_deviate_revenue[coin]:
            continue
            
        # Extract ratios and revenues
        t, ratios, revenues = zip(*strategy.residual_deviate_revenue[coin])
        ratios = np.array(ratios)
        revenues = np.array(revenues)
        
        # Calculate average revenue for each ratio bin
        avg_revenues = []
        for i in range(len(bins)-1):
            mask = (ratios >= bins[i]) & (ratios < bins[i+1])
            if np.any(mask):
                avg_revenues.append(np.mean(revenues[mask]))
            else:
                avg_revenues.append(np.nan)
        
        # Plot average revenues
        plt.plot(bin_centers, avg_revenues, label=f'{coin}', alpha=0.7)
    
    plt.title('Average Revenue by Deviation Ratio')
    plt.xlabel('Deviation Ratio (|residual|/std)')
    plt.ylabel('Average Revenue')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, 'ratio_revenue_analysis.png'))
    plt.close()

    


def plot_correlation_timeline(timeline, output_path="output", d_t=None, start_time=None, end_time=None):
    """Plot correlations and value on the same graph"""
    plt.figure(figsize=(15, 8))
    
    # Create two y-axes
    ax1 = plt.gca()  # For correlations
    ax2 = ax1.twinx()  # For value
    
    # Process and plot data series
    window = 50  # Rolling window size
    lines = []  # Store lines for legend
    labels = []  # Store labels for legend
    
    def safe_convert_timestamps(timestamps):
        
        try:
            processed_timestamps = []
            for ts in timestamps:
                if ts < 1000000000000:  # 1e12
                    ts = ts * d_t if d_t is not None else ts
                processed_timestamps.append(pd.to_datetime(ts, unit='ms'))
            return processed_timestamps
        except Exception as e:
            # Fallback to relative timestamps if conversion fails
            min_ts = min(timestamps)
            return [pd.to_datetime(ts - min_ts, unit='ms') for ts in timestamps]
    
    # 1. Plot price correlation
    if hasattr(timeline, 'correlation_price') and timeline.correlation_price:
        times, corrs = zip(*timeline.correlation_price)
        dates = safe_convert_timestamps(times)
        
        rolling_mean = pd.Series(corrs).rolling(window=window, min_periods=1).mean()
        line1, = ax1.plot(dates, rolling_mean, color='blue', linewidth=2, label='Price Correlation')
        lines.append(line1)
        labels.append('Price Correlation')
    
    # 2. Plot residual correlation
    if hasattr(timeline, 'correlation_residual') and timeline.correlation_residual:
        times, corrs = zip(*timeline.correlation_residual)
        dates = safe_convert_timestamps(times)
        
        rolling_mean = pd.Series(corrs).rolling(window=window, min_periods=1).mean()
        line2, = ax1.plot(dates, rolling_mean, color='green', linewidth=2, label='Residual Correlation')
        lines.append(line2)
        labels.append('Residual Correlation')
    
    # 3. Plot total value
    if hasattr(timeline, 'total_value') and len(timeline.total_value) > 0:
        values = timeline.total_value.numpy()
        times = values[:, 0]
        vals = values[:, 1]
        dates = safe_convert_timestamps(times)
        
        line3, = ax2.plot(dates, vals, color='orange', linewidth=2, 
                         linestyle='--', label='Total Value')
        lines.append(line3)
        labels.append('Total Value')
    
    # Rest of the function remains the same
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Correlation', color='black')
    ax2.set_ylabel('Total Value', color='orange')
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.legend(lines, labels, loc='upper left')
    plt.title('Market Correlations and Total Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, 'correlation_value_timeline.png'), 
                bbox_inches='tight')
    plt.close()