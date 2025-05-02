







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


def plot_all_coin_features(timeline, feature_name, output_path="output", d_t=None,        start_time=None, end_time=None):
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

def plot_all_features(timeline, output_path="output", d_t=None, start_time=None, end_time=None):
    """Plot all features"""
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    feature_path = os.path.join(output_path, 'features')
    os.makedirs(feature_path, exist_ok=True)
    
    # Plot each feature
    features = ['price', 'beta', 'residual', 'std']
    for feature in features:
        plot_all_coin_features(timeline, feature, feature_path, d_t)