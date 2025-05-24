import os
import torch
import logging
import pandas as pd
from typing import Type, Dict
from timeline import StatArbitrageTimeline, BaseTimeline
from Strategy import BaseStrategy
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import statsmodels.api as sm  # 添加这行

def analyze_ccf(strategy, max_lag=45, output_path="output"):
    """Analyze cross-correlation between ratio and revenue for each coin"""
    os.makedirs(output_path, exist_ok=True)
    ccf_results = {}
    
    for coin, data in strategy.residual_deviate_revenue.items():
        if coin == 'BTC' or not data:
            continue
            
        # Extract time series
        timestamps, ratios, revenues = zip(*sorted(data))
        ratios = np.array(ratios)
        revenues = np.array(revenues)
        
        results = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                x_shifted = ratios[:lag]
                y_shifted = revenues[-lag:]
            elif lag > 0:
                x_shifted = ratios[lag:]
                y_shifted = revenues[:-lag]
            else:
                x_shifted = ratios
                y_shifted = revenues
                
            try:
                # Calculate beta using OLS
                X_reg = sm.add_constant(x_shifted)
                model = sm.OLS(y_shifted, X_reg).fit()
                beta = model.params[1]
                
                # Calculate correlation
                corr = np.corrcoef(x_shifted, y_shifted)[0, 1]
                
                results.append({
                    'Lag': lag,
                    'Beta': beta,
                    'Correlation': corr,
                    'P_value': model.pvalues[1]
                })
            except:
                continue
        
        ccf_results[coin] = pd.DataFrame(results)
    
    return ccf_results


def visualize_ccf_heatmap(ccf_results, max_lag=45, output_path="output"):
    """Create heatmap tables for CCF analysis with better text visibility"""
    # Calculate figure size based on data
    n_coins = len(ccf_results)
    n_lags = 2 * max_lag + 1
    fig_width = max(20, n_lags * 0.5)  # Minimum width of 20
    fig_height = max(10, n_coins * 1.0) # Adjust height based on number of coins
    
    plt.figure(figsize=(fig_width, fig_height))
    
    coins = list(ccf_results.keys())
    lags = range(-max_lag, max_lag + 1)
    
    beta_matrix = np.zeros((len(coins), len(lags)))
    corr_matrix = np.zeros((len(coins), len(lags)))
    
    # Prepare data matrices
    for i, coin in enumerate(coins):
        df = ccf_results[coin]
        for j, lag in enumerate(lags):
            row = df[df['Lag'] == lag]
            if not row.empty:
                beta_matrix[i, j] = row['Beta'].values[0]
                corr_matrix[i, j] = row['Correlation'].values[0]
    
    # Plot beta heatmap
    plt.subplot(2, 1, 1)
    im1 = plt.imshow(beta_matrix, aspect='auto', cmap='RdYlBu')
    plt.colorbar(im1, label='Beta Coefficient')
    plt.yticks(range(len(coins)), coins)
    plt.xticks(range(len(lags)), lags, rotation=45)
    plt.title('Beta Coefficients by Lag and Coin', pad=20)
    plt.xlabel('Lag')
    plt.ylabel('Coin')
    
    # Add text annotations for beta
    for i in range(len(coins)):
        for j in range(len(lags)):
            value = beta_matrix[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            plt.text(j, i, f'{value:.2f}', 
                    ha='center', va='center',
                    color=color,
                    fontsize=8,  # Smaller font size
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))  # Add background
    
    # Plot correlation heatmap
    plt.subplot(2, 1, 2)
    im2 = plt.imshow(corr_matrix, aspect='auto', cmap='RdYlBu')
    plt.colorbar(im2, label='Correlation Coefficient')
    plt.yticks(range(len(coins)), coins)
    plt.xticks(range(len(lags)), lags, rotation=45)
    plt.title('Correlation Coefficients by Lag and Coin', pad=20)
    plt.xlabel('Lag')
    plt.ylabel('Coin')
    
    # Add text annotations for correlation
    for i in range(len(coins)):
        for j in range(len(lags)):
            value = corr_matrix[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            plt.text(j, i, f'{value:.2f}', 
                    ha='center', va='center',
                    color=color,
                    fontsize=8,  # Smaller font size
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))  # Add background
    
    plt.tight_layout(h_pad=2.0)  # Increase spacing between subplots
    plt.savefig(os.path.join(output_path, 'ccf_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    return pd.DataFrame({
        'Coin': coins,
        'Max Beta': [np.max(np.abs(beta_matrix[i,:])) for i in range(len(coins))],
        'Beta Lag': [lags[np.argmax(np.abs(beta_matrix[i,:]))] for i in range(len(coins))],
        'Max Corr': [np.max(np.abs(corr_matrix[i,:])) for i in range(len(coins))],
        'Corr Lag': [lags[np.argmax(np.abs(corr_matrix[i,:]))] for i in range(len(coins))]
    })



def analyze_results(timeline, strategy, config, logger):
    """分析回测结果并生成报告"""
    logger.info("开始分析回测结果...")
    
    # 创建输出目录
    output_path = config.get('output_path', 'output')
    os.makedirs(output_path, exist_ok=True)
    
    # 提取收益率数据
    capital = timeline.capital.numpy()
    total_value = timeline.total_value.numpy()
    tradeamount = timeline.tradeamount
    
    # 转换为pandas DataFrame便于分析
    df = pd.DataFrame({
        'timestamp': capital[:, 0],
        'capital': capital[:, 1],
        'total_value': total_value[:, 1]
    })
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    # 计算收益指标
    initial_capital = config['backtest']['initial_capital']
    final_value = df['total_value'].iloc[-1]  # 使用total_value而不是capital
    absolute_return = final_value - initial_capital
    total_return = (final_value / initial_capital - 1) * 100
    
    # 计算最大回撤
    df['cummax'] = df['total_value'].cummax()
    df['drawdown'] = (df['total_value'] / df['cummax'] - 1) * 100
    max_drawdown = df['drawdown'].min()
    


    # 计算交易效率指标
    trading_fee = config['backtest']['trading_fee']
    turnover_ratio = tradeamount / initial_capital  # 换手率
    roi_turnover_ratio = absolute_return / tradeamount if tradeamount > 0 else 0  # 收益交易比
    fee_cost = tradeamount * trading_fee  # 总手续费成本
    
    # 统计交易信号
    total_signals = timeline.signal_count['total']
    signal_counts = {coin: count for coin, count in timeline.signal_count.items() if coin != 'total'}
    most_active_coin = max(signal_counts.items(), key=lambda x: x[1]) if signal_counts else ('None', 0)
    freeze_rate = timeline.freeze_rate

    # 计算收益率统计
    revenue_rates = np.array(strategy.revenue_rates)
    avg_revenue_rate = np.mean(revenue_rates) if len(revenue_rates) > 0 else 0
    std_revenue_rate = np.std(revenue_rates) if len(revenue_rates) > 0 else 0
    skewness = scipy.stats.skew(revenue_rates)
    kurtosis = scipy.stats.kurtosis(revenue_rates)
    
    
    
    # 输出结果
    logger.info("\n====== 回测结果摘要 ======")

    logger.info("\n各币种交易次数:")
    for coin, count in sorted(signal_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:  # 只显示有交易的币种
            logger.info(f"{coin}: {count}次")

    logger.info(f"初始资金: {initial_capital:.2f}")
    logger.info(f"最终价值: {final_value:.2f}")
    logger.info(f"绝对收益: {absolute_return:.2f}")
    logger.info(f"总收益率: {total_return:.2f}%")

    logger.info(f"\n交易效率:")
    logger.info(f"总手续费: {fee_cost:.2f}")
    logger.info(f"总交易量: {tradeamount:.2f}")
    logger.info(f"换手率: {turnover_ratio:.2f}")
    logger.info(f"收益交易比: {roi_turnover_ratio*100:.5f}%")

    logger.info(f"\n风险指标:")
    logger.info(f"最大回撤: {max_drawdown:.2f}%")

    logger.info(f"\n交易统计:")
    logger.info(f"信号次数: {total_signals}")
    logger.info(f"冻结比例: {freeze_rate*100:.2f}%")
    logger.info(f"平均每笔收益率: {avg_revenue_rate*100:.2f}%")
    logger.info(f"收益率标准差: {std_revenue_rate*100:.2f}%")
    logger.info(f"收益率偏度: {skewness:.2f}")  # 正值表示右偏，负值表示左偏
    logger.info(f"收益率峰度: {kurtosis:.2f}")  # 正值表示尖峰，负值表示平峰


    ccf_results = analyze_ccf(strategy, max_lag=45, output_path=output_path)
    ccf_summary = visualize_ccf_heatmap(ccf_results, max_lag=45, output_path=output_path)
    

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'total_signals': total_signals,
        'signal_counts': signal_counts,
        'most_active_coin': most_active_coin,
        'avg_revenue_rate': avg_revenue_rate,
        'std_revenue_rate': std_revenue_rate,
        'skewness': skewness,
        'kurtosis': kurtosis,
    }

def analyze_performance(timeline: BaseTimeline, logger: logging.Logger) -> None:
    """分析特征计算性能随时间的变化"""
    if not hasattr(timeline, '_performance_log'):
        logger.info("没有可用的性能日志")
        return
        
    logs = timeline._performance_log
    data_sizes = [log['data_size'] for log in logs]
    total_times = [sum(log['times'].values()) for log in logs]
    
    logger.info("\n=== 性能分析 ===")
    logger.info(f"样本数: {len(logs)}")
    logger.info(f"数据量增长: {data_sizes[0]} -> {data_sizes[-1]}")
    logger.info(f"计算时间增长: {total_times[0]:.4f}s -> {total_times[-1]:.4f}s")
    
    # 分析各特征的时间变化
    for feature in logs[0]['times'].keys():
        feature_times = [log['times'][feature] for log in logs]
        logger.info(f"\n{feature} 特征:")
        logger.info(f"初始时间: {feature_times[0]:.4f}s")
        logger.info(f"最终时间: {feature_times[-1]:.4f}s")

def run_backtest(strategy: BaseStrategy, logger: logging.Logger, backtest_params: dict, progress_callback=None) -> Dict:
    """执行回测"""
    timeline = StatArbitrageTimeline(
        starttime=backtest_params['backtest']['start_time'],
        endtime=backtest_params['backtest']['end_time'],
        feature_configs=strategy.get_feature_configs(),
        trading_fee=backtest_params['backtest']['trading_fee'],
        initial_capital=backtest_params['backtest']['initial_capital']
    )

    strategy.timeline = timeline
    start_load = time.time()

    # Initialize data dictionary
    data_dict = {}
    for filename in os.listdir(backtest_params['data']['folder_path']):
        if filename.endswith('_klines_5m.csv'):
            coin = filename.split('_')[0]
            file_path = os.path.join(backtest_params['data']['folder_path'], filename)
            df = pd.read_csv(file_path)
            data_dict[coin] = df
            timeline._initialize_for_coin(coin)
    logger.info(f"数据加载耗时: {time.time() - start_load:.4f} 秒")

    
    # Get common timestamps
    all_timestamps = sorted(set.intersection(*[set(df['timestamp']) for df in data_dict.values()]))
    total_len = len(all_timestamps)
    logger.info(f"Found {total_len} common timestamps")

    load_time = time.time() - start_load
    total_time = 0
    feature_calc_time = 0
    data_update_time = 0
    signal_calc_time = 0
    loop_overhead_time = 0

    # Process timestamps with progress tracking
    loop_start = time.time()

    for i, timestamp in enumerate(all_timestamps):
        try:
            # 判断是否是最后两个时间戳
            is_last2_timestamp = (i == len(all_timestamps) - 2)
            is_last_timestamp = (i == len(all_timestamps) - 1)

            data_start = time.time()
            new_data = {}
            
            # Collect data for this timestamp
            for coin, df in data_dict.items():
                row = df[df['timestamp'] == timestamp]
                if not row.empty:
                    new_data[coin] = torch.tensor(row.values[0], dtype=torch.float32).unsqueeze(0)
            data_update_time += time.time() - data_start

            if new_data:
                # 1. 更新时间线和计算特征
                feature_start = time.time()
                timeline.time_pass(new_data)
                feature_calc_time += time.time() - feature_start
                
                if is_last2_timestamp:
                    # 在倒数第二个时间戳执行强制平仓
                    logger.info("执行最终平仓操作")
                    POSITION_THRESHOLD = 0.0001  # 设置最小仓位阈值
                    
                    for coin in timeline.coins:
                        current_position = timeline.coin_features['position_size']['data'][coin][-1]
                        if abs(current_position) > POSITION_THRESHOLD:
                            logger.info(f"平仓 {coin}: {current_position:.4f}")
                            timeline.trade(coin, -current_position)  # 生成反向交易信号
                
                elif is_last_timestamp:
                    # 最后一个时间戳只记录最终状态
                    for coin in timeline.coins:
                        timeline.trade(coin, 0)
                
                else:
                    # 正常交易时段
                    signals = strategy.analyze_features()
                    for coin, signal in signals.items():
                        timeline.trade(coin, signal)
            
            
            # Print comprehensive timing stats every 1000 timestamps
            if i > 0 and i % 1000 == 0:
                total_time = time.time() - loop_start
                logger.info(f"\n=== 性能统计 [{i}/{total_len}] ===")
                logger.info(f"总运行时间: {total_time:.4f}s")
                logger.info(f"数据加载时间: {load_time:.4f}s ({load_time/total_time*100:.1f}%)")
                logger.info(f"数据更新时间: {data_update_time:.4f}s ({data_update_time/total_time*100:.1f}%)")
                logger.info(f"特征计算与交易时间: {feature_calc_time:.4f}s ({feature_calc_time/total_time*100:.1f}%)")
                logger.info("========================\n")
            
        except Exception as e:
            logger.error(f"Error at timestamp {timestamp} (index {i}): {str(e)}")
            logger.error("Detailed error:", exc_info=True)
            raise
        
    analyze_performance(timeline, logger)
    

    
    return timeline
