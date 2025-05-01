import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Set
import logging

def preprocess_data(config: dict, logger: logging.Logger) -> None:
    """
    预处理原始数据:
    1. 对齐各币种时间戳
    2. 标准化时间戳 (除以d_t)
    3. 保存处理后的数据
    
    Args:
        config: 配置字典，包含数据路径和回测参数
        logger: 日志记录器
    """
    raw_folder = config['data']['raw_folder_path']
    output_folder = config['data']['folder_path']
    d_t = config['backtest']['d_t']  # 时间间隔，用于标准化时间戳
    start_time = config['backtest']['start_time'] 
    end_time = config['backtest']['end_time'] 
    
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"使用时间间隔 d_t={d_t} 进行标准化")
    
    # 1. 收集所有币种的时间戳
    logger.info("开始收集时间戳...")
    all_timestamps: List[Set[float]] = []
    coin_files = {}
    
    for filename in os.listdir(raw_folder):
        if filename.endswith("_klines_5m.csv"):
            coin = filename.split("_")[0]
            file_path = os.path.join(raw_folder, filename)
            coin_files[coin] = file_path
            
            # 读取该币种的所有时间戳
            timestamps = set()
            with open(file_path, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    try:
                        timestamp = float(line.strip().split(',')[0])
                        if start_time <= timestamp <= end_time:
                            timestamps.add(timestamp)
                    except (ValueError, IndexError):
                        continue
            
            if timestamps:
                all_timestamps.append(timestamps)
                logger.info(f"收集到 {coin} 的 {len(timestamps)} 个时间戳")
    
    # 2. 找出共同的时间戳并标准化
    common_timestamps = sorted(set.intersection(*all_timestamps))
    normalized_timestamps = set(int(ts / d_t) for ts in common_timestamps)
    logger.info(f"找到 {len(common_timestamps)} 个共同时间戳")
    logger.info(f"标准化后时间戳范围: {min(normalized_timestamps)} 到 {max(normalized_timestamps)}")
    
    # 3. 对每个币种的数据进行处理并保存
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
              'close_time', 'quote_volume', 'trades', 
              'taker_buy_base', 'taker_buy_quote', 'ignore']
    
    for coin, input_path in coin_files.items():
        # 读取数据到DataFrame
        df = pd.read_csv(input_path, names=columns)
        
        # 标准化时间戳
        df['timestamp'] = (df['timestamp'] / d_t).astype(int)
        df['close_time'] = (df['close_time'] / d_t).astype(int)
        
        # 筛选共同时间戳的数据
        df = df[df['timestamp'].isin(normalized_timestamps)]
        
        # 按时间戳排序
        df = df.sort_values('timestamp')
        initial_close = df['close'].iloc[0]
        logger.info(f"{coin} 初始价格: {initial_close}")
        
        # 将所有价格列除以初始收盘价
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df[col] = df[col] / initial_close
        
        # 检查数据连续性
        timestamps = df['timestamp'].values
        gaps = np.diff(timestamps)
        if not np.all(gaps == 1):
            logger.warning(f"{coin} 存在时间间隔异常，最大间隔: {gaps.max()}")
            
        # 保存处理后的数据
        output_path = os.path.join(output_folder, f"{coin}_klines_5m.csv")
        df.to_csv(output_path, index=False)
        
        logger.info(f"处理完成 {coin}，保存了 {len(df)} 行数据" + 
                   ("（覆盖已存在文件）" if os.path.exists(output_path) else ""))
        
        # 验证数据
        min_ts = df['timestamp'].min()
        max_ts = df['timestamp'].max()
        logger.info(f"{coin} 时间范围: {min_ts} 到 {max_ts}")
        
    logger.info("数据预处理完成")

    # 4. 数据验证
    logger.info("\n开始验证数据一致性...")
    first_file = True
    reference_timestamps = None
    
    for filename in os.listdir(output_folder):
        if not filename.endswith("_klines_5m.csv"):
            continue
            
        df = pd.read_csv(os.path.join(output_folder, filename))
        current_timestamps = set(df['timestamp'])
        
        if first_file:
            reference_timestamps = current_timestamps
            first_file = False
        else:
            if current_timestamps != reference_timestamps:
                logger.error(f"时间戳不一致: {filename}")
                diff = reference_timestamps.symmetric_difference(current_timestamps)
                logger.error(f"差异时间戳: {sorted(diff)[:10]}...")
            
    logger.info("数据验证完成")