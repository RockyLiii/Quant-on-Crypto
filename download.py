#!/usr/bin/env python
"""
独立版 BTC 5分钟数据下载脚本 (2024.1.1-2025.1.1)
"""
import os
import sys
import requests
from datetime import datetime, timedelta
import pandas as pd

def ensure_directory(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_klines(symbol, interval, start_date, end_date, output_dir):
    """
    下载Binance K线数据
    :param symbol: 交易对，如 BTCUSDT
    :param interval: 时间间隔，如 5m
    :param start_date: 开始日期 (YYYY-MM-DD)
    :param end_date: 结束日期 (YYYY-MM-DD)
    :param output_dir: 输出目录
    """
    base_url = "https://data.binance.vision/data/spot/monthly/klines"
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    current_dt = start_dt
    while current_dt <= end_dt:
        year = current_dt.year
        month = current_dt.month
        
        # 生成文件名
        filename = f"{symbol.upper()}-{interval}-{year}-{month:02d}.zip"
        url = f"{base_url}/{symbol.upper()}/{interval}/{filename}"
        
        # 本地保存路径
        local_path = os.path.join(output_dir, filename)
        
        # 如果文件已存在则跳过
        if os.path.exists(local_path):
            print(f"文件已存在，跳过: {filename}")
            current_dt += timedelta(days=32)  # 跳到下个月
            current_dt = current_dt.replace(day=1)
            continue
        
        print(f"正在下载: {filename}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"下载完成: {filename}")
        except requests.exceptions.HTTPError as e:
            print(f"下载失败 {filename}: {e}")
        
        # 跳到下个月
        current_dt += timedelta(days=32)  # 确保跳到下个月
        current_dt = current_dt.replace(day=1)

if __name__ == "__main__":
    # 配置参数
    SYMBOL = "BTCUSDT"
    INTERVAL = "5m"
    START_DATE = "2024-01-01"
    END_DATE = "2025-01-01"
    OUTPUT_DIR = "/Users/lizeyu/Downloads/bbb"  # 修改为你想要的目录
    
    # 确保输出目录存在
    ensure_directory(OUTPUT_DIR)
    
    print(f"开始下载 {SYMBOL} {INTERVAL} 数据 ({START_DATE} 至 {END_DATE})")
    download_klines(SYMBOL, INTERVAL, START_DATE, END_DATE, OUTPUT_DIR)
    print(f"下载完成! 数据已保存到: {os.path.abspath(OUTPUT_DIR)}")