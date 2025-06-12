#!/usr/bin/env python
"""批量下载多个币种的 5分钟 K线数据（来自 Binance ZIP 文件），并合并保存为 CSV"""

import os
import zipfile
from datetime import datetime, timedelta

import pandas as pd
import requests
from tqdm import tqdm


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_zip_monthly(symbol, interval, year, month, output_dir):
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{filename}"
    local_path = os.path.join(output_dir, filename)

    if os.path.exists(local_path):
        print(f"✅ 文件已存在，跳过: {filename}")
        return True

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            open(local_path, "wb") as f,
            tqdm(desc=filename, total=total, unit="B", unit_scale=True) as bar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"✅ 下载完成: {filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ 下载失败: {filename} - {e}")
        return False


def extract_and_read_csv(zip_path, output_dir):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
        all_dfs = []
        for name in zf.namelist():
            csv_path = os.path.join(output_dir, name)
            df = pd.read_csv(csv_path, header=None)
            all_dfs.append(df)
            os.remove(csv_path)  # 删除临时文件
        return all_dfs


def download_and_merge(symbol, interval, start_date, end_date, output_dir):
    all_data = []
    zip_files = []  # Track zip files for cleanup
    current_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    while current_dt <= end_dt:
        year, month = current_dt.year, current_dt.month
        success = download_zip_monthly(symbol, interval, year, month, output_dir)
        if success:
            zip_file = os.path.join(
                output_dir, f"{symbol}-{interval}-{year}-{month:02d}.zip"
            )
            zip_files.append(zip_file)  # Add to cleanup list
            dfs = extract_and_read_csv(zip_file, output_dir)
            all_data.extend(dfs)
        current_dt += timedelta(days=32)
        current_dt = current_dt.replace(day=1)

    # Cleanup zip files
    for zip_file in zip_files:
        try:
            os.remove(zip_file)
            print(f"🗑️ 删除临时文件: {zip_file}")
        except OSError as e:
            print(f"⚠️ 无法删除文件 {zip_file}: {e}")

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        df.columns = [
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close Time",
            "Quote Asset Volume",
            "Number of Trades",
            "Taker Buy Base Volume",
            "Taker Buy Quote Volume",
            "Ignore",
        ]
        return df
    print(f"⚠️ {symbol} 没有数据可用")
    return None


if __name__ == "__main__":
    # coins = ['BTC', 'DOGE', 'SHIB', 'PEPE', 'TRUMP', 'BONK', 'FARTCOIN', 'WIF', 'FLOKI', 'TURBO', 'PNUT', 'NEIRO', 'ORDI', 'BOME', 'HMSTR', 'PEOPLE', 'ELON']
    coins = ["BTC", "DOGE", "SHIB", "PEPE", "TRUMP", "BONK"]
    interval = "1m"
    start_date = "2025-02-01"
    end_date = "2025-06-01"
    output_dir = "/Users/lizeyu/Downloads/Quant-on-Crypto/data/1m_klines_n"
    ensure_directory(output_dir)

    for coin in coins:
        symbol = f"{coin}USDT"
        print(f"\n🚀 开始处理 {symbol} …")
        df = download_and_merge(symbol, interval, start_date, end_date, output_dir)
        if df is not None:
            output_csv = os.path.join(output_dir, f"{coin}_klines_{interval}.csv")
            df.to_csv(output_csv, index=False)
            print(f"✅ 合并后的数据已保存: {output_csv}")
