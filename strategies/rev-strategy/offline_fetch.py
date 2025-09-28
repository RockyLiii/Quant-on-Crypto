import datetime
import os

import arcticdb as adb

#!/usr/bin/env python
"""批量下载多个币种的 5分钟 K线数据（来自 Binance ZIP 文件），并合并保存为 CSV"""

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


def check_data_coverage(library, symbol_key, start_date, end_date, interval):
    """逐条检查数据库中数据是否完全覆盖指定的时间范围，确保没有数据缺口

    Args:
        library: Arctic库对象
        symbol_key: 数据的键名
        start_date: 目标开始日期 (YYYY-MM-DD)
        end_date: 目标结束日期 (YYYY-MM-DD)
        interval: 时间间隔 (例如 "1m", "5m", "1h")

    Returns:
        tuple: (is_covered, missing_periods, existing_df)
            - is_covered: 布尔值，表示是否完全覆盖
            - missing_periods: 缺失数据的时间段列表 [(start1, end1), (start2, end2), ...]
            - existing_df: 数据库中的现有数据，如果没有则为空DataFrame
    """
    from datetime import datetime, timedelta

    import pandas as pd

    # 解析时间间隔
    interval_map = {
        "1s": timedelta(seconds=1),
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }

    if interval not in interval_map:
        print(f"⚠️ 不支持的时间间隔: {interval}，使用默认间隔5分钟")
        interval_delta = timedelta(minutes=5)
    else:
        interval_delta = interval_map[interval]

    # 转换日期字符串为datetime对象
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # 转换为毫秒时间戳
    start_timestamp = int(start_dt.timestamp() * 1000000)
    end_timestamp = int(end_dt.timestamp() * 1000000)

    print(start_timestamp, end_timestamp)

    if symbol_key in library.list_symbols():
        try:
            # 读取数据
            arctic_result = library.read(symbol_key)
            existing_df = arctic_result.data

            if existing_df.empty:
                print(f"数据库中 {symbol_key} 的数据为空")
                return False, [(start_dt, end_dt)], existing_df

            # 检查索引类型并确保是时间戳格式
            print(f"当前索引: {(existing_df.index)}")

            # 处理索引问题
            if "Close Time" in existing_df.columns:
                print("'Close Time'在列中，设置为索引")
                existing_df = existing_df.reset_index()
                existing_df.set_index("Close Time", inplace=True)

            # 确保索引是按时间排序的
            existing_df = existing_df.sort_index()

            # 获取数据库中的最早和最晚时间戳
            db_min_time = existing_df.index.min()
            db_max_time = existing_df.index.max()
            print(db_min_time, db_max_time)

            print(
                f"数据库中的整体时间范围: {datetime.fromtimestamp(int(db_min_time / 1000000))} 到 {datetime.fromtimestamp(int(db_max_time / 1000000))}"
            )
            print(f"请求的时间范围: {start_dt} 到 {end_dt}")

            # 检查数据是否覆盖了请求的时间范围
            is_in_range = (db_min_time <= start_timestamp) and (
                db_max_time >= end_timestamp
            )

            if not is_in_range:
                print("⚠️ 数据库时间范围不完全覆盖请求的时间范围")
                missing_periods = []

                # 添加前期缺失区间
                if db_min_time > start_timestamp:
                    missing_periods.append(
                        (start_dt, datetime.fromtimestamp(db_min_time / 1000))
                    )
                    print(
                        f"  缺少前期数据: {start_dt} 到 {datetime.fromtimestamp(db_min_time / 1000)}"
                    )

                # 添加后期缺失区间
                if db_max_time < end_timestamp:
                    missing_periods.append(
                        (datetime.fromtimestamp(db_max_time / 1000), end_dt)
                    )
                    print(
                        f"  缺少后期数据: {datetime.fromtimestamp(db_max_time / 1000)} 到 {end_dt}"
                    )

                return False, missing_periods, existing_df

            # 逐条检查数据连续性
            print("正在逐条检查数据连续性...")

            # 将索引转换为datetime，便于处理
            existing_df_datetime = existing_df.reset_index()
            existing_df_datetime["datetime"] = pd.to_datetime(
                existing_df_datetime["Close Time"], unit="us"
            )
            existing_df_datetime = existing_df_datetime.sort_values("datetime")

            # 限制在请求的时间范围内
            mask = (existing_df_datetime["datetime"] >= start_dt) & (
                existing_df_datetime["datetime"] <= end_dt
            )
            filtered_df = existing_df_datetime[mask]

            if filtered_df.empty:
                print("⚠️ 过滤后的数据为空，可能是日期范围问题")
                return False, [(start_dt, end_dt)], existing_df

            # 检查每个时间点之间的间隔
            missing_periods = []
            current_dt = filtered_df["datetime"].iloc[0]
            expected_dt = start_dt

            # 先检查开始时间
            if current_dt > expected_dt + interval_delta:
                missing_periods.append((expected_dt, current_dt - interval_delta))
                print(f"  缺少数据: {expected_dt} 到 {current_dt - interval_delta}")

            # 检查所有数据点间隔
            for i in range(1, len(filtered_df)):
                prev_dt = filtered_df["datetime"].iloc[i - 1]
                curr_dt = filtered_df["datetime"].iloc[i]

                # 预期的下一个时间点
                expected_next_dt = prev_dt + interval_delta

                # 如果实际时间点晚于预期，说明有缺口
                if curr_dt > expected_next_dt + timedelta(seconds=1):  # 允许1秒误差
                    missing_periods.append((expected_next_dt, curr_dt - interval_delta))
                    print(
                        f"  发现数据缺口: {expected_next_dt} 到 {curr_dt - interval_delta}"
                    )

            # 检查结束时间
            last_dt = filtered_df["datetime"].iloc[-1]
            if last_dt < end_dt - interval_delta:
                missing_periods.append((last_dt + interval_delta, end_dt))
                print(f"  缺少数据: {last_dt + interval_delta} 到 {end_dt}")

            if missing_periods:
                print(f"⚠️ 共发现 {len(missing_periods)} 个数据缺口")
                return False, missing_periods, existing_df
            print("✓ 数据连续性检查通过，完全覆盖请求的时间范围")
            return True, [], existing_df

        except Exception as e:
            print(f"读取数据时出错: {e}")
            return False, [(start_dt, end_dt)], pd.DataFrame()
    else:
        print(f"数据库中没有 {symbol_key} 的数据")
        return False, [(start_dt, end_dt)], pd.DataFrame()


def get_all_timestamps_in_range(library, coins, interval, start_date, end_date):
    """获取特定日期范围内所有币种的K线时间戳"""
    from datetime import datetime

    # 转换日期为时间戳 (微秒级)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_timestamp = int(start_dt.timestamp() * 1000000)
    end_timestamp = int(end_dt.timestamp() * 1000000)

    all_timestamps = []
    for coin in coins:
        symbol_key = f"{coin}_klines_{interval}"
        if symbol_key in library.list_symbols():
            print(f"读取 {symbol_key} 的时间戳...")
            arctic_result = library.read(symbol_key)
            df = arctic_result.data

            # 筛选指定日期范围内的时间戳
            filtered_timestamps = [
                ts for ts in df.index if start_timestamp <= ts <= end_timestamp
            ]
            print(f"  {symbol_key} 在日期范围内有 {len(filtered_timestamps)} 个时间戳")
            all_timestamps.extend(filtered_timestamps)

    # 去重并排序
    unique_timestamps = sorted(list(set(all_timestamps)))
    print(f"总共获取到 {len(unique_timestamps)} 个唯一时间戳")

    # 将微秒时间戳转换为可读日期时间进行展示
    if unique_timestamps:
        first_time = datetime.fromtimestamp(unique_timestamps[0] / 1000000)
        last_time = datetime.fromtimestamp(unique_timestamps[-1] / 1000000)
        print(f"时间戳范围: {first_time} 到 {last_time}")

        # 显示部分样本
        print("时间戳样本:")
        for i in range(min(5, len(unique_timestamps))):
            dt = datetime.fromtimestamp(unique_timestamps[i] / 1000000)
            print(f"  {i + 1}. {dt} ({unique_timestamps[i]})")

    return unique_timestamps


def fetch_for_offline(coins, interval, start_date, end_date, output_dir, library):
    ensure_directory(output_dir)
    for coin in coins:
        symbol = coin
        print(f"\n🚀 开始处理 {symbol} …")

        # 从数据库读取现有数据
        symbol_key = f"{coin}_klines_{interval}"

        # 检查数据覆盖情况，逐条检查连续性
        is_covered, missing_periods, from_storage_df = check_data_coverage(
            library, symbol_key, start_date, end_date, interval
        )
        storage_length = len(from_storage_df)

        if is_covered:
            print("✅ 数据库已有完整连续数据，跳过下载")
            # 可以选择展示一些数据预览
            print("\n现有数据预览:")
            print(from_storage_df.head(5))
            continue

        # 数据不完整或不连续，需要下载
        print("📥 需要下载新数据来填补数据缺口")

        # 下载数据并继续原来的处理逻辑...
        # 如果要针对具体的缺失区间下载，可以遍历 missing_periods
        for start_period, end_period in missing_periods:
            period_start = start_period.strftime("%Y-%m-%d")
            period_end = end_period.strftime("%Y-%m-%d")
            print(f"  下载数据: {period_start} 到 {period_end}")

        if symbol_key in library.list_symbols():
            try:
                # 读取数据
                arctic_result = library.read(symbol_key)
                from_storage_df = arctic_result.data

                # 检查当前索引是什么
                print(f"当前索引类型: {type(from_storage_df.index)}")
                print(f"当前列: {from_storage_df.columns.tolist()}")

                # 检查'Close Time'是否已经是索引
                if "Close Time" in from_storage_df.columns:
                    print("'Close Time'在列中，设置为索引")
                    from_storage_df = from_storage_df.reset_index()
                    from_storage_df.set_index("Close Time", inplace=True)
                else:
                    print("'Close Time'不在列中，可能已经是索引或不存在")

                storage_length = len(from_storage_df)
                print(f"现有数据条数: {storage_length} 行")
            except Exception as e:
                print(f"读取数据时出错: {e}")
                from_storage_df = pd.DataFrame()
                storage_length = 0
        else:
            from_storage_df = pd.DataFrame()
            storage_length = 0
            print(f"数据库中没有 {symbol_key} 的数据")

        # 下载新数据
        new_df = download_and_merge(symbol, interval, start_date, end_date, output_dir)
        if new_df is not None:
            # 设置索引以便合并
            new_df = new_df.reset_index()

            new_df.set_index("Close Time", inplace=True)
            new_length = len(new_df)
            print(f"新下载数据条数: {new_length} 行")

            # 合并数据框，使用索引去重
            if not from_storage_df.empty:
                # 合并并保留所有数据，然后删除重复索引，保留最后一个出现的行
                combined_df = pd.concat([from_storage_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
                combined_length = len(combined_df)

                print(f"合并前总条数: {storage_length + new_length} 行")
                print(f"合并后总条数: {combined_length} 行")
                print(f"重复条数: {storage_length + new_length - combined_length} 行")
            else:
                combined_df = new_df
                combined_length = new_length
                print(f"首次创建数据，总条数: {combined_length} 行")

            # 保存到CSV和数据库
            output_csv = os.path.join(output_dir, f"{symbol_key}.csv")
            combined_df.to_csv(output_csv)
            print(f"✅ 数据已保存到 {output_csv}")

            # 写入数据库
            library.write(f"{symbol_key}", combined_df)
            print(f"✅ 数据已写入数据库 {symbol_key}")

            # 显示合并后的前几行数据
            print("\n合并后数据预览:")
            print(combined_df.head(5))
        else:
            print("⚠️ 没有新数据下载，保持现有数据不变")

    # timestamps = get_all_timestamps_in_range(library, coins, interval, start_date, end_date)
    # return timestamps


if __name__ == "__main__":
    # this will set up the storage using the local file system
    uri = "lmdb://strategies/grid-strategy/data/database_offline/"
    ac = adb.Arctic(uri)

    library = ac.get_library("market", create_if_missing=True)
    print(ac.list_libraries())

    coins = ["BTCUSDT", "DOGEUSDT"]
    interval = "5m"
    start_date = "2025-01-01"
    end_date = "2025-06-01"
    output_dir = "/Users/lizeyu/Desktop/qoc/tmp/raw/5m_klines_raw"

    fetch_for_offline(coins, interval, start_date, end_date, output_dir, library)
