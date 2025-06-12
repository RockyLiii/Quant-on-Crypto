import logging
import os

import numpy as np
import pandas as pd


def has_header(file_path: str) -> bool:
    """Check if CSV file has a header"""
    try:
        with open(file_path) as f:
            first_line = f.readline().strip().split(",")
            # Check if first line contains any non-numeric values (likely headers)
            return any(
                not val.replace(".", "").replace("-", "").isdigit()
                for val in first_line
            )
    except Exception:
        return False


def standardize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize timestamp format to milliseconds"""
    # Check each timestamp individually
    mask = (df["timestamp"] > 2000000000000) & (df["timestamp"] < 2000000000000000)
    if mask.any():
        # Only convert timestamps that are in microseconds
        df.loc[mask, "timestamp"] = (df.loc[mask, "timestamp"] / 1000).astype("int64")
        df.loc[mask, "close_time"] = (df.loc[mask, "close_time"] / 1000).astype("int64")
    return df


def preprocess_data(
    config: dict,
    raw_folder,
    output_folder,
    freq,
    stan_dict: dict,
    logger: logging.Logger,
) -> None:
    """预处理原始数据:
    1. 对齐各币种时间戳
    2. 标准化时间戳 (除以d_t)
    3. 保存处理后的数据

    Args:
        config: 配置字典，包含数据路径和回测参数
        logger: 日志记录器
    """
    d_t = 60000
    start_time = config["backtest"]["start_time"]
    end_time = config["backtest"]["end_time"]

    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    kline_suffix = f"_klines_{freq}.csv"
    logger.info(f"处理K线频率: {freq}")

    # 1. 收集所有币种的时间戳
    logger.info("开始收集时间戳...")
    all_timestamps: list[set[float]] = []
    coin_files = {}

    for filename in os.listdir(raw_folder):
        if not filename.endswith(kline_suffix):
            continue

        coin = filename.split("_")[0]
        file_path = os.path.join(raw_folder, filename)
        coin_files[coin] = file_path

        # 检查是否有表头
        has_header_row = has_header(file_path)
        logger.info(f"{coin}: {'有' if has_header_row else '没有'}表头")

        # 读取该币种的所有时间戳
        timestamps = set()
        with open(file_path) as f:
            if has_header_row:
                next(f)  # Skip header if exists
            for line in f:
                try:
                    timestamp = float(line.strip().split(",")[0])
                    # Standardize timestamp to milliseconds if needed
                    if 2000000000000 < timestamp < 2000000000000000:
                        timestamp /= 1000
                    # Check time range after standardization
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
    logger.info(
        f"标准化后时间戳范围: {min(normalized_timestamps)} 到 {max(normalized_timestamps)}"
    )

    # 3. 处理每个币种的数据
    for coin, input_path in coin_files.items():
        try:
            # 检查文件是否有表头
            has_header_row = has_header(input_path)

            # 读取数据到DataFrame
            df = pd.read_csv(
                input_path, names=columns, header=0 if has_header_row else None
            )

            if df.empty:
                logger.warning(f"{coin} 数据为空，跳过处理")
                continue

            # 标准化时间戳格式（确保使用毫秒）
            df = standardize_timestamp(df)

            # 标准化时间戳除以d_t
            df["timestamp"] = (df["timestamp"] / d_t).astype(int)
            df["close_time"] = (df["close_time"] / d_t).astype(int)

            # 筛选共同时间戳的数据
            df = df[df["timestamp"].isin(normalized_timestamps)]

            if df.empty:
                logger.warning(f"{coin} 筛选后没有数据，跳过处理")
                continue

            # 按时间戳排序
            df = df.sort_values("timestamp")

            # 安全地获取初始收盘价
            try:
                initial_close = float(df["close"].iloc[0])
                logger.info(f"{coin} 初始价格: {initial_close}")
            except (IndexError, ValueError) as e:
                logger.error(f"{coin} 获取初始价格失败: {e}")
                continue

            # Get or set standardization price
            if coin not in stan_dict:
                stan_dict[coin] = initial_close
            else:
                initial_close = stan_dict[coin]
                logger.info(f"使用已存在的标准化价格 {coin}: {initial_close}")

            # 归一化价格数据
            price_columns = ["open", "high", "low", "close"]

            for col in price_columns:
                df[col] = df[col].astype(float) / initial_close

            # 检查数据连续性
            timestamps = df["timestamp"].values
            gaps = np.diff(timestamps)
            if not np.all(gaps == 1):
                logger.warning(f"{coin} 存在时间间隔异常，最大间隔: {gaps.max()}")

            # 保存处理后的数据
            output_path = os.path.join(output_folder, f"{coin}_klines_{freq}.csv")
            df.to_csv(output_path, index=False)

            logger.info(
                f"处理完成 {coin}，保存了 {len(df)} 行数据"
                + ("（覆盖已存在文件）" if os.path.exists(output_path) else "")
            )

        except Exception as e:
            logger.error(f"处理 {coin} 时发生错误: {e!s}")
            continue

    # 4. 数据验证
    logger.info("\n开始验证数据一致性...")
    first_file = True
    reference_timestamps = None

    for filename in os.listdir(output_folder):
        if not filename.endswith(kline_suffix):
            continue

        df = pd.read_csv(os.path.join(output_folder, filename))
        current_timestamps = set(df["timestamp"])

        if first_file:
            reference_timestamps = current_timestamps
            first_file = False
        elif current_timestamps != reference_timestamps:
            logger.error(f"时间戳不一致: {filename}")
            diff = reference_timestamps.symmetric_difference(current_timestamps)
            logger.error(f"差异时间戳: {sorted(diff)[:10]}...")

    logger.info("数据验证完成")
