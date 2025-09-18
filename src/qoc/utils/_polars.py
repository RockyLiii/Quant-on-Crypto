import datetime

import polars as pl


def insert_time(
    df: pl.DataFrame, time: datetime.datetime, column_name: str = "time"
) -> pl.DataFrame:
    if isinstance(time, datetime.datetime):
        time_value = int(time.timestamp() * 1000000)  # 转换为微秒时间戳
    
    return df.insert_column(0, pl.Series(column_name, [time_value]))
