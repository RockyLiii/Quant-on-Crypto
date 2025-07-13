import datetime

import polars as pl


def insert_time(
    df: pl.DataFrame, time: datetime.datetime, column_name: str = "time"
) -> pl.DataFrame:
    return df.insert_column(0, pl.Series(column_name, [time]))
