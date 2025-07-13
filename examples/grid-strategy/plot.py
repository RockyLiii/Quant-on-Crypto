from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pydantic
import seaborn as sns

import qoc


class Config(pydantic.BaseModel):
    db: str = qoc.data_dir("database").as_uri().replace("file://", "lmdb://")
    fig_dir: Path = qoc.fig_dir()

    balances: list[str] = ["PENGU", "USDT"]
    quotes: list[str] = ["USDT"]


def main(cfg: Config) -> None:
    plt.rc("figure", dpi=300)

    db = qoc.Database(cfg.db)
    library: qoc.Library = db.get_library("balance")
    data: pd.DataFrame = library.read("quote")
    for quote in cfg.quotes:
        lineplot(data, x="time", y=quote, file=cfg.fig_dir / "quote" / f"{quote}.png")
    data = library.read("balance")
    for balance in cfg.balances:
        lineplot(
            data, x="time", y=balance, file=cfg.fig_dir / "balance" / f"{balance}.png"
        )


def lineplot(data: pd.DataFrame, x: str, y: str, *, file: Path) -> None:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x=x, y=y)
    plt.xticks(rotation=45)
    plt.tight_layout()
    file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(file)
    plt.close()


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
