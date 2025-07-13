import os
import sys
from pathlib import Path


def entrypoint() -> Path:
    return Path(sys.argv[0]).absolute()


def data_dir(path: str | os.PathLike[str] = ".") -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return working_dir() / "data" / path


def fig_dir(path: str | os.PathLike[str] = ".") -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return working_dir() / "fig" / path


def working_dir(path: str | os.PathLike[str] = ".") -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return entrypoint().parent / path
