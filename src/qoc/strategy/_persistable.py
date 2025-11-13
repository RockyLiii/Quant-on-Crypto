from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any

import attrs


class PersistableMixin:
    def dump_state(self, file: str | os.PathLike[str] | None = None) -> None:
        file = _as_path(file)
        state: dict[str, Any] = attrs.asdict(
            self, recurse=False, filter=_filter_persist
        )
        with file.open("wb") as fp:
            pickle.dump(state, fp)

    def load_state(self, file: str | os.PathLike[str] | None = None) -> None:
        file = _as_path(file)
        if not file.exists():
            return
        with file.open("rb") as fp:
            state: dict[str, Any] = pickle.load(fp)  # noqa: S301
        for key, value in state.items():
            setattr(self, key, value)


def _as_path(file: str | os.PathLike[str] | None) -> Path:
    return Path(sys.argv[0]).with_suffix(".pickle") if file is None else Path(file)


def _filter_persist[T](attr: attrs.Attribute[T], _value: T) -> bool:
    return attr.metadata.get("persist", True)
