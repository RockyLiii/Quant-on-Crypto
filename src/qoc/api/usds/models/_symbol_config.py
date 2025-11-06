from typing import Any

import pydantic

from qoc.typing import SymbolName

from ._base import BaseModel
from ._enum import MarginType
from ._mixin import DictMixin
from ._utils import get


class SymbolConfig(BaseModel):
    symbol: SymbolName
    margin_type: MarginType
    leverage: int


class SymbolConfigDict(
    DictMixin[SymbolName, SymbolConfig],
    pydantic.RootModel[dict[SymbolName, SymbolConfig]],
):
    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_root(
        cls, value: list[dict[str, Any]]
    ) -> dict[SymbolName, dict[str, Any]]:
        if not isinstance(value, list):
            raise ValueError  # noqa: TRY004
        return {get(item, "symbol"): item for item in value}
