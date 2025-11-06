from typing import Any

import pydantic
from pydantic_extra_types.pendulum_dt import DateTime

from qoc.typing import AssetName, SymbolName

from ._base import BaseModel
from ._enum import PositionSide
from ._mixin import DictMixin


class AccountAsset(BaseModel):
    asset: AssetName
    wallet_balance: float
    unrealized_profit: float
    margin_balance: float
    available_balance: float
    update_time: DateTime


class AccountAssetsDict(
    DictMixin[AssetName, AccountAsset],
    pydantic.RootModel[dict[AssetName, AccountAsset]],
):
    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_dict(cls, value: Any) -> dict[AssetName, dict[str, Any]]:
        if isinstance(value, list):
            return {item["asset"]: item for item in value}
        return value


class AccountPosition(BaseModel):
    symbol: SymbolName
    position_side: PositionSide
    position_amt: float
    unrealized_profit: float
    isolated_margin: float
    notional: float
    isolated_wallet: float
    update_time: DateTime


class AccountPositionsDict(
    DictMixin[AssetName, AccountPosition],
    pydantic.RootModel[dict[AssetName, AccountPosition]],
):
    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_dict(cls, value: Any) -> dict[AssetName, dict[str, Any]]:
        if isinstance(value, list):
            return {item["symbol"]: item for item in value}
        return value


class Account(BaseModel):
    assets: AccountAssetsDict
    positions: AccountPositionsDict
