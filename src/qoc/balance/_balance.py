import collections
from datetime import datetime

import attrs
import polars as pl

from qoc import api as _api
from qoc import database, market, utils


@attrs.define
class Balance:
    library: database.Library = attrs.field()
    symbols: list[str] = attrs.field(factory=list)

    def step(self, api: _api.ApiBinance|_api.ApiOffline, market: market.Market, now: datetime) -> None:
        account: _api.Account = api.account()
        balances_dict: dict[str, float] = {
            b.asset: b.free + b.locked for b in account.balances
        }
        balances_df: pl.DataFrame = pl.from_dicts([balances_dict])
        balances_df = utils.insert_time(balances_df, now)
        self.library.append("balance", balances_df)

        # quote_dict: dict[str, float] = collections.defaultdict(lambda: 0.0)
        # exchange_info: _api.ExchangeInfo = api.exchange_info(symbols=self.symbols)
        # for symbol in self.symbols:
        #     info: _api.ExchangeInfoSymbol = exchange_info.get_symbol(symbol)
        #     quote_dict[info.quote_asset] += market.convert(
        #         qty=balances_dict[info.base_asset],
        #         base=info.base_asset,
        #         quote=info.quote_asset,
        #     )
        # quote_df: pl.DataFrame = pl.from_dicts([quote_dict])
        # quote_df = utils.insert_time(quote_df, now)
        # self.library.append("quote", quote_df)

    def step_offline(self, market, library, coins, interval, now) -> None:
        pass
        # now = datetime.fromtimestamp(now/1000000)

        # account: _api.Account = api.account()
        # balances_dict: dict[str, float] = {
        #     b.asset: b.free + b.locked for b in account.balances
        # }
        # balances_df: pl.DataFrame = pl.from_dicts([balances_dict])
        # balances_df = utils.insert_time(balances_df, now)
        # self.library.append("balance", balances_df)

        # quote_dict: dict[str, float] = collections.defaultdict(lambda: 0.0)
        # exchange_info: _api.ExchangeInfo = api.exchange_info(symbols=self.symbols)
        # for symbol in self.symbols:
        #     info: _api.ExchangeInfoSymbol = exchange_info.get_symbol(symbol)
        #     quote_dict[info.quote_asset] += market.convert(
        #         qty=balances_dict[info.base_asset],
        #         base=info.base_asset,
        #         quote=info.quote_asset,
        #     )
        # quote_df: pl.DataFrame = pl.from_dicts([quote_dict])
        # quote_df = utils.insert_time(quote_df, now)
        # self.library.append("quote", quote_df)
