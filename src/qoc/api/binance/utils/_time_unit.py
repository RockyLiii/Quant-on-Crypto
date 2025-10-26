import functools
from typing import Any

import requests
from binance.spot import Spot
from binance_sdk_derivatives_trading_usds_futures.rest_api import (
    DerivativesTradingUsdsFuturesRestAPI,
)

import qoc.time_utils as tu


@functools.singledispatch
def get_time_unit(_client: Any) -> tu.TimeUnit:
    """.

    All time and timestamp related fields in the JSON responses are in **milliseconds by default**.

    References:
        1. <https://developers.binance.com/docs/binance-spot-api-docs/testnet/rest-api/general-api-information>
    """
    return tu.TimeUnit.MILLISECOND


@get_time_unit.register(requests.Session)
def _get_time_unit_session(session: requests.Session) -> tu.TimeUnit:
    return tu.TimeUnit(session.headers.get("X-MBX-TIME-UNIT", tu.TimeUnit.MILLISECOND))


@get_time_unit.register(Spot)
def _get_time_unit_spot(client: Spot) -> tu.TimeUnit:
    return get_time_unit(client.session)


@get_time_unit.register(DerivativesTradingUsdsFuturesRestAPI)
def _get_time_unit_derivatives_trading_usds_futures(
    client: DerivativesTradingUsdsFuturesRestAPI,
) -> tu.TimeUnit:
    return get_time_unit(client._session)  # noqa: SLF001
