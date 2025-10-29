from binance.spot import Spot
from binance_sdk_derivatives_trading_usds_futures.rest_api import (
    DerivativesTradingUsdsFuturesRestAPI,
)

type RestApiLike = Spot | DerivativesTradingUsdsFuturesRestAPI
