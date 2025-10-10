import binance.spot

import qoc.time_utils as tu


def get_time_unit(client: binance.spot.Spot) -> tu.TimeUnit:
    """.

    All time and timestamp related fields in the JSON responses are in **milliseconds by default**.

    References:
        1. <https://developers.binance.com/docs/binance-spot-api-docs/testnet/rest-api/general-api-information>
    """
    return tu.TimeUnit(
        client.session.headers.get("X-MBX-TIME-UNIT", tu.TimeUnit.MILLISECOND)
    )
