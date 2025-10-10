from typing import Any

import cachetools
import httpx
import pydantic


class GeoIPLocation(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    country: str
    country_code: str


class GeoIP(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    ip: pydantic.IPvAnyAddress
    location: GeoIPLocation


@cachetools.cached(cache=cachetools.TTLCache(maxsize=1, ttl=3600))
def geoip() -> GeoIP:
    response: httpx.Response = httpx.get("https://api.ipapi.is")
    raw: Any = response.json()
    data: GeoIP = GeoIP.model_validate(raw)
    return data
