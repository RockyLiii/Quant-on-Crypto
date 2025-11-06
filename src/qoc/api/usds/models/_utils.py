from typing import Any

import pydantic


def get(obj: Any, key: str) -> Any:
    if isinstance(obj, pydantic.BaseModel):
        return getattr(obj, key)
    return obj[key]
