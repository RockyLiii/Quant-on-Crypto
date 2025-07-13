import enum
from typing import Any, override


class CaseInsensitiveEnum(enum.StrEnum):
    @override
    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        return name.upper()

    @override
    @classmethod
    def _missing_(cls, value: object) -> Any:
        value = value.lower() if isinstance(value, str) else value
        for member in cls:
            if value in (member, member.lower()):
                return member
        return super()._missing_(value)
