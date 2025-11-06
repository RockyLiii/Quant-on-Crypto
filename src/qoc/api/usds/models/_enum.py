import enum


class OrderSide(enum.StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(enum.StrEnum):
    BOTH = "BOTH"
    LONG = "LONG"
    SHORT = "SHORT"


class MarginType(enum.StrEnum):
    ISOLATED = "ISOLATED"
    CROSSED = "CROSSED"
