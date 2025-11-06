import enum


class OrderSide(enum.StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class MarginType(enum.StrEnum):
    ISOLATED = "ISOLATED"
    CROSSED = "CROSSED"
