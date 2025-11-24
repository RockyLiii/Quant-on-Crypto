import logging

from liblaf import grapes

from .handlers import TelegramHandler


def init() -> None:
    grapes.logging.init()
    handler = TelegramHandler(level=logging.WARNING)
    handler.name = "telegram"
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        )
    )
    logging.root.addHandler(handler)
