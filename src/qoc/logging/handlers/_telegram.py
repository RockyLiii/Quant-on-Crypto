import logging
from typing import override

import telegram
from environs import env

from ._async import AsyncHandler


class TelegramHandler(AsyncHandler):
    bot: telegram.Bot
    chat_id: int | str

    def __init__(
        self,
        bot: telegram.Bot | None = None,
        chat_id: int | str | None = None,
        level: int | str = logging.NOTSET,
    ) -> None:
        super().__init__(level)
        if bot is None:
            bot = telegram.Bot(token=env.str("TELEGRAM_BOT_TOKEN"))
        if chat_id is None:
            chat_id = env.str("TELEGRAM_CHAT_ID")
        self.bot = bot
        self.chat_id = chat_id

    @override
    async def _emit(self, record: logging.LogRecord) -> None:
        message: str = self.format(record)
        message = message[-telegram.constants.MessageLimit.MAX_TEXT_LENGTH :]
        await self.bot.send_message(self.chat_id, message)
