import abc
import asyncio
import logging
import queue
import threading
import traceback


class AsyncHandler(abc.ABC, logging.Handler):
    _queue: queue.Queue[logging.LogRecord]
    _stop_event: threading.Event
    _thread: threading.Thread

    def __init__(self, level: int | str = logging.NOTSET) -> None:
        super().__init__(level)
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join()
        super().close()

    def emit(self, record: logging.LogRecord) -> None:
        self._queue.put(record)

    @abc.abstractmethod
    async def _emit(self, record: logging.LogRecord) -> None:
        raise NotImplementedError

    async def _daemon(self) -> None:
        while not self._stop_event.is_set():
            try:
                record: logging.LogRecord = self._queue.get(timeout=0.1)
                await self._emit(record)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def _loop(self) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._daemon())
        loop.close()
