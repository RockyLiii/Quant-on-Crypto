import abc


class TradeApi(abc.ABC):
    @abc.abstractmethod
    def ping(self) -> None:
        raise NotImplementedError


class Trade(abc.ABC):
    @property
    @abc.abstractmethod
    def api(self) -> TradeApi:
        raise NotImplementedError
