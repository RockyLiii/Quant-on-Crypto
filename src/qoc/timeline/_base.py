import logging
from collections import defaultdict, deque

import torch

logger = logging.getLogger(__name__)


class BaseTimeline:
    starttime: float
    """回测开始时间"""

    endtime: float
    """回测结束时间"""

    current_timestamp: int | None = None
    """当前时间戳"""

    inittime: int | None = None
    """初始化时间"""

    initial_capital: float
    """初始资金"""

    trading_fee: float
    """交易手续费率"""

    tradeamount: float
    """总交易金额"""

    coins: list[str]
    """交易币种列表"""

    data: dict[str, torch.Tensor]
    """K线数据 [timestamp, open, high, low, close, ...]"""

    positions: dict[str, torch.Tensor]
    """持仓信息 [timestamp, quantity, avg_price]"""

    trading_records: dict[str, torch.Tensor]
    """交易记录 [timestamp, price, quantity, fee]"""

    capital: torch.Tensor
    """资金曲线"""

    total_value: torch.Tensor
    """总资产曲线"""

    signal_count: dict[str, int]
    """信号计数 [coin: count]"""

    feature_configs: dict[str, dict[str, dict[str, int]]]
    """特征配置"""

    coin_features: dict[str, dict[str, deque]]
    """币种特征数据"""

    global_features: dict[str, dict[str, deque]]
    """全局特征数据"""

    cor_r: float
    cor_p: float

    _capital: float
    """当前可用资金"""

    _total_value: float
    """当前总资产"""

    _bond: float
    """空仓保证金"""

    def __init__(
        self,
        starttime: float,
        endtime: float,
        feature_configs: dict[str, dict[str, dict[str, int]]] | None = None,
        trading_fee: float = 0.001,
        initial_capital: float = 10000,
    ) -> None:
        """初始化时间轴."""
        # 1. 基础时间参数
        self.starttime = starttime  # 回测开始时间
        self.endtime = endtime  # 回测结束时间
        self.current_timestamp = None  # 当前时间戳
        self.inittime = None  # 初始化时间

        # 2. 账户参数
        self.initial_capital = initial_capital  # 初始资金
        self.trading_fee = trading_fee  # 交易手续费率
        self._capital = initial_capital  # 当前可用资金
        self._total_value = initial_capital  # 当前总资产
        self._bond = 0  # 空仓保证金
        self.tradeamount = 0  # 总交易金额

        # 3. 交易品种
        self.coins = []  # 交易币种列表

        # 4. 市场数据存储
        self.data = {
            "total": torch.empty(
                (0, 12), dtype=torch.float32
            )  # K线数据 [timestamp, open, high, low, close, ...]
        }

        # 5. 仓位和交易记录
        self.positions = {
            "total": torch.empty(
                (0, 3), dtype=torch.float32
            )  # 持仓信息 [timestamp, quantity, avg_price]
        }
        self.trading_records = {
            "total": torch.empty(
                (0, 4), dtype=torch.float32
            )  # 交易记录 [timestamp, price, quantity, fee]
        }

        # 6. 资金记录
        self.capital = torch.tensor(
            [[starttime, initial_capital]], dtype=torch.float32
        )  # 资金曲线
        self.total_value = torch.tensor(
            [[starttime, initial_capital]], dtype=torch.float32
        )  # 总资产曲线

        # 7. 交易统计
        self.signal_count = {"total": 0}  # 信号计数 [coin: count]

        # 8. 特征配置和存储
        self.feature_configs = feature_configs or {
            "coin_features": {},  # 币种特征配置
            "global_features": {},  # 全局特征配置
        }

        # 9. 特征数据存储
        self.coin_features = {}  # 币种特征数据
        self.global_features = {  # 全局特征数据
            feature_name: {"data": deque(maxlen=config["window"])}
            for feature_name, config in self.feature_configs["global_features"].items()
        }

        self.cor_r = 1
        self.cor_p = 1

    def feature_calc(self) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        """基类特征计算框架.

        Returns:
            Dict: {
                'coin_features': {
                    feature_name: {coin: tensor},
                    ...
                },
                'global_features': {
                    feature_name: tensor,
                    ...
                }
            }
        """
        raise NotImplementedError

    def _initialize_for_coin(self, coinname: str) -> None:
        """为新币种初始化数据结构."""
        if coinname not in self.coins:
            self.coins.append(coinname)

            # Initialize with empty tensors
            self.data[coinname] = torch.empty((0, 12), dtype=torch.float32)
            self.positions[coinname] = torch.empty((0, 3), dtype=torch.float32)
            self.trading_records[coinname] = torch.empty((0, 4), dtype=torch.float32)

            self.signal_count[coinname] = 0

            # Initialize features
            for feature_name, config in self.feature_configs["coin_features"].items():
                if feature_name not in self.coin_features:
                    self.coin_features[feature_name] = {
                        "data": defaultdict(lambda: deque(maxlen=config["window"]))
                    }
                self.coin_features[feature_name]["data"][coinname] = deque(
                    maxlen=config["window"]
                )

    def time_pass(self, new_data: dict[str, torch.Tensor]) -> None:
        """时间推进, 更新数据并计算特征."""
        self.inittime = (
            self.get_current_timestamp()
            if self.inittime is None or self.inittime == 0
            else self.inittime
        )

        for coinname, new_row in new_data.items():
            self.data[coinname] = new_row

        for coinname in new_data:
            if coinname not in self.coins:
                self._initialize_for_coin(coinname)
                self.current_timestamp = new_data[coinname][0, 0].item()

        self.current_timestamp = self.get_current_timestamp()

        self.feature_calc()

        # Update records
        totals = (0.0, 0.0)  # quantity, value
        for coin in self.coins:
            coin_results = self._update_records(coin)
            totals = tuple(t + r for t, r in zip(totals, coin_results, strict=False))

        # Update total records and capital
        self._update_total_records(totals)
        self._update_capital_and_value(totals[1])

    def get_current_timestamp(self) -> int:
        """获取当前时间戳."""
        for coin in self.coins:
            if coin in self.data and self.data[coin].size(0) > 0:
                return int(self.data[coin][-1, 0].item())
        return 0

    def _update_records(self, coin: str) -> tuple[float, float]:
        """更新各类记录并返回统计值.

        Returns:
            tuple: (quantity, value)
        """
        if self.data[coin].size(0) == 0:
            return 0.0, 0.0

        current_price = self.data[coin][-1, 4].item()

        # Calculate profits
        qty = (
            self.positions[coin][-1, 1].item()
            if self.positions[coin].size(0) > 0
            else 0.0
        )
        avg_price = (
            self.positions[coin][-1, 2].item()
            if self.positions[coin].size(0) > 0
            else 0.0
        )

        # Update total profit

        # Calculate position value
        value = qty * current_price  # ❓关于做空

        return qty, value

    def _update_total_records(self, totals: tuple[float, float]) -> None:
        """更新汇总记录.

        Args:
            totals: (total_quantity, total_value)
        """
        qty, value = totals

        # Update total position
        total_position = torch.tensor(
            [[self.current_timestamp, qty, value]], dtype=torch.float32
        )

        self.positions["total"] = torch.cat(
            (self.positions["total"], total_position), dim=0
        )

    def _update_capital_and_value(self, total_position_value: float) -> None:
        """更新可用资金和总资产价值.

        Args:
            total_position_value: 所有持仓的当前市值
        """
        capital_record = torch.tensor(
            [[self.current_timestamp, self._capital]], dtype=torch.float32
        )

        self._total_value = self._capital + total_position_value + 2 * self._bond

        value_record = torch.tensor(
            [[self.current_timestamp, self._total_value]], dtype=torch.float32
        )

        self.capital = torch.cat((self.capital, capital_record), dim=0)
        self.total_value = torch.cat((self.total_value, value_record), dim=0)

    def trade(self, coinname: str, quantity: float) -> None:
        """执行交易操作, 处理开仓、加仓、减仓及手续费."""
        if coinname not in self.coins:
            return

        if quantity != 0:
            self.signal_count[coinname] += 1
            self.signal_count["total"] += 1

        # 获取当前持仓状态
        if self.positions[coinname].size(0) == 0:
            current_qty = 0.0
            current_avg = 0.0
        else:
            current_qty = self.positions[coinname][
                -1, 1
            ].item()  # quantity is in column 1
            current_avg = self.positions[coinname][
                -1, 2
            ].item()  # avg_price is in column 2

        # 计算交易金额和手续费
        trade_price = self.data[coinname][-1, 4].item()  # Use closing price
        transaction_amount = trade_price * abs(quantity)
        fee_cost = transaction_amount * self.trading_fee

        self._capital -= fee_cost
        self.tradeamount += transaction_amount

        # Record trade with fee
        record = torch.tensor(
            [[self.current_timestamp, trade_price, quantity, fee_cost]],
            dtype=torch.float32,
        )
        self.trading_records[coinname] = torch.cat(
            (self.trading_records[coinname], record), dim=0
        )

        aver_price = current_avg

        # 新开仓
        if current_qty == 0:
            aver_price = trade_price
            self._capital -= transaction_amount

            if quantity < 0:
                self._bond += transaction_amount  # 空仓考虑1x保证金

        # 加仓
        elif current_qty * quantity > 0:
            aver_price = (
                current_avg * abs(current_qty) + trade_price * abs(quantity)
            ) / (abs(current_qty) + abs(quantity))
            self._capital -= transaction_amount
            if quantity < 0:
                self._bond += transaction_amount  # 空仓考虑1x保证金

        # 减仓
        elif current_qty * quantity < 0:
            remaining_qty = abs(quantity) - abs(current_qty)

            # 多空反转
            if remaining_qty > 0:
                if quantity < 0:
                    aver_price = trade_price
                    self._capital += trade_price * abs(current_qty)
                    self._capital -= trade_price * remaining_qty
                    self._bond += trade_price * abs(remaining_qty)  # 空仓保证金

                if quantity > 0:
                    aver_price = trade_price
                    self._capital -= trade_price * abs(remaining_qty)

                    self._capital += (2 * current_avg - trade_price) * abs(current_qty)
                    self._bond -= current_avg * abs(current_qty)  # 空仓保证金

            # 多空不反转
            else:
                if quantity < 0:
                    aver_price = current_avg
                    self._capital += trade_price * abs(quantity)

                if quantity > 0:
                    aver_price = current_avg
                    self._capital -= trade_price * abs(quantity)
                    self._capital += 2 * current_avg * abs(quantity)
                    self._bond -= current_avg * abs(quantity)  # 空仓保证金
        position_record = torch.tensor(
            [[self.current_timestamp, current_qty + quantity, aver_price]],
            dtype=torch.float32,
        )
        self.positions[coinname] = torch.cat(
            (self.positions[coinname], position_record), dim=0
        )
