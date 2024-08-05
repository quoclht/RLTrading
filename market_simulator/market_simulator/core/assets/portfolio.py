import cython

from typing import Dict, List
from market_simulator.core.assets.asset import Asset
from market_simulator.typing import (
    ActionType,
    AssetsAvgPrice,
    AssetsMarketPrice,
    AssetsPnL,
    AvaiableAction,
    PortfolioTradeReturn,
    SizeOfAssets,
    ExecuteResult
)
from market_simulator.constant import ACTION
from market_simulator.common import round_size

@cython.dataclasses.dataclass
@cython.cclass
class PortfolioInfo:
        portfolio_value: cython.float
        assets_pnl: AssetsPnL
        changed_value: cython.float

class PortfolioManager:

    def __init__(self, symbols: List[str], initial_equity: cython.float):
        self.symbols: List[str] = symbols
        self.initial_equity: cython.float = initial_equity

        # Internal state
        self._assets_: Dict[str, Asset] = {s: Asset(symbol=s) for s in self.symbols}
        self._market_price_: Dict[str, cython.float] = {s: 0 for s in self.symbols}
        self._previous_portfolio_value_ = self.initial_equity

    @property
    def avg_prices(self) -> AssetsAvgPrice:
        _avg_prices_ = {}
        for a in self._assets_:
            _avg_prices_[a] = round_size.round_number(self._assets_[a].avg_price)

        return _avg_prices_

    @property
    def sizes(self) -> SizeOfAssets:
        _sizes_ = {}
        for a in self._assets_:
            _sizes_[a] = round_size.round_number(self._assets_[a].size)

        return _sizes_

    @property
    def market_prices(self) -> AssetsMarketPrice:
        return self._market_price_

    @property
    def portfolio_info(self) -> PortfolioInfo:
        assets_pnl: AssetsPnL = {}

        for s in self.symbols:
            assets_pnl[s] = round_size.round_number(self._assets_[s].total_pnl)

        # TODO: Support cash in to account after round size of order
        portfolio_value: cython.float = self.initial_equity +\
                                           sum(assets_pnl.values())
        diff_portfolio_value: cython.float = portfolio_value -\
                                                self._previous_portfolio_value_

        return PortfolioInfo(
            portfolio_value=portfolio_value,
            assets_pnl=assets_pnl,
            changed_value=round_size.round_number(diff_portfolio_value),
        )

    def avaiable_action(self, action_id: cython.int) -> AvaiableAction:
        orders_size: Dict[str, cython.float] = {
            self.symbols[0]: 0,
            self.symbols[1]: 0,
        }
        can_rebalance: Dict[str, cython.int] = {
            self.symbols[0]: 0,
            self.symbols[1]: 0,
        }

        ratio = ACTION[action_id]
        for idx, s in enumerate(self.symbols):
            order_value: cython.float = ratio[idx] *\
                                           self.portfolio_info.portfolio_value
            order_value = round_size.size(s, order_value, 1)
            order_size: cython.float = order_value / self.market_prices[s]

            if action_id == 0:
                order_size = 0

            orders_size[s] = round_size.size(s, order_size, 0)
            can_rebalance[s] = self._assets_[s].can_rebalance(order_size)

        if sum(can_rebalance.values()) == 2:
            return True, orders_size
        else:
            return False, orders_size

    def update_market_price(self, market_price: AssetsMarketPrice):
        self._previous_portfolio_value_ = self.portfolio_info.portfolio_value
        self._market_price_ = market_price

    def execute_order(self, action_id: cython.int) -> ExecuteResult:
        
        """action: Allocate asset in percentage"""

        trades_return: PortfolioTradeReturn = {
            self.symbols[0]: 0,
            self.symbols[1]: 0,
        }
        actions_type: ActionType = {
            self.symbols[0]: "",
            self.symbols[1]: "",
        }
        execute_result: ExecuteResult = ()

        can_rebalance, orders_size = self.avaiable_action(action_id=action_id)
        for s in self.symbols:
            order_size = orders_size[s] if can_rebalance is True else 0
            trades_return[s], actions_type[s] = self._assets_[s].update(
                order_size, self.market_prices[s]
            )
            trades_return[s] = round_size.round_number(trades_return[s])
        
        execute_result = (trades_return, actions_type)
        return execute_result

    def reset(self):
        self._previous_portfolio_value_ = self.initial_equity

        for s in self.symbols:
            self._assets_[s].reset()
            self._market_price_[s] = 0

        return self.portfolio_info
