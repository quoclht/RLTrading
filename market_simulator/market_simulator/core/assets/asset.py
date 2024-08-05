import cython

from market_simulator.common import round_size
from cython.cimports.libc import math

@cython.cclass
class Asset:
    cython.declare(symbol=cython.basestring,
                   _realized_pnl_=cython.float,
                   _size_=cython.float,
                   _avg_price_=cython.float,
                   _market_price_=cython.float)


    def _action_type(self,
                     new_size: cython.float,
                     size_change: cython.float) -> cython.basestring:

        # Current Long, new Long
        if new_size > 0 and self._size_ > 0:
            if size_change > 0:
                return "INCREASE"
            elif size_change < 0:
                return "DECREASE"

        # Current Short, new Short
        if new_size < 0 and self._size_ < 0:
            if size_change < 0:
                return "INCREASE"
            elif size_change > 0:
                return "DECREASE"

        if new_size < 0 and self._size_ > 0:
            return "CLOSE_OPEN_NEW"
        elif new_size > 0 and self._size_ < 0:
            return "CLOSE_OPEN_NEW"

    def __init__(self, symbol: cython.basestring):
        self.symbol = symbol

        # Internal state: Must be reset after every episode
        self._realized_pnl_: cython.float = 0
        self._size_: cython.float = 0
        self._avg_price_: cython.float = 0
        self._market_price_: cython.float = 0

    @property
    def market_value(self) -> cython.float:
        return self.market_price * math.fabsf(self.size)

    def can_rebalance(self, new_size: cython.float) -> cython.int:
        size_change: cython.float = new_size - self.size
        size_change = round_size.size(
            symbol=self.symbol, size=size_change, is_tick=0
        )

        if size_change != 0:
            return 1
        return 0

    def update(self, new_size: cython.float, price: cython.float):

        self._market_price_ = price
        trade_return: cython.float = 0
        size_change: cython.float = 0
        action_type: cython.basestring = "HOLD"

        if new_size != 0:
            size_change = new_size - self.size
            size_change = round_size.size(
                symbol=self.symbol, size=size_change, is_tick=0
            )

            fee: cython.float = (math.fabsf(size_change) * price) * 0.0005
            action_type: cython.basestring = self._action_type(new_size, size_change)
            if size_change != 0:
                trade_return = -fee

                if self.size == 0:
                    action_type = "OPEN_NEW"
                    self._avg_price_ = price
                else:
                    if action_type == "INCREASE":
                        total_value: cython.float = self.size * self.avg_price +\
                                                       size_change * price
                        self._avg_price_ = total_value / (self.size + size_change)
                    elif action_type == "DECREASE":
                        trade_return += (price - self.avg_price) * new_size
                        self._realized_pnl_ += trade_return
                    elif action_type == "CLOSE_OPEN_NEW":
                        trade_return += (price - self.avg_price) * self.size
                        self._realized_pnl_ += trade_return
                        self._avg_price_ = price

                    else:
                        raise Exception(
                            f"Out of expected cases, \
                                {self.avg_price, self.size, new_size, size_change}"
                        )

                self._size_ = new_size

        return trade_return, f"{action_type}_{round(size_change, 5)}"

    def reset(self):
        self._realized_pnl_ = 0
        self._size_ = 0
        self._avg_price_ = 0
        self._market_price_ = 0

    @property
    def market_price(self) -> cython.float:
        return self._market_price_

    @property
    def avg_price(self) -> cython.float:
        return self._avg_price_

    @property
    def size(self) -> cython.float:
        return self._size_

    @property
    def unrealized_pnl(self) -> cython.float:

        if self.market_price > 0:
            _unrealized_pnl_: cython.float =\
                (self.market_price - self.avg_price) * self.size
            return _unrealized_pnl_

        else:
            return 0

    @property
    def realized_pnl(self) -> cython.float:
        return self._realized_pnl_

    @property
    def total_pnl(self) -> cython.float:
        _pnl_: cython.float = self.realized_pnl + self.unrealized_pnl
        return _pnl_