from market_simulator.core.data.dataset import TrainData
from typing import List
from datetime import datetime
import cython
import numpy
from cython.cimports.libc import math


class HistoricalPriceManager:
    def _initialize_indices(self) -> iter:
        total_week_id = 0
        if self.evaluation:
            total_week_id = self.train_data.total_week_id
        else:
            total_week_id = self.train_data.total_week_id - 1

        max_range: cython.int = \
            math.lround(self.curr_num_train_weeks_pct * total_week_id)
        arr = numpy.arange(2, max_range)
        self.rng.shuffle(arr)

        return iter(arr)

    def _get_week_id_curr(self) -> cython.int:

        if self.evaluation:
            return self.num_weeks_train

        else:
            try:
                return int(next(self._train_week_id_indices_))
            except StopIteration:
                # Update new indices list
                del self._train_week_id_indices_
                self._train_week_id_indices_: iter =\
                    self._initialize_indices()

            return int(next(self._train_week_id_indices_))


    def __init__(self, symbols: List[str],
                        num_weeks_train: cython.int,
                        evaluation: int) -> None:

        self.symbols: List[str] = symbols
        self.num_weeks_train: cython.int = num_weeks_train
        self.evaluation: int = evaluation
        self.start_time = datetime.now()
        self.train_data: TrainData = TrainData(symbols=symbols)
        self.curr_num_train_weeks_pct: float = 0.1
        self.rng: numpy.random.Generator = numpy.random.default_rng()

        self._train_week_id_indices_: iter = \
            self._initialize_indices()

        self._week_id_curr_: cython.int = self._get_week_id_curr()
        self._trade_time_idx_curr_: cython.int = 0

    def market_obs(self):
        market_state = self.train_data.get_market_state(
            self._week_id_curr_, self._trade_time_idx_curr_
        )
        self._trade_time_idx_curr_ += 1

        return market_state

    @property
    def performance(self):
        return self.train_data.get_week_performance(self._week_id_curr_)

    def reset(self, num_train_weeks_pct: float):
        if num_train_weeks_pct != self.curr_num_train_weeks_pct:
            del self._train_week_id_indices_
            self.curr_num_train_weeks_pct = num_train_weeks_pct
            self._train_week_id_indices_: iter =\
                self._initialize_indices()

        self._week_id_curr_ = self._get_week_id_curr()
        self._trade_time_idx_curr_ = 0

        return self.market_obs()
