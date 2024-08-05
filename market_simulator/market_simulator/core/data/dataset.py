from typing import List, Tuple
from market_simulator.core.data_feed import fetch_data
from market_simulator.typing import AssetsMarketPrice
from scipy.stats import zscore

import cython
import pandas
import numpy

@cython.dataclasses.dataclass
@cython.cclass
class MarketState:
        trading_time: cython.py_int
        obs: numpy.ndarray
        target_price: AssetsMarketPrice
        done: bool


class TrainData:

    def _generate_market_obs(self) -> numpy.ndarray:
        market_data_continuous = []  # OHLC

        market_data_continuous += [
            "basis",
            "mvhr",
            "delta",
            "vega",
        ]
        return self._dataset_[market_data_continuous].to_numpy(copy=True)

    def _generate_filter_index_with_weekly_perf(self):
        week_ids = self._dataset_["week_id"].unique()
        week_id_by_idx = []
        weekly_performance: List[float] = []

        for w_id in week_ids:
            df = self._dataset_.query(f"week_id == {w_id}")
            
            def stat_perf():
                perf_stat: List[float] = []
                for symbol in self.symbols:
                    a = df[f'close_{symbol}'].iloc[-1]
                    b = df[f'open_{symbol}'].iloc[0]
                    perf = (a - b)/b
                    perf_stat.append(max(perf, 0.01))

                return max(perf_stat)

            w_id_idx = df["week_id"]
            start_end = (w_id_idx.index[0], w_id_idx.index[-1])
            week_perf = stat_perf()
            weekly_performance.append(week_perf)
            week_id_by_idx.append(start_end)

        return week_id_by_idx, weekly_performance

    def _target_price_series(self):
        return [
            self._dataset_[f"close_{self.symbols[0]}"].shift(-1).fillna(-99),
            self._dataset_[f"close_{self.symbols[1]}"].shift(-1).fillna(-99),
        ]

    def _trading_time_series(self):
        return self._dataset_["open_time"]

    def __init__(self, symbols: List[str]) -> None:
        self.symbols = symbols

        self._dataset_: pandas.DataFrame = fetch_data.generate_dataset(self.symbols)
        self._target_price_series_: List[pandas.Series] = self._target_price_series()
        self._trading_time_: pandas.Series = self._trading_time_series()
        
        self._week_id_by_idx_: List[Tuple[int, int]] = []
        self._weekly_performance_: List[int] = []
        self._week_id_by_idx_, self._weekly_performance_ =\
            self._generate_filter_index_with_weekly_perf()
        self._market_obs_: numpy.ndarray = self._generate_market_obs()

    def get_market_state(self,
                         week_id: cython.int,
                         trade_time_id: cython.int) -> MarketState:
        start_last_2, _ = self._week_id_by_idx_[week_id - 2]
        start_curr, end_curr = self._week_id_by_idx_[week_id]

        # Index start from zero
        done = False
        _cursor_ = start_curr + trade_time_id
        if _cursor_ == end_curr:
            done = True

        obs: numpy.ndarray = zscore(self._market_obs_[start_last_2:_cursor_])[-1]

        _targets_price_curr_: AssetsMarketPrice = {
            self.symbols[0]: self._target_price_series_[0][_cursor_],
            self.symbols[1]: self._target_price_series_[1][_cursor_],
        }

        _trading_time_curr_: int = int(self._trading_time_[_cursor_])

        return MarketState(
            trading_time=_trading_time_curr_,
            obs=obs,
            target_price=_targets_price_curr_,
            done=done
        )

    @property
    def total_week_id(self):
        return len(self._week_id_by_idx_)
    
    def get_week_performance(self, week_id: int):
        return self._weekly_performance_[week_id]
    
    def refresh(self):
        del self._dataset_
        del self._target_price_series_
        del self._trading_time_
        del self._week_id_by_idx_
        del self._weekly_performance_
        del self._market_obs_

        self._dataset_: pandas.DataFrame = fetch_data.generate_dataset(self.symbols)
        self._target_price_series_: List[pandas.Series] = self._target_price_series()
        self._trading_time_: pandas.Series = self._trading_time_series()
        self._week_id_by_idx_, self._weekly_performance_ =\
            self._generate_filter_index_with_weekly_perf()
        self._market_obs_: numpy.ndarray = self._generate_market_obs()