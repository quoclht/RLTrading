import numpy
import pandas
import gym
import gc
import cython

from market_simulator.core.data.dataset import MarketState
from market_simulator.core.assets.portfolio import PortfolioManager
from market_simulator.core.data.market_price import HistoricalPriceManager
from market_simulator.constant import ACTION, NUM_ACTIONS
from market_simulator.typing import ExecuteResult
from market_simulator.common import round_size

from datetime import datetime
from gym import spaces
from typing import Dict, List
from scipy.stats import zscore

@cython.dataclasses.dataclass
@cython.cclass
class SimulatorConfigurations:
        symbols: List[str]
        initial_equity: cython.float
        num_weeks_train: cython.int
        evaluation: cython.int
        collect_step_detail: cython.int

class Simulator(gym.Env):


    def _update_state(
        self,
        action: cython.int,
        execute_result: ExecuteResult,
        reward: cython.float,
        predicted_reward: cython.float,
    ):
        trades_return = list(execute_result[0].values())
        actions_type = list(execute_result[1].values())

        portfolio_info = self._portfolio_manager_.portfolio_info
        portfolio_value = portfolio_info.portfolio_value
        assets_pnl = list(portfolio_info.assets_pnl.values())
        change_value = portfolio_info.changed_value

        _returns_ = trades_return + [portfolio_value] + assets_pnl + [change_value]
        self._returns_collection_.append(_returns_)
        self._reward_collection_.append(reward)

        if self.configs.collect_step_detail:
            step_info = {
                "Date": str(self._market_state_.trading_time),
                "Action": ACTION[action],
                "ActionsType": actions_type,
                "Reward": round_size.round_number(reward),
                "PredictedReward": round_size.round_number(float(predicted_reward)),
                "TargetPrices": list(self._market_state_.target_price.values()),
                "AvgPrices": list(self._portfolio_manager_.avg_prices.values()),
                "Sizes": list(self._portfolio_manager_.sizes.values()),
                "TradesReturn": trades_return,
                "PortfolioValue": portfolio_value,
                "AssetsPnL": assets_pnl,
                "Change": change_value,
            }
            self._step_info_collection_.append(step_info)

    def __init__(self, configs: Dict) -> None:
        if cython.compiled:
            print("Running compiled mode!")
            gc.enable()
            print("Enabled GC automatically!")

        self.configs = SimulatorConfigurations(
            symbols=['BTCUSDT', 'ETHUSDT'],
            initial_equity=1500,
            num_weeks_train=configs['num_weeks_train'],
            evaluation=configs['evaluation'],
            collect_step_detail=configs['show_trade_result']
        )

        self._returns_collection_: List[List[float]] = []
        self._reward_collection_: List[float] = []
        self._step_info_collection_: List[Dict] = []
        self._action_: cython.int = 0

        self._price_manager_ = HistoricalPriceManager(
                                    symbols=self.configs.symbols,
                                    num_weeks_train=self.configs.num_weeks_train,
                                    evaluation=self.configs.evaluation
        )

        self._portfolio_manager_ = PortfolioManager(
                                    symbols=self.configs.symbols,
                                    initial_equity=self.configs.initial_equity
        )
        self._market_state_: MarketState = None
        self._compelete_ = False
    
    def observation_state(self):

        # Generate user stat
        portfolio_obs: numpy.ndarray = \
            zscore(numpy.array(self._returns_collection_))[-1:]
        numpy.nan_to_num(portfolio_obs, copy=False)
        reward_obs: numpy.ndarray = zscore(numpy.array(self._reward_collection_))[-1:]
        numpy.nan_to_num(reward_obs, copy=False)
        action_obs: numpy.ndarray = numpy.array([0 for _ in range(NUM_ACTIONS)])
        action_obs[self._action_] = 1

        # Combine market data with user stat
        _concat_obs: numpy.ndarray = numpy.concatenate(
            (portfolio_obs.squeeze(), reward_obs, action_obs, self._market_state_.obs)
        )

        _concat_obs: numpy.ndarray = numpy.array(_concat_obs, dtype=numpy.float32)

        # Check avaiable actions
        action_mask: numpy.ndarray = \
            numpy.array([1 for _ in range(NUM_ACTIONS)], dtype=numpy.int8)

        for i in range(NUM_ACTIONS):
            can_rebalance, _ = self._portfolio_manager_.avaiable_action(i)

            # No action
            if can_rebalance is False and i != 5:
                action_mask[i] = 0

        return {
            "action_mask": action_mask,
            "observations": _concat_obs,
        }
    
    def _reward_scheme(
        self,
        action: cython.int,
        execute_result: ExecuteResult,
    ) -> cython.float:
        trades_return = sum(execute_result[0].values())
        portfolio_info = self._portfolio_manager_.portfolio_info
        assets_pnl = sum(portfolio_info.assets_pnl.values())

        reward: cython.float = 0

        # Reward for total portfolio return
        if assets_pnl < 0:
            reward -= 0.004
        else:
            reward += 0.004

        # Reward closing position
        if trades_return < -0.001:
            reward -= 0.05
        elif trades_return > 0.001:
            reward += 0.05

        return reward

    def step(self, action: numpy.int32, predicted_value: float = 0.0):
        _action_: cython.int = int(action)
        execute_result: ExecuteResult = self._portfolio_manager_.execute_order(_action_)
        reward: cython.float = self._reward_scheme(_action_, execute_result)
        self._action_ = _action_

        if self._compelete_ is False:
            self._update_state(_action_, execute_result, reward, predicted_value)
            self._market_state_ = self._price_manager_.market_obs()

            self._portfolio_manager_.update_market_price(
                self._market_state_.target_price
            )
            self._compelete_ = self._market_state_.done

            if self.portfolio_return[0] < -30:
                self._compelete_ = True

            return self.observation_state(), reward, False, {}

        else:
            if self.portfolio_return[1] > self._price_manager_.performance:
                reward += 2
            else:
                reward -= 2

            if self.portfolio_return[0] > 20:
                reward += 4
            else:
                reward -= 4

            self._update_state(_action_, execute_result, reward, predicted_value)
            return self.observation_state(), reward, True, {}

    @property
    def action_space(self):
        return spaces.Discrete(NUM_ACTIONS)

    @property
    def observation_space(self):
        total = 39 # Portfolio, Reward, PrevAction, MarketObs

        return spaces.Dict(
            {
                "action_mask": spaces.Box(0, 1, shape=(NUM_ACTIONS,), dtype=numpy.int8),
                "observations": spaces.Box(
                    low=numpy.array([-20 for _ in range(total)]),
                    high=numpy.array([20 for _ in range(total)]),
                    dtype=numpy.float32,
                ),
            }
        )

    def reset(self, num_train_weeks_pct: float):

        self._returns_collection_.clear()
        self._reward_collection_.clear()
        self._step_info_collection_.clear()
        self._compelete_ = False
        self._action_ = None


        self._market_state_ = \
            self._price_manager_.reset(num_train_weeks_pct=num_train_weeks_pct)
        self._portfolio_manager_.reset()

        portfolio_obs = numpy.array([0, 0, 0, 0, 0, 0])
        reward_obs = numpy.array([0])
        action_obs = numpy.array([0 for _ in range(NUM_ACTIONS)])

        self._portfolio_manager_.update_market_price(self._market_state_.target_price)

        _concat_obs = numpy.concatenate((portfolio_obs,
                                         reward_obs,
                                         action_obs,
                                         self._market_state_.obs))
        gc.collect()

        return {
            "action_mask": numpy.array(object=[1 for _ in range(NUM_ACTIONS)],
                                       dtype=numpy.int8),
            "observations": _concat_obs,
        }

    @property
    def portfolio_return(self):
        portfolio_info = self._portfolio_manager_.portfolio_info
        pnl_value = portfolio_info.portfolio_value - self.configs.initial_equity
        pnl_pct = pnl_value / self.configs.initial_equity

        return pnl_value, pnl_pct

    @property
    def trade_result(self):
        trade_result_df = pandas.DataFrame(self._step_info_collection_)
        trade_result_df["Date"] = \
            trade_result_df["Date"].apply(
            lambda x: datetime.utcfromtimestamp(float(x) / 1000)
        )
        return trade_result_df