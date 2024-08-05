from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override

from market_simulator.engine import Simulator as StockMarket

class CurriculumSimulatorEnv(TaskSettableEnv):

    def __init__(self, config: EnvContext):
        self.config = config

        self.num_train_weeks_pct: float = 0.1 # 10%

        self.stock_market = None
        self._make_env()
        self.observation_space = self.stock_market.observation_space
        self.action_space = self.stock_market.action_space
        self.switch_env = False
        self._timesteps = 0

    def reset(self, *, seed=None, options=None):
        return self.stock_market.reset(self.num_train_weeks_pct)

    def step(self, action):
        return self.stock_market.step(action)

    @override(TaskSettableEnv)
    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        return self.num_train_weeks_pct

    @override(TaskSettableEnv)
    def set_task(self, task: float):
        """Implement this to set the task (curriculum level) for this env."""
        self.num_train_weeks_pct = task

    def _make_env(self):
        self.stock_market = StockMarket({
            "num_weeks_train": self.config.get("num_weeks_train"),
            "evaluation": self.config.get("evaluation"),
            "show_trade_result": self.config.get("show_trade_result"),
        })