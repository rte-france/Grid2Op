import numpy as np
from abc import ABC, abstractmethod

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.BaseReward import BaseReward


class IncreasingFlatReward(BaseReward):
    """
    This reward just counts the number of timestep the agent has sucessfully manage to perform.

    It adds a constant reward for each time step sucessfully handled.

    """
    def __init__(self, per_timestep=1):
        BaseReward.__init__(self)
        self.per_timestep = per_timestep
        self.total_reward = 0
        self.reward_min = 0

    def initialize(self, env):
        if env.chronics_handler.max_timestep() > 0:
            self.reward_max = env.chronics_handler.max_timestep() * self.per_timestep
        else:
            self.reward_max = np.inf

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not has_error:
            res = env.nb_time_step * self.per_timestep
        else:
            res = self.reward_min
        return res
