import numpy as np
from abc import ABC, abstractmethod

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.Reward import Reward

class FlatReward(Reward):
    """
    This reward return a fixed number (if there are not error) or 0 if there is an error.

    """
    def __init__(self, per_timestep=1):
        Reward.__init__(self)
        self.per_timestep = per_timestep
        self.total_reward = 0
        self.reward_min = 0
        self.reward_max = per_timestep

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not has_error:
            res = self.per_timestep
        else:
            res = self.reward_min
        return res
