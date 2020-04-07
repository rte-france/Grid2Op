import numpy as np
from abc import ABC, abstractmethod

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.BaseReward import BaseReward


class CloseToOverflowReward(BaseReward):
    """
    This reward finds all lines close to overflowing and then sums a negative fixed reward for each of the matched lines
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = 0.0
        self.reward_max = 1000.0

    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        penalty = 0.0
        
        obs_dict = env.current_obs.to_dict()
        thermal_limits = env.backend.get_thermal_limit()
        lineflow_ratio = obs_dict["rho"]
        
        for ratio, limit in zip(lineflow_ratio, thermal_limits):
            # Seperate big line and small line
            if (limit < 400.00 and ratio > 0.93) or (limit >= 400.00 and ratio > 0.97):
                penalty -= 250.0

        if penalty != 0.0:
            return penalty
        return self.reward_max
