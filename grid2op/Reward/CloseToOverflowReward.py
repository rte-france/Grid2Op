import numpy as np
from grid2op.Reward.BaseReward import BaseReward


class CloseToOverflowReward(BaseReward):
    """
    This reward finds all lines close to overflowing.
    Returns max reward when there is no overflow, min reward if more than one line is close to overflow
    and the mean between max and min reward if one line is close to overflow
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = 0.0
        self.reward_max = 1.0
        self.max_overflowed = 5.0

    def initialize(self, env):
        pass
        
    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        thermal_limits = env.backend.get_thermal_limit()
        lineflow_ratio = env.current_obs.rho

        close_to_overflow = 0.0
        for ratio, limit in zip(lineflow_ratio, thermal_limits):
            # Seperate big line and small line
            if (limit < 400.00 and ratio > 0.90) or ratio >= 0.95:
                close_to_overflow += 1.0

        close_to_overflow = np.clip(close_to_overflow, 0.0, self.max_overflowed)
        penalty = np.interp(close_to_overflow, [0.0, self.max_overflowed], [self.reward_min, self.reward_max])
        return self.reward_max - penalty
