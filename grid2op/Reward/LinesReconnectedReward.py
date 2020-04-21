import numpy as np

from grid2op.Reward.BaseReward import BaseReward

class LinesReconnectedReward(BaseReward):
    """
    This reward computes a penalty
    based on the number of off cooldown disconnected lines
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = -1.0
        self.reward_max = 0.0
        self.penalty_max_at_n_lines = 2.0

    def __call__(self, action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        # Get obs from env
        obs = env.current_obs

        # All lines ids
        lines_id = np.array(list(range(env.n_line)))
        # Only off cooldown lines
        lines_off_cooldown = lines_id[
            np.logical_and(
                (obs.time_before_cooldown_line <= 0), # Can be acted on
                (obs.time_before_line_reconnectable <= 0) # Can be reconnected
            )
        ]

        n_penalties = 0.0
        for line_id in lines_off_cooldown:
            # Line could be reconnected but isn't
            if obs.line_status[line_id] == False:
                n_penalties += 1.0

        max_p = self.penalty_max_at_n_lines
        n_penalties = max(max_p, n_penalties)
        r = np.interp(n_penalties, [0.0, max_p],
                      [self.reward_min, self.reward_max])
        return r
