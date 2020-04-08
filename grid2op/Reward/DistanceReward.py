import numpy as np

from grid2op.Reward.BaseReward import BaseReward

class DistanceReward(BaseReward):
    """
    This reward computes a penalty based on the distance of the current grid to the grid at time 0.
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = 0.0
        self.reward_max = 1.0

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        # Get topo from env
        obs = env.current_obs
        topo = obs.topo_vect

        idx = 0
        diff = 0.0
        for n_elems_on_sub in obs.sub_info:
            # Find this substation elements range in topology vect
            sub_start = idx
            sub_end = idx + n_elems_on_sub
            current_sub_topo = topo[sub_start:sub_end]

            # Count number of elements not on bus 1
            # Because at the initial state, all elements are on bus 1
            diff += 1.0 * np.count_nonzero(current_sub_topo != 1)

            # Set index to next sub station
            idx += n_elems_on_sub


        r = np.interp(diff, [0.0, len(topo) * 1.0], [self.reward_min, self.reward_max])
        r = self.reward_max - r
        return r
