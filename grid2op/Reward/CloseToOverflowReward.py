# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import numpy as np
from grid2op.Reward.BaseReward import BaseReward
from grid2op.dtypes import dt_float


class CloseToOverflowReward(BaseReward):
    """
    This reward finds all lines close to overflowing.
    Returns max reward when there is no overflow, min reward if more than one line is close to overflow
    and the mean between max and min reward if one line is close to overflow

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import CloseToOverflowReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "rte_case14_realistic"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=CloseToOverflowReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with this class (computing the penalty based on the number of overflow)

    """
    def __init__(self, max_lines=5):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.max_overflowed = dt_float(max_lines)

    def initialize(self, env):
        pass
        
    def __call__(self,  action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        thermal_limits = env.backend.get_thermal_limit()
        lineflow_ratio = env.current_obs.rho

        close_to_overflow = dt_float(0.0)
        for ratio, limit in zip(lineflow_ratio, thermal_limits):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= 0.95) or ratio >= 0.975:
                close_to_overflow += dt_float(1.0)

        close_to_overflow = np.clip(close_to_overflow,
                                    dt_float(0.0), self.max_overflowed)
        reward = np.interp(close_to_overflow,
                           [dt_float(0.0), self.max_overflowed],
                           [self.reward_max, self.reward_min])
        return reward
