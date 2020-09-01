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


class LinesCapacityReward(BaseReward):
    """
    Reward based on lines capacity usage
    Returns max reward if no current is flowing in the lines
    Returns min reward if all lines are used at max capacity

    Compared to `:class:L2RPNReward`:
    This reward is linear (instead of quadratic) and only 
    considers connected lines capacities
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)

    def initialize(self, env):
        pass

    def __call__(self,  action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        obs = env.get_obs()
        n_connected = np.sum(obs.line_status.astype(dt_float))
        usage = np.sum(obs.rho[obs.line_status == True])
        usage = np.clip(usage, 0.0, float(n_connected))
        reward = np.interp(n_connected - usage,
                           [dt_float(0.0), float(n_connected)],
                           [self.reward_min, self.reward_max])
        return reward
