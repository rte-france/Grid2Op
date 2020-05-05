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

class LinesReconnectedReward(BaseReward):
    """
    This reward computes a penalty
    based on the number of off cooldown disconnected lines
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.penalty_max_at_n_lines = dt_float(2.0)

    def __call__(self, action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get obs from env
        obs = env.current_obs

        # All lines ids
        lines_id = np.array(list(range(env.n_line)))
        # Only off cooldown lines
        lines_off_cooldown = lines_id[obs.time_before_cooldown_line <= 0 ]

        n_penalties = dt_float(0.0)
        for line_id in lines_off_cooldown:
            # Line could be reconnected but isn't
            if obs.line_status[line_id] == False:
                n_penalties += dt_float(1.0)

        max_p = self.penalty_max_at_n_lines
        n_penalties = dt_float(max(max_p, n_penalties))
        r = np.interp(n_penalties, [dt_float(0.0), max_p],
                      [self.reward_min, self.reward_max])
        return dt_float(r)
