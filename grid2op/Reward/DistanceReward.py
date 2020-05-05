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

class DistanceReward(BaseReward):
    """
    This reward computes a penalty based on the distance of the current grid to the grid at time 0.
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get topo from env
        obs = env.current_obs
        topo = obs.topo_vect

        idx = 0
        diff = dt_float(0.0)
        for n_elems_on_sub in obs.sub_info:
            # Find this substation elements range in topology vect
            sub_start = idx
            sub_end = idx + n_elems_on_sub
            current_sub_topo = topo[sub_start:sub_end]

            # Count number of elements not on bus 1
            # Because at the initial state, all elements are on bus 1
            diff += dt_float(1.0) * np.count_nonzero(current_sub_topo != 1)

            # Set index to next sub station
            idx += n_elems_on_sub

        r = np.interp(diff,
                      [dt_float(0.0), len(topo) * dt_float(1.0)],
                      [self.reward_max, self.reward_min])
        return r
