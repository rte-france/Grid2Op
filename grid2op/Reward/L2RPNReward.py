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


class L2RPNReward(BaseReward):
    """
    This is the historical :class:`BaseReward` used for the Learning To Run a Power Network competition on WCCI 2019

    See `L2RPN <https://l2rpn.chalearn.org/>`_ for more information.

    This rewards makes the sum of the "squared margin" on each powerline.

    The margin is defined, for each powerline as:
    `margin of a powerline = (thermal limit - flow in amps) / thermal limit`
    (if flow in amps <= thermal limit) else `margin of a powerline  = 0.`

    This rewards is then: `sum (margin of this powerline) ^ 2`, for each powerline.


    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import L2RPNReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "rte_case14_realistic"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=L2RPNReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the L2RPNReward class

    """
    def __init__(self):
        BaseReward.__init__(self)

    def initialize(self, env):
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(env.backend.n_line)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            res = np.sum(line_cap)
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        return res

    @staticmethod
    def __get_lines_capacity_usage(env):
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-1  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

        x = np.minimum(relative_flow, dt_float(1.0))
        lines_capacity_usage_score = np.maximum(dt_float(1.0) - x ** 2, np.zeros(x.shape, dtype=dt_float))
        return lines_capacity_usage_score
