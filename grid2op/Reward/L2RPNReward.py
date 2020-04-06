# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Reward.BaseReward import BaseReward


class L2RPNReward(BaseReward):
    """
    This is the historical :class:`BaseReward` used for the Learning To Run a Power Network competition.

    See `L2RPN <https://l2rpn.chalearn.org/>`_ for more information.

    """
    def __init__(self):
        BaseReward.__init__(self)

    def initialize(self, env):
        self.reward_min = 0.
        self.reward_max = env.backend.n_line

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
        ampere_flows = np.abs(env.backend.get_line_flow())
        thermal_limits = np.abs(env.backend.get_thermal_limit())
        relative_flow = np.divide(ampere_flows, thermal_limits)

        x = np.minimum(relative_flow, 1)
        lines_capacity_usage_score = np.maximum(1 - x ** 2, 0.)
        return lines_capacity_usage_score
