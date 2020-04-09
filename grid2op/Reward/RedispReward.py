# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.BaseReward import BaseReward


class RedispReward(BaseReward):
    """
    This reward can be used for environments where redispatching is availble. It assigns a cost to redispatching action
    and penalizes with the losses.
    """
    def __init__(self, alpha_redisph=5.0):
        BaseReward.__init__(self)
        self.reward_min = None
        self.reward_max = None
        self.max_regret = 0.
        self.alpha_redisph = alpha_redisph

    def initialize(self, env):
        if not env.redispatching_unit_commitment_availble:
            raise Grid2OpException("Impossible to use the RedispReward reward with an environment without generators"
                                   "cost. Please make sure env.redispatching_unit_commitment_availble is available.")
        worst_marginal_cost = np.max(env.gen_cost_per_MW)
        worst_load = np.sum(env.gen_pmax)
        worst_losses = 0.05 * worst_load  # it's not the worst, but definitely an upper bound
        worst_redisp = self.alpha_redisph * np.sum(env.gen_pmax)  # not realistic, but an upper bound
        self.max_regret = (worst_losses + worst_redisp)*worst_marginal_cost
        self.reward_min = -10

        least_loads = (worst_load * 0.5)  # half the capacity of the grid
        least_losses = 0.015 * least_loads  # 1.5% of losses
        least_redisp = 0.  # lower_bound is 0
        base_marginal_cost = np.min(env.gen_cost_per_MW[env.gen_cost_per_MW > 0.])
        min_regret = (least_losses + least_redisp) * base_marginal_cost
        self.reward_max = (self.max_regret - min_regret) / least_loads

    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            res = self.reward_min * 0.5
        else:
            # compute the losses
            gen_p, *_ = env.backend.generators_info()
            load_p, *_ = env.backend.loads_info()
            losses = np.sum(gen_p) - np.sum(load_p)

            # compute the marginal cost
            marginal_cost = np.max(env.gen_cost_per_MW[env.gen_activeprod_t > 0.])

            # redispatching amount
            redisp_cost = self.alpha_redisph * np.sum(np.abs(env.actual_dispatch)) * marginal_cost

            # cost of losses
            losses_cost = losses * marginal_cost

            # total "regret"
            regret = losses_cost + redisp_cost

            # compute reward
            reward = self.max_regret - regret

            # divide it by load, to be less sensitive to load variation
            res = reward / np.sum(load_p)

        return res
