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
from grid2op.dtypes import dt_float


class RedispReward(BaseReward):
    """
    This reward can be used for environments where redispatching is available. It assigns a cost to redispatching action
    and penalizes with the losses.

    This is the closest reward to the score used for the l2RPN competitions.

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import RedispReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "rte_case14_realistic"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=RedispReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the RedispReward class

        # NB this is the default reward of many environments in the grid2op framework

    """
    def __init__(self, alpha_redisph=5.0):
        BaseReward.__init__(self)
        self.reward_min = None
        self.reward_max = None
        self.max_regret = dt_float(0.0)
        self.alpha_redisph = dt_float(alpha_redisph)

    def initialize(self, env):
        if not env.redispatching_unit_commitment_availble:
            raise Grid2OpException("Impossible to use the RedispReward reward with an environment without generators"
                                   "cost. Please make sure env.redispatching_unit_commitment_availble is available.")
        worst_marginal_cost = np.max(env.gen_cost_per_MW)
        worst_load = dt_float(np.sum(env.gen_pmax))
        worst_losses = dt_float(0.05) * worst_load  # it's not the worst, but definitely an upper bound
        worst_redisp = self.alpha_redisph * np.sum(env.gen_pmax)  # not realistic, but an upper bound
        self.max_regret = (worst_losses + worst_redisp)*worst_marginal_cost
        self.reward_min = dt_float(-10.0)

        least_loads = dt_float(worst_load * 0.5)  # half the capacity of the grid
        least_losses = dt_float(0.015 * least_loads)  # 1.5% of losses
        least_redisp = dt_float(0.0)  # lower_bound is 0
        base_marginal_cost = np.min(env.gen_cost_per_MW[env.gen_cost_per_MW > 0.])
        min_regret = (least_losses + least_redisp) * base_marginal_cost
        self.reward_max = dt_float((self.max_regret - min_regret) / least_loads)

    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            res = self.reward_min
        else:
            # compute the losses
            gen_p, *_ = env.backend.generators_info()
            load_p, *_ = env.backend.loads_info()
            losses = np.sum(gen_p) - np.sum(load_p)

            # compute the marginal cost
            gen_activeprod_t = env._gen_activeprod_t
            marginal_cost = np.max(env.gen_cost_per_MW[gen_activeprod_t > 0.])

            # redispatching amount
            actual_dispatch = env._actual_dispatch
            redisp_cost = self.alpha_redisph * np.sum(np.abs(actual_dispatch)) * marginal_cost

            # cost of losses
            losses_cost = losses * marginal_cost

            # total "regret"
            regret = losses_cost + redisp_cost

            # compute reward
            reward = self.max_regret - regret

            # divide it by load, to be less sensitive to load variation
            res = dt_float(reward / np.sum(load_p))

        return res
