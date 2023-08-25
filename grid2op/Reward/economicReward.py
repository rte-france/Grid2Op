# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.baseReward import BaseReward
from grid2op.dtypes import dt_float


class EconomicReward(BaseReward):
    """
    This reward computes the marginal cost of the powergrid. As RL is about maximising a reward, while we want to
    minimize the cost, this class also ensures that:

    - the reward is positive if there is no game over, no error etc.
    - the reward is inversely proportional to the cost of the grid (the higher the reward, the lower the economic cost).

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import EconomicReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=EconomicReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the EconomicReward class

    """

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.worst_cost = None

    def initialize(self, env):
        if not env.redispatching_unit_commitment_availble:
            raise Grid2OpException(
                "Impossible to use the EconomicReward reward with an environment without generators"
                "cost. Please make sure env.redispatching_unit_commitment_availble is available."
            )
        self.worst_cost = dt_float((env.gen_cost_per_MW * env.gen_pmax).sum() * env.delta_time_seconds / 3600.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            res = self.reward_min
        else:
            # compute the cost of the grid
            res = dt_float((env.get_obs(_do_copy=False).prod_p * env.gen_cost_per_MW).sum() * env.delta_time_seconds / 3600.0)
            # we want to minimize the cost by maximizing the reward so let's take the opposite
            res *= dt_float(-1.0)
            # to be sure it's positive, add the highest possible cost
            res += self.worst_cost

        res = np.interp(
            res, [dt_float(0.0), self.worst_cost], [self.reward_min, self.reward_max]
        )
        return dt_float(res)
