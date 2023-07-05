# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Reward.baseReward import BaseReward
from grid2op.dtypes import dt_float


class FlatReward(BaseReward):
    """
    This reward return a fixed number (if there are not error) or 0 if there is an error.

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import FlatReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "rte_case14_realistic"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=FlatReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the FlatReward class

    """

    def __init__(self, per_timestep=1, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.per_timestep = dt_float(per_timestep)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(per_timestep)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not has_error:
            res = self.per_timestep
        else:
            res = self.reward_min
        return res
