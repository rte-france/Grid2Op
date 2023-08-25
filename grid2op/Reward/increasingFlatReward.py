# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Reward.baseReward import BaseReward
from grid2op.dtypes import dt_float


class IncreasingFlatReward(BaseReward):
    """
    This reward just counts the number of timestep the agent has successfully manage to perform.

    It adds a constant reward for each time step successfully handled.

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import IncreasingFlatReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=IncreasingFlatReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the IncreasingFlatReward class

    """

    def __init__(self, per_timestep=1, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.per_timestep = dt_float(per_timestep)
        self.reward_min = dt_float(0.0)

    def initialize(self, env):
        if env.chronics_handler.max_timestep() > 0:
            self.reward_max = env.chronics_handler.max_timestep() * self.per_timestep
        else:
            self.reward_max = np.inf

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not has_error:
            res = dt_float(env.nb_time_step * self.per_timestep)
        else:
            res = self.reward_min
        return res
