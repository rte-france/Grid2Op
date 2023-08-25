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


class EpisodeDurationReward(BaseReward):
    """
    This reward will always be 0., unless at the end of an episode where it will return the number
    of steps made by the agent divided by the total number of steps possible in the episode.

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import EpisodeDurationReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=EpisodeDurationReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the EpisodeDurationReward class

    Notes
    -----
    In case of an environment being "fast forward" (see :func:`grid2op.Environment.BaseEnv.fast_forward_chronics`)
    the time "during" the fast forward are counted "as if" they were successful.

    This means that if you "fast forward" up until the end of an episode, you are likely to receive a reward of 1.0


    """

    def __init__(self, per_timestep=1, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.per_timestep = dt_float(per_timestep)
        self.total_time_steps = dt_float(0.0)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)

    def initialize(self, env):
        self.reset(env)

    def reset(self, env):
        if env.chronics_handler.max_timestep() > 0:
            self.total_time_steps = env.max_episode_duration() * self.per_timestep
        else:
            self.total_time_steps = np.inf
            self.reward_max = np.inf

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            res = env.nb_time_step
            if np.isfinite(self.total_time_steps):
                res /= self.total_time_steps
        else:
            res = self.reward_min
        return res
