# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Reward.baseReward import BaseReward
from grid2op.dtypes import dt_float


class ConstantReward(BaseReward):
    """
    Most basic implementation of reward: everything has the same values: 0.0

    Note that this :class:`BaseReward` subtype is not useful at all, whether to train an :attr:`BaseAgent`
    nor to assess its performance of course.


    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import ConstantReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=ConstantReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is 0., always... Not really useful

    """

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        return dt_float(0.0)
