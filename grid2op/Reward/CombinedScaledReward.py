# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.Reward.CombinedReward import CombinedReward
from grid2op.dtypes import dt_float


class CombinedScaledReward(CombinedReward):
    """
    This class allows to combine multiple rewards. 
    It will compute a scaled reward of the weighted sum of the registered rewards.
    Scaling is done by linearly interpolating the weighted sum,
    from the range [min_sum; max_sum] to [reward_min; reward_max]

    min_sum and max_sum are computed from the weights and ranges of registered rewards.
    See :class:`Reward.BaseReward` for setting the output range.

    Examples
    --------

    .. code-block:: python

        import grid2op
        from grid2op.Reward import GameplayReward, FlatReward, CombinedScaledReward

        env = grid2op.make(..., reward_class=CombinedScaledReward)
        cr = self.env.get_reward_instance()
        cr.addReward("Gameplay", GameplayReward(), 1.0)
        cr.addReward("Flat", FlatReward(), 1.0)
        cr.initialize(self.env)

        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())

        # reward here is computed by summing the results of what would have
        # given `GameplayReward` and the one from `FlatReward`


    """

    def __init__(self):
        super().__init__()
        self.reward_min = dt_float(-0.5)
        self.reward_max = dt_float(0.5)
        self._sum_max = dt_float(0.0)
        self._sum_min = dt_float(0.0)
        self.rewards = {}

    def initialize(self, env):
        """
        Overloaded initialze from `Reward.CombinedReward`.
        This is because it needs to store the ranges internaly
        """
        self._sum_max = dt_float(0.0)
        self._sum_min = dt_float(0.0)
        for key, reward in self.rewards.items():
            reward_w = dt_float(reward["weight"])
            reward_instance = reward["instance"]
            reward_instance.initialize(env)
            self._sum_max += dt_float(reward_instance.reward_max * reward_w)
            self._sum_min += dt_float(reward_instance.reward_min * reward_w)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        # Get weighted sum from parent
        ws = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
        # Scale to range
        res = np.interp(ws, [self._sum_min, self._sum_max], [self.reward_min, self.reward_max])
        return dt_float(res)
