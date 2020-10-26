# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Reward.BaseReward import BaseReward
from grid2op.dtypes import dt_float


class CombinedReward(BaseReward):
    """
    This class allows to combine multiple pre defined reward. The reward it computes will
    be the sum of all the sub rewards it is made of.

    Each sub reward is identified by a key.

    It is used a bit differently that the other rewards. See the section example for more information.

    Examples
    --------

    .. code-block:: python

        import grid2op
        from grid2op.Reward import GameplayReward, FlatReward, CombinedReward

        env = grid2op.make(..., reward_class=CombinedReward)
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
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(0.0)
        self.rewards = {}

    def addReward(self, reward_name, reward_instance, reward_weight = 1.0):
        self.rewards[reward_name] = {
            "instance": reward_instance,
            "weight": dt_float(reward_weight)
        }
        return True

    def removeReward(self, reward_name):
        if reward_name in self.rewards:
            self.rewards.pop(reward_name)
            return True
        return False

    def updateRewardWeight(self, reward_name, reward_weight):
        if reward_name in self.rewards:
            self.rewards[reward_name]["weight"] = reward_weight
            return True
        return False

    def __iter__(self):
        for k, v in super().__iter__():
            yield (k, v)
        for k, v in self.rewards.items():
            r_dict = dict(v["instance"])
            r_dict["weight"] = float(v["weight"])
            yield (k, r_dict)

    def initialize(self, env):
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(0.0)

        for key, reward in self.rewards.items():
            reward_w = reward["weight"]
            reward_instance = reward["instance"]
            reward_instance.initialize(env)
            self.reward_max += dt_float(reward_instance.reward_max * reward_w)
            self.reward_min += dt_float(reward_instance.reward_min * reward_w)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        res = dt_float(0.0)
        # Loop over registered rewards
        for key, reward in self.rewards.items():
            r_instance = reward["instance"]
            # Call individual reward
            r = r_instance(action, env, has_error, is_done, is_illegal, is_ambiguous)
            # Sum by weighted result
            w = dt_float(reward["weight"])
            res += dt_float(r) * w
        # Return total sum
        return res
