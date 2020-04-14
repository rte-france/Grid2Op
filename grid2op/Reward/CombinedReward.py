# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Reward.BaseReward import BaseReward


class CombinedReward(BaseReward):
    """
    This class allows to combine multiple rewards, by summing them for example.
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = 0.0
        self.reward_max = 0.0
        self.rewards = {}

    def addReward(self, reward_name, reward_instance, reward_weight = 1.0):
        self.rewards[reward_name] = {
            "instance": reward_instance,
            "weight": reward_weight
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
        for k, v in self.rewards.items():
            r_dict = dict(v["instance"])
            r_dict["weight"] = v["weight"]
            yield (k, r_dict)

    def initialize(self, env):
        for key, reward in self.rewards.items():
            reward_w = reward["weight"]
            reward_instance = reward["instance"]
            reward_instance.initialize(env)
            self.reward_max += reward_instance.reward_max * reward_w
            self.reward_min += reward_instance.reward_min * reward_w

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        res = 0.0
        # Loop over registered rewards
        for key, reward in self.rewards.items():
            r_instance = reward["instance"]
            # Call individual reward
            r = r_instance(action, env, has_error, is_done, is_illegal, is_ambiguous)
            # Sum by weighted result
            w = reward["weight"]
            res += r * w
        # Return total sum
        return res
