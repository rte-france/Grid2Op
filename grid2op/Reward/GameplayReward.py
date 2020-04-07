# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Reward.BaseReward import BaseReward


class GameplayReward(BaseReward):
    """
    Most basic implementation of reward: everything has the same values.

    Note that this :class:`BaseReward` subtype is not usefull at all, whether to train an :attr:`BaseAgent` nor to assess its
    performance of course.

    """
    def __init__(self):
        BaseReward.__init__(self)
        self.min_reward = -1000.0
        self.max_reward = 1000.0

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            # Broke the game or did not respect the rules
            return self.min_reward
        elif is_done:
            # Bonus for playing a full episode
            return self.max_reward
        else:
            # Keep playing
            return 0.0
