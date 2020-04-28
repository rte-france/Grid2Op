# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Reward.BaseReward import BaseReward
from grid2op.dtypes import dt_float

class GameplayReward(BaseReward):
    """
    This rewards is strictly computed based on the Game status.
    It yields a negative reward in case of game over.
    A positive reward if the game is won (finished an episode)
    Otherwise the reward is zero
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            # Broke the game or did not respect the rules
            return self.reward_min
        else:
            # Keep playing or finished episode
            return self.reward_max
