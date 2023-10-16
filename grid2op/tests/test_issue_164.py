# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

import grid2op
from grid2op.Reward import BaseReward
from grid2op.dtypes import dt_float
from grid2op.Exceptions import DivergingPowerFlow


class Test164_Reward(BaseReward):
    def __init__(self):
        super().__init__()
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error:
            return self.reward_min
        elif is_done:
            return self.reward_max

        return dt_float(0.0)


class Issue164Tester(unittest.TestCase):
    def test_issue_164(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make('rte_case14_realistic', reward_class=Test164_Reward, test=True,
                               _add_to_name=type(self).__name__)
        max_timestep = env.chronics_handler.max_timestep()

        obs = env.reset()
        env.fast_forward_chronics(max_timestep - 3)
        obs = env.get_obs()

        while True:
            obs, reward, done, info = env.step(env.action_space())

            assert not info["exception"], "there should not be any exception"

            if done:
                assert reward == 1.0, "wrong reward computed when episode is over"
                break


if __name__ == "__main__":
    unittest.main()
