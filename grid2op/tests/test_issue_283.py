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
from grid2op.Rules.AlwaysLegal import AlwaysLegal
from grid2op.gym_compat import GymEnv
from grid2op.gym_compat import BoxGymActSpace


class Issue283Tester(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage", test=True, gamerules_class=AlwaysLegal,
                _add_to_name=type(self).__name__
            )
            self.env_gym = GymEnv(self.env)

        self.env_gym.action_space.close()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_gym.action_space = BoxGymActSpace(self.env.action_space)
        # self.env_gym.seed(0)
        self.env_gym.reset(seed=0)

    def tearDown(self):
        self.env.close()
        self.env_gym.close()

    def test_can_make(self):
        """test that the opponent state is correctly copied"""
        gym_act = self.env_gym.action_space.sample()
        gym_act[
            : self.env.n_line
        ] = 0.0  # do not change line status ! (otherwise it diverges)
        act = self.env_gym.action_space.from_gym(gym_act)
        obs, reward, done, info = self.env.step(act)
        assert len(info["exception"]) == 0, f"{info['exception'] = }"


if __name__ == "__main__":
    unittest.main()
