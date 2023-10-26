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
from grid2op.Action import CompleteAction
from grid2op.gym_compat import GymEnv
from grid2op.gym_compat import BoxGymActSpace


class Issue282Tester(unittest.TestCase):
    def setUp(self):
        self._default_act_attr_to_keep = ["redispatch", "curtail", "set_storage"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage", test=True, action_class=CompleteAction,
                _add_to_name=type(self).__name__
            )
        self.env_gym = GymEnv(self.env)

        self.env_gym.action_space.close()
        self.env_gym.action_space = BoxGymActSpace(
            self.env.action_space, attr_to_keep=self._default_act_attr_to_keep
        )
        self.env_gym.reset(seed=0)  #reset and seed

    def tearDown(self):
        self.env.close()
        self.env_gym.close()

    def test_can_make(self):
        """test that the opponent state is correctly copied"""
        act = self.env_gym.action_space.from_gym(self.env_gym.action_space.sample())
        obs, reward, done, info = self.env.step(act)
        assert len(info["exception"]) == 0, f"{info['exception'] = }"


if __name__ == "__main__":
    unittest.main()
