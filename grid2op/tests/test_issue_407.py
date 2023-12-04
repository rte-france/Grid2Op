# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.gym_compat import GymEnv
import unittest
import warnings


class CustomGym(GymEnv):
    def __init__(self, env_init, shuffle_chronics=True, render_mode="rgb_array"):
        super().__init__(env_init, shuffle_chronics, render_mode)
        self._reset_called = 0
        
    def reset(self, *args, **kwargs):
        self._reset_called += 1
        super().reset(*args, **kwargs)

        
class Issue407Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
    
    def test_reset(self):
        gym_env = CustomGym(self.env)
        obs = gym_env.reset()
        assert gym_env._reset_called == 1


if __name__ == "__main__":
    unittest.main()
