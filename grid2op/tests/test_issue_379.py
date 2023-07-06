# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/rte-france/Grid2Op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings

from grid2op.gym_compat import (GymEnv, GYM_AVAILABLE, GYMNASIUM_AVAILABLE)
import grid2op


CAN_TEST_ALL = True
if GYMNASIUM_AVAILABLE:
    from gymnasium.utils.env_checker import check_env
    from gymnasium.utils.env_checker import check_reset_return_type, check_reset_options, check_reset_seed
elif GYM_AVAILABLE:
    from gym.utils.env_checker import check_env
    from gym.utils.env_checker import check_reset_return_type, check_reset_options, check_reset_seed
else:
    CAN_TEST_ALL = False


class Issue379Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True)
            self.gym_env = GymEnv(self.env)
    
    def tearDown(self) -> None:
        self.env.close()
        self.gym_env.close()
        return super().tearDown()
    
    def test_check_env(self):
        if CAN_TEST_ALL:
            check_reset_return_type(self.gym_env)
            check_reset_seed(self.gym_env)
            check_reset_options(self.gym_env)
        check_env(self.gym_env)
    

if __name__ == "__main__":
    unittest.main()
