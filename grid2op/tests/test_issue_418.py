# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.gym_compat import GymEnv, GYMNASIUM_AVAILABLE
import unittest
import warnings
import numpy as np
from grid2op.gym_compat.utils import (check_gym_version, sample_seed,
                                      _MAX_GYM_VERSION_RANDINT, GYM_VERSION)

class Issue418Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
    
    def test_seed(self):
        gymenv = GymEnv(self.env)
        if GYM_VERSION <= _MAX_GYM_VERSION_RANDINT and not GYMNASIUM_AVAILABLE:
            # legacy gym, with old gym gym version
            gymenv.seed(42)
            obs = gymenv.reset()
            curt = np.array([1,1.,0.35566905,0.23095788,0.6338101,1])
            year = 1249
            day = 28
        else:
            # most recent gym API
            obs = gymenv.reset(seed=42)
            curt = np.array([1,1.,0.18852758,0.5537014,0.43770432,1])
            curt = np.array([-1,-1.,0.18852758,0.5537014,0.43770432,-1])
            year = 571
            day = 9
            # year = 1887
            # day = 9
        
        # test that the seeding worked also in action space and observation space
        sampled_act = gymenv.action_space.sample()
        assert np.allclose(sampled_act['curtail'], curt), f"{sampled_act['curtail']}"

        sampled_obs = gymenv.observation_space.sample()
        assert sampled_obs["year"] == year, f'{sampled_obs["year"]}'
        assert sampled_obs["day"] == day, f'{sampled_obs["day"]}'


if __name__ == "__main__":
    unittest.main()
