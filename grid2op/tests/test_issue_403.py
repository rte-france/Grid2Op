# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace
import unittest
import warnings
import numpy as np
import pdb


class Issue403Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
    
    def test_box_action_space(self):
        # We considers only redispatching actions
        gym_action_space = BoxGymActSpace(self.env.action_space,
                                          attr_to_keep=["redispatch"],
                                          add={"redispatch": [0.0, 0.0, 0.0]}, # Normalization part
                                          multiply={"redispatch": [5.0, 10.0, 15.0]} # Normalization part
                                          )
        assert np.all(gym_action_space.low == [-1, -1, -1])
        assert np.all(gym_action_space.high == [1, 1, 1])

        assert np.all(gym_action_space._dict_properties["redispatch"][0] == [ -5., -10., -15.])
        assert np.all(gym_action_space._dict_properties["redispatch"][1] == [5, 10, 15])
        
    
    def test_box_obs_space(self):
        # We considers only redispatching actions
        gym_obs_space = BoxGymObsSpace(self.env.observation_space,
                                       attr_to_keep=["target_dispatch"],
                                       subtract={"target_dispatch":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, # Normalization part
                                       divide={"target_dispatch":[140., 120.,  70.,  70.,  40., 100.]} # Normalization part
                                       )
        assert np.all(gym_obs_space.low == [-1, -1, -1, -1, -1, -1])
        assert np.all(gym_obs_space.high == [1, 1, 1, 1, 1, 1])
        
        assert np.all(gym_obs_space._dict_properties["target_dispatch"][0] == [-140., -120.,  -70.,  -70.,  -40., -100.])
        assert np.all(gym_obs_space._dict_properties["target_dispatch"][1] == [140., 120.,  70.,  70.,  40., 100.])
        
        
if __name__ == "__main__":
    unittest.main()
