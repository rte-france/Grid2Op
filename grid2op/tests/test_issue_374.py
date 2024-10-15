# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import grid2op
import unittest
import warnings

import re

from grid2op.Parameters import Parameters


class Issue367Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Creation of the environment
            param = Parameters()
            param.NB_TIMESTEP_COOLDOWN_SUB = 3
            param.NB_TIMESTEP_COOLDOWN_LINE = 3
            param.NO_OVERFLOW_DISCONNECTION = True
            self.env = grid2op.make('l2rpn_wcci_2022', param=param, _add_to_name=type(self).__name__)
            
        self.env.set_id(0) 
        self.env.seed(0)
        self.obs = self.env.reset()

    def test_cooldown(self):
        # second, get some action that has no actual effect on the grid
        action_space = self.env.action_space
        env = self.env
        do_nothing = action_space()
        
        # take an action
        set_sub_10 = action_space.get_all_unitary_topologies_set(action_space, sub_id=10)
        set_sub10_to_bus1 = set_sub_10[5]
        
        # do the action to trigger the cooldown
        obs, _, done, info = env.step(set_sub10_to_bus1)
        assert obs.time_before_cooldown_sub[10] == 3
        assert info['exception'] == []

        # check cooldown
        obs, _, done, info = env.step(do_nothing)
        assert obs.time_before_cooldown_sub[10] == 2
        assert info['exception'] == []
        # check i cannot simulate
        obs_, _, done, info = obs.simulate(set_sub10_to_bus1)
        assert  info['exception'] != [], "simulate should have raised an error"

        # check cooldown
        obs, _, done, info = env.step(do_nothing)
        assert obs.time_before_cooldown_sub[10] == 1
        assert info['exception'] == []
        
        # check i cannot simulate
        obs_, _, done, info = obs.simulate(set_sub10_to_bus1)
        assert obs_.time_before_cooldown_sub[10] == 0
        assert  info['exception'] != [], "simulate should have raised an error"
        # check i cannot "step"
        obs, _, done, info = env.step(set_sub10_to_bus1)
        assert obs.time_before_cooldown_sub[10] == 0
        assert info['exception'] != [], "step should have raised an error"


        # check i can simulate
        obs_, _, done, info = obs.simulate(set_sub10_to_bus1)
        assert obs_.time_before_cooldown_sub[10] == 3
        assert  info['exception'] == []
        # check i can "step"
        obs, _, done, info = env.step(set_sub10_to_bus1)
        assert obs.time_before_cooldown_sub[10] == 3
        assert info['exception'] == []


    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    

if __name__ == "__main__":
    unittest.main()
