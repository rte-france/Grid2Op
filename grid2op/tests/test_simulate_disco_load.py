# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import numpy as np
import unittest
import warnings

import grid2op
from lightsim2grid import LightSimBackend

import pdb

class TestSimulateDiscoLoad(unittest.TestCase):
    def setUp(self) -> None:
        """its important to keep the lightims2grid backend here. It tests indirectly that the objects
        are properly set to "unchanged" without actually having to check the _BackendAction of 
        the obs.simulate underlying backend, which is quite annoying"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            # self.env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend(), test=True)  # TODO when lightsim will be fixed !
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_simulate_ok(self):
        obs = self.env.reset()
        simo, simr, simd, simi = obs.simulate(self.env.action_space())
        assert not simd
        
        simo, simr, simd, simi = obs.simulate(self.env.action_space({"set_bus": {"loads_id": [(0, -1)]}}))
        assert simd
        
        simo, simr, simd, simi = obs.simulate(self.env.action_space())
        assert not simd
    
    def test_backend_action(self):
        obs = self.env.reset()
        simo, simr, simd, simi = obs.simulate(self.env.action_space())
        l_id = 0
        l_pos = type(self.env).load_pos_topo_vect[l_id]
        assert obs._obs_env._backend_action_set.current_topo.values[l_pos] == 1
        assert obs._obs_env._backend_action_set.load_p.changed[l_id]
        assert np.allclose(obs._obs_env._backend_action_set.load_p.values[l_id], 22.3), f"{obs._obs_env._backend_action_set.load_p.values[l_id]:.2f} vs 22.3"
        
        obs._obs_env._backend_action_set += self.env.action_space({"set_bus": {"loads_id": [(l_id, -1)]}})
        assert obs._obs_env._backend_action_set.current_topo.values[l_pos] == -1
        tmp = obs._obs_env._backend_action_set()  # do as if the action has been processed
        assert not obs._obs_env._backend_action_set.load_p.changed[l_id]  # it's disconnected, so marked as unchanged now
        assert np.allclose(obs._obs_env._backend_action_set.load_p.values[l_id], 22.3), f"{obs._obs_env._backend_action_set.load_p.values[l_id]:.2f} vs 22.3"
        
        
if __name__ == '__main__':
    unittest.main()
