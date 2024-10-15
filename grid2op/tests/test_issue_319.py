# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""Script to reproduce and demonstrate the bug in the BackendAction class of grid2op."""
import pdb
import warnings
import unittest

import grid2op
import numpy as np

class Issue319Tester(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "rte_case14_realistic",
                test=True,
                _add_to_name=type(self).__name__
                )
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_backend_action_simple(self):
        disc_lines = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
                               -1, -1, -1, -1, -1, -1, -1])
        self.env._backend_action.update_state(disc_lines)
        assert np.all(self.env._backend_action.current_topo.values >= 1)
        disc_lines[5] = 0
        self.env._backend_action.update_state(disc_lines)
        # line 0 is not disconnected
        assert self.env._backend_action.current_topo.values[self.env.line_or_pos_topo_vect[0]] == 1
        assert self.env._backend_action.current_topo.values[self.env.line_ex_pos_topo_vect[0]] == 1
        # line 5 should be disconnected
        assert self.env._backend_action.current_topo.values[self.env.line_or_pos_topo_vect[5]] == -1
        assert self.env._backend_action.current_topo.values[self.env.line_ex_pos_topo_vect[5]] == -1
    
    def test_in_real_scenario(self):
        # change the thermal limit to fake a cascading failure
        # on line 10 and 19
        th_lim = 1.0 * self.env._thermal_limit_a
        th_lim[[10, 19]] = 0.8 * np.array([17.683975, 635.8966]) 
        self.env.set_thermal_limit(th_lim)
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        # cascading failure should happen now !
        obs, rew, done, info = self.env.step(self.env.action_space())
        assert not done       
        assert np.all(info["disc_lines"][[10, 19]] == 0)
        # line 0 is not disconnected
        assert self.env._backend_action.current_topo.values[self.env.line_or_pos_topo_vect[0]] == 1
        assert self.env._backend_action.current_topo.values[self.env.line_ex_pos_topo_vect[0]] == 1
        # lines 10, 19 should be disconnected
        assert np.all(self.env._backend_action.current_topo.values[self.env.line_or_pos_topo_vect[[10, 19]]] == -1)
        assert np.all(self.env._backend_action.current_topo.values[self.env.line_ex_pos_topo_vect[[10, 19]]] == -1)
    
    
if __name__ == '__main__':
    unittest.main()
