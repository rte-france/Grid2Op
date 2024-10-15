# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import warnings
import unittest

import grid2op
import numpy as np
import pdb


class Issue327Tester(unittest.TestCase):
    """test that i can retrieve all the "graph" information if the observation is "done", 
    this made grid2op <= 1.7.0 versions fail"""
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def _aux_set_game_over(self):
        act = self.env.action_space({"set_bus": {"loads_id": [(0, -1)]}})
        obs, reward, done, info = self.env.step(act)
        assert done
        return obs
     
    def test_get_energy_graph(self):
        obs = self._aux_set_game_over()
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            graph = obs.get_energy_graph()
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0
    
    def test_flow_bus_matrix(self):
        obs = self._aux_set_game_over()
        flow_mat, (load_bus, prod_bus, stor_bus, lor_bus, lex_bus) = obs.flow_bus_matrix()
        assert flow_mat.shape == (1,1)
        assert flow_mat.sum() == 0.
        
    def test_bus_connectivity_matrix(self):
        obs = self._aux_set_game_over()
        bus_bus_graph = obs.bus_connectivity_matrix(return_lines_index=False)
        assert bus_bus_graph.shape == (1,1)
        assert bus_bus_graph.sum() == 0.
        
    def test_connectivity_matrix(self):
        obs = self._aux_set_game_over()
        connectivity_matrix = obs.connectivity_matrix()
        assert np.all(connectivity_matrix == 0)
        assert np.all(connectivity_matrix.shape == (self.env.dim_topo, self.env.dim_topo))
        
        
if __name__ == "__main__":
    unittest.main()
