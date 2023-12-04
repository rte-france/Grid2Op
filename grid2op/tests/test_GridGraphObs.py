# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import numpy as np
import networkx

import grid2op

import pdb


class TestNetworkXGraph(unittest.TestCase):
    """this class test the networkx representation of an observation."""

    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_neurips_2020_track1", test=True, _add_to_name=type(self).__name__)
        self.tol = 1e-5

    def test_kirchhoff(self):
        """
        test kirchhoff law

        in case of parallel lines
        """
        obs = self.env.reset()
        graph = obs.get_energy_graph()
        assert isinstance(graph, networkx.Graph), "graph should be a networkx object"
        ps = np.array([graph.nodes[el]["p"] for el in graph.nodes])
        qs = np.array([graph.nodes[el]["q"] for el in graph.nodes])

        p_out = np.zeros(ps.shape[0])
        q_out = np.zeros(ps.shape[0])
        for or_, ex_ in graph.edges:
            me = graph.edges[(or_, ex_)]
            p_out[or_] += me["p_or"]
            q_out[or_] += me["q_or"]
            p_out[ex_] += me["p_ex"]
            q_out[ex_] += me["q_ex"]

        assert np.max(np.abs(ps - p_out)) <= self.tol, "error for active flow"
        assert np.max(np.abs(qs - q_out)) <= self.tol, "error for reactive flow"

    def test_global_bus(self):
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 1, 2, 1, 2))]}})
        obs, reward, done, info = self.env.step(act)
        assert not done
        graph = obs.get_energy_graph()
        assert len(graph.nodes) == self.env.n_sub + 1
        assert (4, 36) in graph.edges
        assert graph.edges[(4, 36)]["bus_or"] == 2
        assert graph.edges[(4, 36)]["sub_id_or"] == 1
        assert graph.edges[(4, 36)]["sub_id_ex"] == 4
        assert graph.edges[(4, 36)]["node_id_or"] == 36
    
    def test_bus_cooldown(self):
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 1, 2, 1, 2))]}})
        obs, reward, done, info = self.env.step(act)
        assert not done
        graph = obs.get_energy_graph()
        assert graph.nodes[1]["cooldown"] == 3
        assert graph.nodes[36]["cooldown"] == 3
        
    def test_parrallel_lines(self):
        obs = self.env.reset()
        graph_init = obs.get_energy_graph()
        assert (9, 16) in graph_init.edges
        assert graph_init.edges[(9, 16)]["p_or"] == obs.p_or[18] + obs.p_or[19]
        assert graph_init.edges[(9, 16)]["p_ex"] == obs.p_ex[18] + obs.p_ex[19]
        assert graph_init.edges[(9, 16)]["v_or"] == obs.v_or[18]
        assert graph_init.edges[(9, 16)]["v_ex"] == obs.v_ex[18]
        assert graph_init.edges[(9, 16)]["nb_connected"] == 2
        
        act = self.env.action_space({"set_line_status": [(19, -1)]})  # parrallel to line 18
        obs, reward, done, info = self.env.step(act)
        assert not done
        graph = obs.get_energy_graph()
        assert len(graph.edges) == len(graph_init.edges)
        assert (9, 16) in graph.edges
        assert graph.edges[(9, 16)]["p_or"] == obs.p_or[18]
        assert graph.edges[(9, 16)]["p_ex"] == obs.p_ex[18]
        assert graph.edges[(9, 16)]["v_or"] == obs.v_or[18]
        assert graph.edges[(9, 16)]["v_ex"] == obs.v_ex[18]
        assert graph.edges[(9, 16)]["nb_connected"] == 1, f'{graph.edges[(9, 16)]["nb_connected"]}'
        
    def test_disconnected_line(self):
        obs = self.env.reset()
        graph_init = obs.get_energy_graph()
        assert (2, 3) in graph_init.edges
        act = self.env.action_space({"set_line_status": [(0, -1)]})  # parrallel to line 18
        obs, reward, done, info = self.env.step(act)
        assert not done
        graph = obs.get_energy_graph()
        assert (2, 3) not in graph.edges
        
        
if __name__ == "__main__":
    unittest.main()
