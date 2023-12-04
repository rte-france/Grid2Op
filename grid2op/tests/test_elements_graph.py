# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.Action import PlayableAction
import networkx
import unittest
import warnings
import pdb


class TestElementsGraph14SandBox(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
        self.tol = 1e-5
        
    def _aux_get_topo_action(self):
        sub_id = 4
        topo = (1, 2, 1, 2, 1)
        return sub_id, topo

    def _aux_test_kcl(self, graph):
        for bus_id, node_id in enumerate(graph.graph["bus_nodes_id"]):
            sum_p = 0.
            sum_q = 0.
            for ancestor in graph.predecessors(node_id):
                this_edge = graph.edges[(ancestor, node_id)]
                if "p" in this_edge:
                    sum_p += this_edge["p"]
                if "q" in this_edge:
                    sum_q += this_edge["q"]
            assert abs(sum_p) <= self.tol, f"error for node {node_id} representing bus {bus_id}: {abs(sum_p)} != 0."
            assert abs(sum_q) <= self.tol, f"error for node {node_id} representing bus {bus_id}: {abs(sum_q)} != 0."
    
    def _aux_test_bus_consistent(self, graph):
        for bus_id, node_id in enumerate(graph.graph["bus_nodes_id"]):
            if len(list(graph.predecessors(node_id))):
                # if I have predecessor (so some elements are connected to me), I am connected
                assert graph.nodes[node_id]["connected"], f"error for node {node_id} representing bus {bus_id}"
            else:
                assert not graph.nodes[node_id]["connected"], f"error for node {node_id} representing bus {bus_id}"

    def _aux_test_basic_props(self, obs, graph):        
        # test loads
        for l_id in range(obs.n_load):
            node_el_id = graph.graph["load_nodes_id"][l_id]
            assert graph.nodes[node_el_id]["id"] == l_id
            if graph.nodes[node_el_id]["connected"]:
                global_bus_id = type(obs).local_bus_to_global_int(obs.load_bus[l_id], obs.load_to_subid[l_id])
                node_bus_id = graph.graph["bus_nodes_id"][global_bus_id]
                assert graph.edges[node_el_id, node_bus_id]["id"] == l_id
        
        # test gen
        for g_id in range(obs.n_gen):
            node_el_id = graph.graph["gen_nodes_id"][g_id]
            assert graph.nodes[node_el_id]["id"] == g_id
            if graph.nodes[node_el_id]["connected"]:
                global_bus_id = type(obs).local_bus_to_global_int(obs.gen_bus[g_id], obs.gen_to_subid[g_id])
                node_bus_id = graph.graph["bus_nodes_id"][global_bus_id]
                assert graph.edges[node_el_id, node_bus_id]["id"] == g_id
            
        # test line
        for l_id in range(obs.n_line):
            node_el_id = graph.graph["line_nodes_id"][l_id]
            assert graph.nodes[node_el_id]["id"] == l_id
            if graph.nodes[node_el_id]["connected"]:
                # or side
                global_bus_id = type(obs).local_bus_to_global_int(obs.line_or_bus[l_id], obs.line_or_to_subid[l_id])
                node_bus_id = graph.graph["bus_nodes_id"][global_bus_id]
                assert graph.edges[node_el_id, node_bus_id]["id"] == l_id
                assert graph.edges[node_el_id, node_bus_id]["side"] == "or"
                # ex side
                global_bus_id = type(obs).local_bus_to_global_int(obs.line_ex_bus[l_id], obs.line_ex_to_subid[l_id])
                node_bus_id = graph.graph["bus_nodes_id"][global_bus_id]
                assert graph.edges[node_el_id, node_bus_id]["id"] == l_id
                assert graph.edges[node_el_id, node_bus_id]["side"] == "ex"
            
        # test storage
        for s_id in range(obs.n_storage):
            node_el_id = graph.graph["storage_nodes_id"][s_id]
            assert graph.nodes[node_el_id]["id"] == s_id
            if graph.nodes[node_el_id]["connected"]:
                global_bus_id = type(obs).local_bus_to_global_int(obs.storage_bus[s_id], obs.storage_to_subid[s_id])
                node_bus_id = graph.graph["bus_nodes_id"][global_bus_id]
                assert graph.edges[node_el_id, node_bus_id]["id"] == s_id
            
        # test shunts
        for s_id in range(obs.n_shunt):
            node_el_id = graph.graph["shunt_nodes_id"][s_id]
            assert graph.nodes[node_el_id]["id"] == s_id
            if graph.nodes[node_el_id]["connected"]:
                global_bus_id = type(obs).local_bus_to_global_int(obs._shunt_bus[s_id], obs.shunt_to_subid[s_id])
                node_bus_id = graph.graph["bus_nodes_id"][global_bus_id]
                assert graph.edges[node_el_id, node_bus_id]["id"] == s_id
        
    def test_can_make(self):
        obs = self.env.reset()
        complete_graph = obs.get_elements_graph()
        cls = type(obs)
        assert len(complete_graph.nodes) == cls.n_sub + 2*cls.n_sub + cls.n_load + cls.n_gen + cls.n_line + cls.n_storage + cls.n_shunt
        self._aux_test_kcl(complete_graph)
        self._aux_test_bus_consistent(complete_graph)
        self._aux_test_basic_props(obs, complete_graph)
        
    def test_disconnected_lines(self):
        obs = self.env.reset()
        l_id = 4
        obs, reward, done, info = self.env.step(self.env.action_space({"set_line_status": [(l_id, -1)]}))
        assert not done
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            # an error in the construction of the edges
            graph = obs.get_elements_graph()
        lines_id = graph.graph['line_nodes_id']
        assert not graph.nodes[lines_id[l_id]]["connected"]
        neighbors = list(networkx.neighbors(graph, lines_id[4]))
        assert not neighbors
        self._aux_test_kcl(graph)
        self._aux_test_bus_consistent(graph)
        self._aux_test_basic_props(obs, graph)
    
    def test_topo(self):
        obs = self.env.reset()
        sub_id, topo = self._aux_get_topo_action()
        action = self.env.action_space({"set_bus": {"substations_id": [(sub_id, topo)]}})
        obs, reward, done, info = self.env.step(action)
        assert not done
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            # an error in the construction of the edges
            graph = obs.get_elements_graph()
        # correct bus are connected
        bus_ids = graph.graph["bus_nodes_id"]
        assert graph.nodes[bus_ids[sub_id]]["connected"]
        assert graph.nodes[bus_ids[sub_id + obs.n_sub]]["connected"]
        # check some other things
        self._aux_test_kcl(graph)
        self._aux_test_bus_consistent(graph)
        self._aux_test_basic_props(obs, graph)
    
    def test_game_over(self):
        obs = self.env.reset()
        action = self.env.action_space({"set_bus": {"loads_id": [(0, -1)]}})
        obs, reward, done, info = self.env.step(action)
        assert done
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            # an error in the construction of the edges
            graph = obs.get_elements_graph()
        # no edges
        assert len(graph.edges) == 2 * self.env.n_sub  # each node to its substation
        # every element disconnected
        for el in range(obs.n_sub, len(graph.nodes)):
            assert not graph.nodes[el]["connected"], f"node {el} should be disconnected"
        
        
class TestElementsGraph14Storage(TestElementsGraph14SandBox):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("educ_case14_storage", test=True, action_class=PlayableAction, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
        self.tol = 1e-5
        
    def _aux_get_topo_action(self):
        sub_id = 4
        topo = (1, 2, 1, 2, 1)
        return sub_id, topo
        
        
class TestElementsGraph118(TestElementsGraph14SandBox):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_wcci_2022", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
        self.tol = 3e-5
        
    def _aux_get_topo_action(self):
        sub_id = 4
        topo = (1, 2, 1, 2, 1)
        return sub_id, topo


if __name__ == "__main__":
    unittest.main()
