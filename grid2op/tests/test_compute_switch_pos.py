# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import pandas as pd
import os
import numpy as np
import networkx as nx

from grid2op.tests.helper_path_test import *
import grid2op

from grid2op.Space import DetailedTopoDescription
from grid2op.Exceptions import ImpossibleTopology
import pdb


class TestComputeSwitchPos(unittest.TestCase):
    # TODO detailed topo: not tested in case of shunt
    def _aux_read_case(self, case_id):
        path_data = os.path.join(PATH_DATA_TEST, "test_detailed_topo")
        switches = pd.read_csv(os.path.join(path_data, f"test_topo_connections{case_id}.txt"),
                               sep=" ")
        elements = pd.read_csv(os.path.join(path_data, f"test_topo_elements{case_id}.txt"),
                               sep=" ")
        target_bus = pd.read_csv(os.path.join(path_data, f"test_topo_valid{case_id}.txt"),
                                 sep=" ")
        dtd = DetailedTopoDescription()
        dtd._n_sub = 1
        all_nodes = np.unique(np.concatenate((switches["node1"].values, switches["node2"].values)))
        nb_switch = switches.shape[0]
        dtd.conn_node_name = np.array([None for _ in all_nodes], dtype=str)
        dtd.conn_node_to_subid = np.zeros(len(all_nodes), dtype=int)
        dtd.switches = np.zeros((nb_switch, 3), dtype=int)
        dtd.switches[:, 0] = 0
        dtd.switches[:, 1] = switches["node1"].values
        dtd.switches[:, 2] = switches["node2"].values
        # fill the elements
        # we do as if everything is a line here
        dtd.load_to_conn_node_id = np.array([], dtype=int)
        dtd.gen_to_conn_node_id = np.array([], dtype=int)
        dtd.line_ex_to_conn_node_id = np.array([], dtype=int)
        dtd.storage_to_conn_node_id = np.array([], dtype=int)
        dtd.shunt_to_conn_node_id = np.array([], dtype=int)
        # now fill the line part
        mask_el = elements["element_id"] == "'el'"
        dtd.line_or_to_conn_node_id = elements["node"].loc[mask_el].values
        # assign the topo vect infoconn_node_to_shunt_id
        dtd.conn_node_to_topovect_id = np.zeros(len(all_nodes), dtype=int) - 1
        dtd.conn_node_to_topovect_id[dtd.line_or_to_conn_node_id] = np.arange(dtd.line_or_to_conn_node_id.shape[0])
        dtd.conn_node_to_shunt_id = np.array([])
        
        # fill the busbars
        mask_el = elements["element_id"] == "'bbs'"
        dtd.busbar_section_to_conn_node_id = elements["node"].loc[mask_el].values
        dtd.busbar_section_to_subid = np.zeros(dtd.busbar_section_to_conn_node_id.shape[0], dtype=int)
        dtd._from_ieee_grid = False
        
        # now get the results
        small_df = target_bus.loc[np.isin(target_bus["node"], dtd.line_or_to_conn_node_id)]
        results = np.zeros(dtd.line_or_to_conn_node_id.shape[0], dtype=int) -1
        for line_id in range(dtd.line_or_to_conn_node_id.shape[0]):
            results[line_id] = small_df.loc[small_df["node"] == dtd.line_or_to_conn_node_id[line_id], "bus_id"].values[0]
        results[results >= 0] += 1  # encoding starts at 0 for input data
        
        # specific because it's not checked
        dtd._dim_topo = dtd.line_or_to_conn_node_id.shape[0]
        dtd._n_shunt = 0
        dtd._n_sub = 1
        return dtd, results
    
    def setUp(self):
        super().setUp()
        
    def _aux_test_switch_topo(self, dtd, results, switches, extra_str=""):
        graph = nx.Graph()
        graph.add_edges_from([(el[1], el[2], {"id": switch_id}) for switch_id, el in enumerate(dtd.switches) if switches[switch_id]])
        tmp = list(nx.connected_components(graph))
        expected_buses = np.unique(results[results != -1])
        assert len(tmp) == expected_buses.shape[0], f"found {len(tmp)} buses when asking for {np.unique(results).shape[0]}"
        # check that element in results connected together are connected together
        # and check that the elements that are not connected together are not
        for el_1 in range(results.shape[0]):
            th_bus_1 = results[el_1]
            conn_bus_1 = dtd.line_or_to_conn_node_id[el_1]
            conn_comp1 = np.array([conn_bus_1 in el for el in tmp]).nonzero()[0]
            if th_bus_1 == -1:
                assert conn_comp1.shape[0] == 0, f"Error for element {el_1}: it should be disconnected but does not appear to be"
                continue
            for el_2 in range(el_1 + 1, results.shape[0]):
                th_bus_2 = results[el_2]
                conn_bus_2 = dtd.line_or_to_conn_node_id[el_2]
                conn_comp2 = np.array([conn_bus_2 in el for el in tmp]).nonzero()[0]
                if th_bus_2 == -1:
                    assert conn_comp2.shape[0] == 0, f"Error for element {el_2}: it should be disconnected but does not appear to be"
                elif th_bus_1 == th_bus_2:
                    # disconnected element should not be together
                    assert conn_comp1 == conn_comp2, f"Error for elements: {el_1} and {el_2}: they should be on the same bus but are not, {extra_str}"
                else:
                    assert conn_comp1 != conn_comp2, f"Error for elements: {el_1} and {el_2}: they should NOT be on the same bus but they are, {extra_str}"
                    
    def test_case1_standard(self):
        """test I can compute this for the reference test case"""
        dtd, results = self._aux_read_case("1")
        dtd._aux_compute_busbars_sections()
        switches = dtd.compute_switches_position(results)
        self._aux_test_switch_topo(dtd, results, switches)
                    
    def test_case1_all_samebus(self):
        """test I can connect every element to the same bus, even if the said bus is not 1"""
        dtd, results = self._aux_read_case("1")
        dtd._aux_compute_busbars_sections()
        for bus in range(dtd.busbar_section_to_subid.shape[0]):
            results[:] = bus + 1
            switches = dtd.compute_switches_position(results)
            self._aux_test_switch_topo(dtd, results, switches)
    
    def test_case1_impossible_toomuch_buses(self):
        """test that when someone ask to connect something to a bus too high (too many buses) then it does not work"""
        dtd, results = self._aux_read_case("1")
        dtd._aux_compute_busbars_sections()
        bus_id_too_high = dtd.busbar_section_to_subid.shape[0] + 1
        for el_id in range(len(results)):
            els = np.array(list(dtd._conn_node_to_bbs_conn_node_id[dtd.line_or_to_conn_node_id[el_id]]))
            results[el_id] = (dtd.busbar_section_to_conn_node_id == els[el_id % len(els)]).nonzero()[0][0] + 1
        # test that it works in general case with all possible buses
        switches = dtd.compute_switches_position(results)
        self._aux_test_switch_topo(dtd, results, switches)
        
        # now test that it breaks if the index of a bus it too high
        for el_id in range(len(results)):
            tmp = 1 * results
            tmp[el_id] = bus_id_too_high
            with self.assertRaises(ImpossibleTopology):
                switches = dtd.compute_switches_position(tmp)
                
    def test_case1_impossible_connectivity(self):
        """test for some more cases where it would be impossible (forced to connect busbar breaker 
        for some elements but not for others)"""
        dtd, results = self._aux_read_case("1")
        dtd._aux_compute_busbars_sections()
        results[0] = 1  # to force busbar sec 0
        results[1] = 2  # to force busbar sec 1
        results[2] = 3  # to force busbar sec 3
        results[3] = 4  # to force busbar sec 4
        results[4] = 2  # is directly connected to busbar sec 1 or 3, in this first example I force it to 1
        
        # now i force every element to a busbar to which it is directly connected
        # so as to make sure it works
        for el_id in range(4, len(results)):
            els = np.array(list(dtd._conn_node_to_bbs_conn_node_id[dtd.line_or_to_conn_node_id[el_id]]))
            results[el_id] = (dtd.busbar_section_to_conn_node_id == els[0]).nonzero()[0][0] + 1
        # should work
        switches = dtd.compute_switches_position(results)
        self._aux_test_switch_topo(dtd, results, switches)
        
        # here I force to connect bbs 1 or 3 to bbs 0
        # which contradicts the 4 other constraints above
        results[4] = 1
        with self.assertRaises(ImpossibleTopology):
            switches = dtd.compute_switches_position(results)
                
    def test_case1_with_disconnected_element(self):
        dtd, results = self._aux_read_case("1")
        dtd._aux_compute_busbars_sections()
        # disconnect element one by one and check it works
        for el_id in range(len(results)):
            tmp = 1 * results
            tmp[el_id] = -1
            switches = dtd.compute_switches_position(tmp)
            self._aux_test_switch_topo(dtd, tmp, switches, f"when disconnecting element {el_id}")
         
         
if __name__ == "__main__":
    unittest.main()
   