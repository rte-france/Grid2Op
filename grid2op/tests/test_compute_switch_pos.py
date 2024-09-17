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
import warnings
import numpy as np
import hashlib

from grid2op.tests.helper_path_test import *
import grid2op

from grid2op.dtypes import dt_bool
from grid2op.Action import BaseAction, CompleteAction
from grid2op.Observation import BaseObservation
from grid2op.Runner import Runner
from grid2op.Backend import PandaPowerBackend
from grid2op.Space import AddDetailedTopoIEEE, DetailedTopoDescription
from grid2op.Agent import BaseAgent
from grid2op.Exceptions import AmbiguousAction
import pdb
REF_HASH = 'c8296b80b3b920b2971bd82e93f998a043ccb3738f04ca0d3f23f524306da8e95109f5af27e28a85597151b3988840674f4e6ad1efa69dbab1a2174765f330ec'

class TestComputeSwitchPos(unittest.TestCase):
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
        dtd.conn_node_to_subid = np.zeros(nb_switch, dtype=int)
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
        
    def test_case1(self):
        dtd, results = self._aux_read_case("1")
        dtd._aux_compute_busbars_sections()
        switches = dtd.compute_switches_position(results)
 
if __name__ == "__main__":
    unittest.main()
   