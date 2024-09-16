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
        dtd = DetailedTopoDescription()
        dtd._n_sub = 1
        all_nodes = np.unique(np.concatenate((switches["node1"].values, switches["node2"].values)))
        nb_switch = switches.shape[0]
        dtd.conn_node_name = [None for _ in all_nodes]
        dtd.conn_node_to_subid = np.zeros(nb_switch, dtype=int)
        dtd.switches = np.zeros((nb_switch, 4), dtype=int)
        dtd.switches[:, 0] = 0
        # dtd.switches[:, 1] = 
        dtd.switches[:, 2] = switches["node1"].values
        dtd.switches[:, 3] = switches["node2"].values
        dtd.switches_to_topovect_id = np.zeros(nb_switch, dtype=int) - 1
        dtd.switches_to_shunt_id = np.zeros(nb_switch, dtype=int) - 1
        dtd.load_to_conn_node_id = np.array([38, 39, 40])  # TODO
        dtd.switches_to_topovect_id[dtd.load_to_conn_node_id] = np.arange(dtd.load_to_conn_node_id.shape[0])
        # dtd.gen_to_conn_node_id
        # dtd.line_or_to_conn_node_id
        # dtd.line_ex_to_conn_node_id
        # dtd.storage_to_conn_node_id
        # dtd.shunt_to_conn_node_id
        dtd.busbar_section_to_conn_node_id = np.array([0, 1, 2, 3])   # TODO
        dtd.busbar_section_to_subid = np.zeros(dtd.busbar_section_to_conn_node_id.shape[0], dtype=int)
        dtd._from_ieee_grid = False
        return dtd
    def setUp(self):
        super().setUp()
        
    def test_case1(self):
        dtd = self._aux_read_case("1")
        switches = dtd.compute_switches_position(np.array([1, 1, 2]))
 
if __name__ == "__main__":
    unittest.main()
   