# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import unittest
import warnings
from abc import ABC, abstractmethod
import numpy as np

import grid2op
from grid2op.Action import *
from grid2op.dtypes import dt_float

import warnings

warnings.simplefilter("error")


class Test_iadd_Base(ABC):
    @abstractmethod
    def _action_setup(self):
        pass

    def _skipMissingKey(self, key):
        if key not in self.action_t.authorized_keys:
            skip_msg = f"Skipped: Missing authorized_key {key}"
            unittest.TestCase.skipTest(self, skip_msg)

    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.action_t = cls._action_setup()
            cls.env = grid2op.make(
                "rte_case14_realistic", test=True, action_class=cls.action_t,
                _add_to_name=cls.__name__
            )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_dn_iadd_dn(self):
        # No skip for do nothing
        # No skip for do nothing

        # Create action me [dn]
        act_me = self.env.action_space({})

        # Test action me [dn]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [dn]
        act_oth = self.env.action_space({})

        # Test action oth [dn]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_dn_iadd_set_line(self):
        # No skip for do nothing
        self._skipMissingKey("set_line_status")

        # Create action me [dn]
        act_me = self.env.action_space({})

        # Test action me [dn]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_line]
        act_oth = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action oth [set_line]
        assert act_oth._set_line_status[0] == -1
        assert np.all(act_oth._set_line_status[1:] == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_dn_iadd_change_line(self):
        # No skip for do nothing
        self._skipMissingKey("change_line_status")

        # Create action me [dn]
        act_me = self.env.action_space({})

        # Test action me [dn]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [change_line]
        act_oth = self.env.action_space({"change_line_status": [0]})

        # Test action oth [change_line]
        assert np.all(act_oth._set_line_status == 0)
        assert act_oth._switch_line_status[0] == True
        assert np.all(act_oth._switch_line_status[1:] == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert np.all(act_me._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_dn_iadd_set_bus(self):
        # No skip for do nothing
        self._skipMissingKey("set_bus")

        # Create action me [dn]
        act_me = self.env.action_space({})

        # Test action me [dn]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_bus]
        act_oth = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action oth [set_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert act_oth._set_topo_vect[0] == 2
        assert np.all(act_oth._set_topo_vect[1:] == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_oth._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_dn_iadd_change_bus(self):
        # No skip for do nothing
        self._skipMissingKey("change_bus")

        # Create action me [dn]
        act_me = self.env.action_space({})

        # Test action me [dn]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [change_bus]
        act_oth = self.env.action_space(
            {
                "change_bus": {
                    "substations_id": [
                        (0, [True] + [False] * (self.env.sub_info[0] - 1))
                    ]
                }
            }
        )

        # Test action oth [change_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert act_oth._change_bus_vect[0] == True
        assert np.all(act_oth._change_bus_vect[1:] == False)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_me._change_bus_vect[0] == True
        assert np.all(act_me._change_bus_vect[1:] == False)
        assert np.all(act_me._redispatch == 0.0)

    def test_dn_iadd_redisp(self):
        # No skip for do nothing
        self._skipMissingKey("redispatch")

        # Create action me [dn]
        act_me = self.env.action_space({})

        # Test action me [dn]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [redisp]
        act_oth = self.env.action_space({"redispatch": {2: 1.42}})

        # Test action oth [redisp]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert act_oth._redispatch[2] == dt_float(1.42)
        assert np.all(act_oth._redispatch[:2] == 0.0)
        assert np.all(act_oth._redispatch[3:] == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_me._redispatch[2] == dt_float(1.42)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)

    def test_set_line_iadd_dn(self):
        self._skipMissingKey("set_line_status")
        # No skip for do nothing

        # Create action me [set_line]
        act_me = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action me [set_line]
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [dn]
        act_oth = self.env.action_space({})

        # Test action oth [dn]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_line_iadd_set_line(self):
        self._skipMissingKey("set_line_status")
        self._skipMissingKey("set_line_status")

        # Create action me [set_line]
        act_me = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action me [set_line]
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_line]
        act_oth = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action oth [set_line]
        assert act_oth._set_line_status[0] == -1
        assert np.all(act_oth._set_line_status[1:] == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_line_iadd_set_line2(self):
        self._skipMissingKey("set_line_status")
        self._skipMissingKey("set_line_status")

        # Create action me [set_line]
        act_me = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action me [set_line]
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_line]
        act_oth = self.env.action_space({"set_line_status": [(1, -1)]})

        # Test action oth [set_line]
        assert act_oth._set_line_status[0] == 0
        assert act_oth._set_line_status[1] == -1
        assert np.all(act_oth._set_line_status[2:] == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_me._set_line_status[0] == -1
        assert act_me._set_line_status[1] == -1
        assert np.all(act_me._set_line_status[2:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_line_iadd_set_line3(self):
        self._skipMissingKey("set_line_status")
        self._skipMissingKey("set_line_status")

        # Create action me [set_line]
        act_me = self.env.action_space({"set_line_status": [(0, 1)]})

        # Test action me [set_line]
        assert act_me._set_line_status[0] == 1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_line]
        act_oth = self.env.action_space({"set_line_status": [(1, -1)]})

        # Test action oth [set_line]
        assert act_oth._set_line_status[0] == 0
        assert act_oth._set_line_status[1] == -1
        assert np.all(act_oth._set_line_status[2:] == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_me._set_line_status[0] == 1
        assert act_me._set_line_status[1] == -1
        assert np.all(act_me._set_line_status[2:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_line_iadd_set_line4(self):
        self._skipMissingKey("set_line_status")
        self._skipMissingKey("set_line_status")

        # Create action me [set_line]
        act_me = self.env.action_space({"set_line_status": [(0, 1)]})

        # Test action me [set_line]
        assert act_me._set_line_status[0] == 1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_line]
        act_oth = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action oth [set_line]
        assert act_oth._set_line_status[0] == -1
        assert np.all(act_oth._set_line_status[1:] == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_line_iadd_change_line(self):
        self._skipMissingKey("set_line_status")
        self._skipMissingKey("change_line_status")

        # Create action me [set_line]
        act_me = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action me [set_line]
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [change_line]
        act_oth = self.env.action_space({"change_line_status": [0]})

        # Test action oth [change_line]
        assert np.all(act_oth._set_line_status == 0)
        assert act_oth._switch_line_status[0] == True
        assert np.all(act_oth._switch_line_status[1:] == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        # set + change = inverted change
        assert act_me._set_line_status[0] == 1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_line_iadd_set_bus(self):
        self._skipMissingKey("set_line_status")
        self._skipMissingKey("set_bus")

        # Create action me [set_line]
        act_me = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action me [set_line]
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_bus]
        act_oth = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action oth [set_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert act_oth._set_topo_vect[0] == 2
        assert np.all(act_oth._set_topo_vect[1:] == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_oth._set_topo_vect[0] == 2
        assert np.all(act_oth._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_line_iadd_change_bus(self):
        self._skipMissingKey("set_line_status")
        self._skipMissingKey("change_bus")

        # Create action me [set_line]
        act_me = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action me [set_line]
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [change_bus]
        act_oth = self.env.action_space(
            {
                "change_bus": {
                    "substations_id": [
                        (0, [True] + [False] * (self.env.sub_info[0] - 1))
                    ]
                }
            }
        )

        # Test action oth [change_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert act_oth._change_bus_vect[0] == True
        assert np.all(act_oth._change_bus_vect[1:] == False)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_oth._change_bus_vect[0] == True
        assert np.all(act_oth._change_bus_vect[1:] == False)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_line_iadd_redisp(self):
        self._skipMissingKey("set_line_status")
        self._skipMissingKey("redispatch")

        # Create action me [set_line]
        act_me = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action me [set_line]
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [redisp]
        act_oth = self.env.action_space({"redispatch": {2: 1.42}})

        # Test action oth [redisp]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert act_oth._redispatch[2] == dt_float(1.42)
        assert np.all(act_oth._redispatch[:2] == 0.0)
        assert np.all(act_oth._redispatch[3:] == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_oth._redispatch[2] == dt_float(1.42)
        assert np.all(act_oth._redispatch[:2] == 0.0)
        assert np.all(act_oth._redispatch[3:] == 0.0)

    def test_change_line_iadd_dn(self):
        self._skipMissingKey("change_line_status")
        # No skip for do nothing

        # Create action me [change_line]
        act_me = self.env.action_space({"change_line_status": [0]})

        # Test action me [change_line]
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert np.all(act_me._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [dn]
        act_oth = self.env.action_space({})

        # Test action oth [dn]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert np.all(act_me._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_change_line_iadd_set_line(self):
        self._skipMissingKey("change_line_status")
        self._skipMissingKey("set_line_status")

        # Create action me [change_line]
        act_me = self.env.action_space({"change_line_status": [0]})

        # Test action me [change_line]
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert np.all(act_me._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_line]
        act_oth = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action oth [set_line]
        assert act_oth._set_line_status[0] == -1
        assert np.all(act_oth._set_line_status[1:] == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_me._set_line_status[0] == -1
        assert np.all(act_me._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_change_line_iadd_change_line(self):
        self._skipMissingKey("change_line_status")
        self._skipMissingKey("change_line_status")

        # Create action me [change_line]
        act_me = self.env.action_space({"change_line_status": [0]})

        # Test action me [change_line]
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert np.all(act_me._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [change_line]
        act_oth = self.env.action_space({"change_line_status": [0]})

        # Test action oth [change_line]
        assert np.all(act_oth._set_line_status == 0)
        assert act_oth._switch_line_status[0] == True
        assert np.all(act_oth._switch_line_status[1:] == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_change_line_iadd_change_line2(self):
        self._skipMissingKey("change_line_status")
        self._skipMissingKey("change_line_status")

        # Create action me [change_line]
        act_me = self.env.action_space({"change_line_status": [0]})

        # Test action me [change_line]
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert np.all(act_me._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [change_line]
        act_oth = self.env.action_space({"change_line_status": [3]})

        # Test action oth [change_line]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status[:3] == False)
        assert act_oth._switch_line_status[3] == True
        assert np.all(act_oth._switch_line_status[4:] == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert act_me._switch_line_status[3] == True
        assert np.all(act_me._switch_line_status[4:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_change_line_iadd_set_bus(self):
        self._skipMissingKey("change_line_status")
        self._skipMissingKey("set_bus")

        # Create action me [change_line]
        act_me = self.env.action_space({"change_line_status": [0]})

        # Test action me [change_line]
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert np.all(act_me._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_bus]
        act_oth = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action oth [set_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert act_oth._set_topo_vect[0] == 2
        assert np.all(act_oth._set_topo_vect[1:] == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert np.all(act_me._switch_line_status[1:] == False)
        assert act_oth._set_topo_vect[0] == 2
        assert np.all(act_oth._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_change_line_iadd_change_bus(self):
        self._skipMissingKey("change_line_status")
        self._skipMissingKey("change_bus")

        # Create action me [change_line]
        act_me = self.env.action_space({"change_line_status": [0]})

        # Test action me [change_line]
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert np.all(act_me._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [change_bus]
        act_oth = self.env.action_space(
            {
                "change_bus": {
                    "substations_id": [
                        (0, [True] + [False] * (self.env.sub_info[0] - 1))
                    ]
                }
            }
        )

        # Test action oth [change_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert act_oth._change_bus_vect[0] == True
        assert np.all(act_oth._change_bus_vect[1:] == False)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert np.all(act_me._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_oth._change_bus_vect[0] == True
        assert np.all(act_oth._change_bus_vect[1:] == False)
        assert np.all(act_me._redispatch == 0.0)

    def test_change_line_iadd_redisp(self):
        self._skipMissingKey("change_line_status")
        self._skipMissingKey("redispatch")

        # Create action me [change_line]
        act_me = self.env.action_space({"change_line_status": [0]})

        # Test action me [change_line]
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert np.all(act_me._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [redisp]
        act_oth = self.env.action_space({"redispatch": {2: 1.42}})

        # Test action oth [redisp]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert act_oth._redispatch[2] == dt_float(1.42)
        assert np.all(act_oth._redispatch[:2] == 0.0)
        assert np.all(act_oth._redispatch[3:] == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert act_me._switch_line_status[0] == True
        assert np.all(act_me._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_oth._redispatch[2] == dt_float(1.42)
        assert np.all(act_oth._redispatch[:2] == 0.0)
        assert np.all(act_oth._redispatch[3:] == 0.0)

    def test_set_bus_iadd_dn(self):
        self._skipMissingKey("set_bus")
        # No skip for do nothing

        # Create action me [set_bus]
        act_me = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action me [set_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [dn]
        act_oth = self.env.action_space({})

        # Test action oth [dn]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_bus_iadd_set_line(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("set_line_status")

        # Create action me [set_bus]
        act_me = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action me [set_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_line]
        act_oth = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action oth [set_line]
        assert act_oth._set_line_status[0] == -1
        assert np.all(act_oth._set_line_status[1:] == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_oth._set_line_status[0] == -1
        assert np.all(act_oth._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_bus_iadd_change_line(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("change_line_status")

        # Create action me [set_bus]
        act_me = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action me [set_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [change_line]
        act_oth = self.env.action_space({"change_line_status": [0]})

        # Test action oth [change_line]
        assert np.all(act_oth._set_line_status == 0)
        assert act_oth._switch_line_status[0] == True
        assert np.all(act_oth._switch_line_status[1:] == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert act_oth._switch_line_status[0] == True
        assert np.all(act_oth._switch_line_status[1:] == False)
        assert act_me._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_bus_iadd_set_bus(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("set_bus")

        # Create action me [set_bus]
        act_me = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action me [set_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_bus]
        act_oth = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action oth [set_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert act_oth._set_topo_vect[0] == 2
        assert np.all(act_oth._set_topo_vect[1:] == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_bus_iadd_set_bus2(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("set_bus")

        # Create action me [set_bus]
        act_me = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action me [set_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_bus]
        act_oth = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [1] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action oth [set_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert act_oth._set_topo_vect[0] == 1
        assert np.all(act_oth._set_topo_vect[1:] == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 1
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_bus_iadd_change_bus(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("change_bus")

        # Create action me [set_bus]
        act_me = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action me [set_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [change_bus]
        act_oth = self.env.action_space(
            {
                "change_bus": {
                    "substations_id": [
                        (0, [True] + [False] * (self.env.sub_info[0] - 1))
                    ]
                }
            }
        )

        # Test action oth [change_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert act_oth._change_bus_vect[0] == True
        assert np.all(act_oth._change_bus_vect[1:] == False)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        # Set + change = inverted
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 1
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == False)
        assert np.all(act_me._redispatch == 0.0)

    def test_set_bus_iadd_redisp(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("redispatch")

        # Create action me [set_bus]
        act_me = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action me [set_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [redisp]
        act_oth = self.env.action_space({"redispatch": {2: 1.42}})

        # Test action oth [redisp]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert act_oth._redispatch[2] == dt_float(1.42)
        assert np.all(act_oth._redispatch[:2] == 0.0)
        assert np.all(act_oth._redispatch[3:] == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_me._set_topo_vect[0] == 2
        assert np.all(act_me._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_oth._redispatch[2] == dt_float(1.42)
        assert np.all(act_oth._redispatch[:2] == 0.0)
        assert np.all(act_oth._redispatch[3:] == 0.0)

    def test_change_bus_iadd_dn(self):
        self._skipMissingKey("change_bus")
        # No skip for do nothing

        # Create action me [change_bus]
        act_me = self.env.action_space(
            {
                "change_bus": {
                    "substations_id": [
                        (0, [True] + [False] * (self.env.sub_info[0] - 1))
                    ]
                }
            }
        )

        # Test action me [change_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_me._change_bus_vect[0] == True
        assert np.all(act_me._change_bus_vect[1:] == False)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [dn]
        act_oth = self.env.action_space({})

        # Test action oth [dn]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_me._change_bus_vect[0] == True
        assert np.all(act_me._change_bus_vect[1:] == False)
        assert np.all(act_me._redispatch == 0.0)

    def test_change_bus_iadd_set_line(self):
        self._skipMissingKey("change_bus")
        self._skipMissingKey("set_line_status")

        # Create action me [change_bus]
        act_me = self.env.action_space(
            {
                "change_bus": {
                    "substations_id": [
                        (0, [True] + [False] * (self.env.sub_info[0] - 1))
                    ]
                }
            }
        )

        # Test action me [change_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_me._change_bus_vect[0] == True
        assert np.all(act_me._change_bus_vect[1:] == False)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_line]
        act_oth = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action oth [set_line]
        assert act_oth._set_line_status[0] == -1
        assert np.all(act_oth._set_line_status[1:] == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_oth._set_line_status[0] == -1
        assert np.all(act_oth._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_me._change_bus_vect[0] == True
        assert np.all(act_me._change_bus_vect[1:] == False)
        assert np.all(act_me._redispatch == 0.0)

    def test_change_bus_iadd_change_line(self):
        self._skipMissingKey("change_bus")
        self._skipMissingKey("change_line_status")

        # Create action me [change_bus]
        act_me = self.env.action_space(
            {
                "change_bus": {
                    "substations_id": [
                        (0, [True] + [False] * (self.env.sub_info[0] - 1))
                    ]
                }
            }
        )

        # Test action me [change_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_me._change_bus_vect[0] == True
        assert np.all(act_me._change_bus_vect[1:] == False)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [change_line]
        act_oth = self.env.action_space({"change_line_status": [0]})

        # Test action oth [change_line]
        assert np.all(act_oth._set_line_status == 0)
        assert act_oth._switch_line_status[0] == True
        assert np.all(act_oth._switch_line_status[1:] == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert act_oth._switch_line_status[0] == True
        assert np.all(act_oth._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_me._change_bus_vect[0] == True
        assert np.all(act_me._change_bus_vect[1:] == False)
        assert np.all(act_me._redispatch == 0.0)

    def test_change_bus_iadd_set_bus(self):
        self._skipMissingKey("change_bus")
        self._skipMissingKey("set_bus")

        # Create action me [change_bus]
        act_me = self.env.action_space(
            {
                "change_bus": {
                    "substations_id": [
                        (0, [True] + [False] * (self.env.sub_info[0] - 1))
                    ]
                }
            }
        )

        # Test action me [change_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_me._change_bus_vect[0] == True
        assert np.all(act_me._change_bus_vect[1:] == False)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [set_bus]
        act_oth = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action oth [set_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert act_oth._set_topo_vect[0] == 2
        assert np.all(act_oth._set_topo_vect[1:] == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_oth._set_topo_vect[0] == 2
        assert np.all(act_oth._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == False)
        assert np.all(act_me._redispatch == 0.0)

    def test_change_bus_iadd_change_bus(self):
        self._skipMissingKey("change_bus")
        self._skipMissingKey("change_bus")

        # Create action me [change_bus]
        act_me = self.env.action_space(
            {
                "change_bus": {
                    "substations_id": [
                        (0, [True] + [False] * (self.env.sub_info[0] - 1))
                    ]
                }
            }
        )

        # Test action me [change_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_me._change_bus_vect[0] == True
        assert np.all(act_me._change_bus_vect[1:] == False)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [change_bus]
        act_oth = self.env.action_space(
            {
                "change_bus": {
                    "substations_id": [
                        (0, [True] + [False] * (self.env.sub_info[0] - 1))
                    ]
                }
            }
        )

        # Test action oth [change_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert act_oth._change_bus_vect[0] == True
        assert np.all(act_oth._change_bus_vect[1:] == False)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == False)
        assert np.all(act_me._redispatch == 0.0)

    def test_change_bus_iadd_redisp(self):
        self._skipMissingKey("change_bus")
        self._skipMissingKey("redispatch")

        # Create action me [change_bus]
        act_me = self.env.action_space(
            {
                "change_bus": {
                    "substations_id": [
                        (0, [True] + [False] * (self.env.sub_info[0] - 1))
                    ]
                }
            }
        )

        # Test action me [change_bus]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_me._change_bus_vect[0] == True
        assert np.all(act_me._change_bus_vect[1:] == False)
        assert np.all(act_me._redispatch == 0.0)

        # Create action oth [redisp]
        act_oth = self.env.action_space({"redispatch": {2: 1.42}})

        # Test action oth [redisp]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert act_oth._redispatch[2] == dt_float(1.42)
        assert np.all(act_oth._redispatch[:2] == 0.0)
        assert np.all(act_oth._redispatch[3:] == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_me._change_bus_vect[0] == True
        assert np.all(act_me._change_bus_vect[1:] == False)
        assert act_oth._redispatch[2] == dt_float(1.42)
        assert np.all(act_oth._redispatch[:2] == 0.0)
        assert np.all(act_oth._redispatch[3:] == 0.0)

    def test_redisp_iadd_dn(self):
        self._skipMissingKey("redispatch")
        # No skip for do nothing

        # Create action me [redisp]
        act_me = self.env.action_space({"redispatch": {2: 1.42}})

        # Test action me [redisp]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_me._redispatch[2] == dt_float(1.42)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)

        # Create action oth [dn]
        act_oth = self.env.action_space({})

        # Test action oth [dn]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_me._redispatch[2] == dt_float(1.42)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)

    def test_redisp_iadd_set_line(self):
        self._skipMissingKey("redispatch")
        self._skipMissingKey("set_line_status")

        # Create action me [redisp]
        act_me = self.env.action_space({"redispatch": {2: 1.42}})

        # Test action me [redisp]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_me._redispatch[2] == dt_float(1.42)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)

        # Create action oth [set_line]
        act_oth = self.env.action_space({"set_line_status": [(0, -1)]})

        # Test action oth [set_line]
        assert act_oth._set_line_status[0] == -1
        assert np.all(act_oth._set_line_status[1:] == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert act_oth._set_line_status[0] == -1
        assert np.all(act_oth._set_line_status[1:] == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_me._redispatch[2] == dt_float(1.42)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)

    def test_redisp_iadd_change_line(self):
        self._skipMissingKey("redispatch")
        self._skipMissingKey("change_line_status")

        # Create action me [redisp]
        act_me = self.env.action_space({"redispatch": {2: 1.42}})

        # Test action me [redisp]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_me._redispatch[2] == dt_float(1.42)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)

        # Create action oth [change_line]
        act_oth = self.env.action_space({"change_line_status": [0]})

        # Test action oth [change_line]
        assert np.all(act_oth._set_line_status == 0)
        assert act_oth._switch_line_status[0] == True
        assert np.all(act_oth._switch_line_status[1:] == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert act_oth._switch_line_status[0] == True
        assert np.all(act_oth._switch_line_status[1:] == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_me._redispatch[2] == dt_float(1.42)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)

    def test_redisp_iadd_set_bus(self):
        self._skipMissingKey("redispatch")
        self._skipMissingKey("set_bus")

        # Create action me [redisp]
        act_me = self.env.action_space({"redispatch": {2: 1.42}})

        # Test action me [redisp]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_me._redispatch[2] == dt_float(1.42)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)

        # Create action oth [set_bus]
        act_oth = self.env.action_space(
            {
                "set_bus": {
                    "substations_id": [(0, [2] + [0] * (self.env.sub_info[0] - 1))]
                }
            }
        )

        # Test action oth [set_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert act_oth._set_topo_vect[0] == 2
        assert np.all(act_oth._set_topo_vect[1:] == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert act_oth._set_topo_vect[0] == 2
        assert np.all(act_oth._set_topo_vect[1:] == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_me._redispatch[2] == dt_float(1.42)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)

    def test_redisp_iadd_change_bus(self):
        self._skipMissingKey("redispatch")
        self._skipMissingKey("change_bus")

        # Create action me [redisp]
        act_me = self.env.action_space({"redispatch": {2: 1.42}})

        # Test action me [redisp]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_me._redispatch[2] == dt_float(1.42)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)

        # Create action oth [change_bus]
        act_oth = self.env.action_space(
            {
                "change_bus": {
                    "substations_id": [
                        (0, [True] + [False] * (self.env.sub_info[0] - 1))
                    ]
                }
            }
        )

        # Test action oth [change_bus]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert act_oth._change_bus_vect[0] == True
        assert np.all(act_oth._change_bus_vect[1:] == False)
        assert np.all(act_oth._redispatch == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert act_oth._change_bus_vect[0] == True
        assert np.all(act_oth._change_bus_vect[1:] == False)
        assert act_me._redispatch[2] == dt_float(1.42)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)

    def test_redisp_iadd_redisp(self):
        self._skipMissingKey("redispatch")
        self._skipMissingKey("redispatch")

        # Create action me [redisp]
        act_me = self.env.action_space({"redispatch": {2: 1.42}})

        # Test action me [redisp]
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_me._redispatch[2] == dt_float(1.42)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)

        # Create action oth [redisp]
        act_oth = self.env.action_space({"redispatch": {2: 1.42}})

        # Test action oth [redisp]
        assert np.all(act_oth._set_line_status == 0)
        assert np.all(act_oth._switch_line_status == False)
        assert np.all(act_oth._set_topo_vect == 0)
        assert np.all(act_oth._change_bus_vect == 0)
        assert act_oth._redispatch[2] == dt_float(1.42)
        assert np.all(act_oth._redispatch[:2] == 0.0)
        assert np.all(act_oth._redispatch[3:] == 0.0)

        # Iadd actions
        act_me += act_oth

        # Test combination:
        assert np.all(act_me._set_line_status == 0)
        assert np.all(act_me._switch_line_status == False)
        assert np.all(act_me._set_topo_vect == 0)
        assert np.all(act_me._change_bus_vect == 0)
        assert act_me._redispatch[2] == dt_float(1.42 * 2.0)
        assert np.all(act_me._redispatch[:2] == 0.0)
        assert np.all(act_me._redispatch[3:] == 0.0)


class Test_iadd_CompleteAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: CompleteAction
    """

    @classmethod
    def _action_setup(self):
        return CompleteAction


class Test_iadd_DispatchAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: DispatchAction
    """

    @classmethod
    def _action_setup(self):
        return DispatchAction


class Test_iadd_DontAct(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: DontAct
    """

    @classmethod
    def _action_setup(self):
        return DontAct


class Test_iadd_PlayableAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: PlayableAction
    """

    @classmethod
    def _action_setup(self):
        return PlayableAction


class Test_iadd_PowerlineChangeAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: PowerlineChangeAction
    """

    @classmethod
    def _action_setup(self):
        return PowerlineChangeAction


class Test_iadd_PowerlineChangeAndDispatchAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: PowerlineChangeAndDispatchAction
    """

    @classmethod
    def _action_setup(self):
        return PowerlineChangeAndDispatchAction


class Test_iadd_PowerlineSetAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: PowerlineSetAction
    """

    @classmethod
    def _action_setup(self):
        return PowerlineSetAction


class Test_iadd_PowerlineSetAndDispatchAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: PowerlineSetAndDispatchAction
    """

    @classmethod
    def _action_setup(self):
        return PowerlineSetAndDispatchAction


class Test_iadd_TopologyAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: TopologyAction
    """

    @classmethod
    def _action_setup(self):
        return TopologyAction


class Test_iadd_TopologyAndDispatchAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: TopologyAndDispatchAction
    """

    @classmethod
    def _action_setup(self):
        return TopologyAndDispatchAction


class Test_iadd_TopologyChangeAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: TopologyChangeAction
    """

    @classmethod
    def _action_setup(self):
        return TopologyChangeAction


class Test_iadd_TopologyChangeAndDispatchAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: TopologyChangeAndDispatchAction
    """

    @classmethod
    def _action_setup(self):
        return TopologyChangeAndDispatchAction


class Test_iadd_TopologySetAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: TopologySetAction
    """

    @classmethod
    def _action_setup(self):
        return TopologySetAction


class Test_iadd_TopologySetAndDispatchAction(Test_iadd_Base, unittest.TestCase):
    """
    Action iadd test suite for subclass: TopologySetAndDispatchAction
    """

    @classmethod
    def _action_setup(self):
        return TopologySetAndDispatchAction


if __name__ == "__main__":
    unittest.main()
