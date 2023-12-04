# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import grid2op
from grid2op.Action import CompleteAction
from grid2op.Backend import PandaPowerBackend
from lightsim2grid import LightSimBackend
import pandapower as pp
import numpy as np
import pdb


class AuxTestBugShuntDC:
    def get_backend(self):
        raise NotImplementedError()
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox",
                                    test=True,
                                    action_class=CompleteAction,
                                    backend=self.get_backend(),
                                    _add_to_name=type(self).__name__)
        self.bk_act = type(self.env.backend).my_bk_act_class()
        
    def tearDown(self) -> None:
        self.env.close()
    
    def _aux_modify_shunt(self):
        self.bk_act += self.env.action_space({"shunt": {"set_bus": np.array([2], dtype=int)}})
        self.env.backend.apply_action(self.bk_act)
        if isinstance(self.env.backend, PandaPowerBackend):
            assert self.env.backend._grid.shunt["bus"][0] == 22
            assert self.env.backend._grid.bus["in_service"][self.env.backend._grid.shunt["bus"][0]]
            
    def test_shunt_dc(self):
        conv, exc_ = self.env.backend.runpf(is_dc=True)
        p_subs, q_subs, p_bus, q_bus, diff_v_bus = self.env.backend.check_kirchoff()
        assert np.abs(p_subs).max() <= 1e-5
        assert np.abs(p_bus).max() <= 1e-5
        # below it does not pass due to https://github.com/e2nIEE/pandapower/issues/1996 (fixed !)
        assert np.abs(diff_v_bus).max() <= 1e-5
        
    def test_shunt_alone_dc(self):
        self._aux_modify_shunt()
        conv, exc_ = self.env.backend.runpf(is_dc=True)
        assert not conv
        # does not work now because of an isolated element
        # p_subs, q_subs, p_bus, q_bus, diff_v_bus = self.env.backend.check_kirchoff()
        # assert np.abs(p_subs).max() <= 1e-5
        # assert np.abs(p_bus).max() <= 1e-5
        # # below it does not pass due to https://github.com/e2nIEE/pandapower/issues/1996
        # assert np.abs(diff_v_bus).max() <= 1e-5
 
    def test_shunt_alone_ac(self):
        self._aux_modify_shunt()
        conv, exc_ = self.env.backend.runpf(is_dc=False)
        assert not conv
            

class TestBugShuntDCPP(AuxTestBugShuntDC, unittest.TestCase):
    def get_backend(self):
        return PandaPowerBackend()
    

class TestBugShuntDCLS(AuxTestBugShuntDC, unittest.TestCase):
    def get_backend(self):
        return LightSimBackend()
    
    
if __name__ == "__main__":
    unittest.main()
