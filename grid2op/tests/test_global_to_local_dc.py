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
import pandapower as pp
import numpy as np
import pdb


class TestBugShuntDC(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True)
            
    def tearDown(self) -> None:
        self.env.close()
        
    def test_shunt_dc(self):
        self.env.backend.runpf(is_dc=True)
        p_subs, q_subs, p_bus, q_bus, diff_v_bus = self.env.backend.check_kirchoff()
        assert np.abs(p_subs).max() <=1e-5
        assert np.abs(p_bus).max() <=1e-5
        # below it does not pass due to https://github.com/e2nIEE/pandapower/issues/1996
        # assert np.abs(diff_v_bus).max() <=1e-5
        
    def test_shunt_dc_alone(self):
        self.env.backend._grid.shunt["bus"][0] += 14
        self.env.backend._grid.bus["in_service"][self.env.backend._grid.shunt["bus"][0]] = True
        self.env.backend.runpf(is_dc=True)
        p_subs, q_subs, p_bus, q_bus, diff_v_bus = self.env.backend.check_kirchoff()
        assert np.abs(p_subs).max() <=1e-5
        assert np.abs(p_bus).max() <=1e-5
        # below it does not pass due to https://github.com/e2nIEE/pandapower/issues/1996
        # assert np.abs(diff_v_bus).max() <=1e-5
            

if __name__ == "__main__":
    unittest.main()
