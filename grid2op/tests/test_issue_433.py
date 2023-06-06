# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import grid2op
import unittest
import warnings
import pdb

class Issue433Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True)
        self.env.seed(0)
        self.env.set_id(0)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_disco_2_lines(self):
        act0 = self.env.action_space({"set_line_status": [(4, -1)]})
        act1 = self.env.action_space({"set_line_status": [(13, -1)]})
        obs = self.env.reset()
        obs, *_ = self.env.step(act0)
        graph = obs.get_energy_graph()
        obs, *_ = self.env.step(act1)
        obs.get_energy_graph()  # crashed

if __name__ == '__main__':
    unittest.main()
