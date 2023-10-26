
# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import warnings
import unittest
import pdb


class Issue527Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
            "l2rpn_case14_sandbox",
            test=True,
            _add_to_name=type(self).__name__
        )
        self.env.seed(0)
        return super().setUp()
    
    def tearDown(self):
        self.env.close()
        
    def test_action_space_sampling(self) -> None:
        obs = self.env.reset()
        for ind in range(1000):
            act = self.env.action_space.sample()
            act_dict = act.as_serializable_dict()
            self.env.action_space(act_dict)       
        
    def test_do_nothing_act_weird(self) -> None:
        obs = self.env.reset()
        self.env.action_space({"change_bus": {}})       
        self.env.action_space({"set_bus": {}})       


if __name__ == '__main__':
    unittest.main()
