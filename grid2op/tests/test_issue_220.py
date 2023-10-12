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

import grid2op


class Issue220Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
            self.env.seed(0)
            self.env.reset()

    def test_flow_bus_matrix(self):
        obs = self.env.reset()
        res, _ = obs.flow_bus_matrix()
        assert res.shape == (14, 14)
        action = self.env.action_space()
        action.change_line_status = [9]
        # two powerlines will be disconnected: powerline 9 (action) and powerline 13 ("cascading failure")
        new_obs, reward, done, info = self.env.step(action)
        res, _ = new_obs.flow_bus_matrix()
        assert res.shape == (14, 14)
