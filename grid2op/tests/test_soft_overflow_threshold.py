# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.Parameters import Parameters
import warnings
import unittest


class TestSoftOverflowThreshold(unittest.TestCase):    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
        th_lim = self.env.get_thermal_limit()
        th_lim[0] = 161
        self.env.set_thermal_limit(th_lim)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_default_param(self):
        """test nothing is broken, and by default it works normally"""
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        assert not obs.line_status[0] 
    
    def test_1point1_param_nodisc(self):
        """test line is NOT disconnected when its flow is bellow the threshold"""
        param = self.env.parameters
        param.SOFT_OVERFLOW_THRESHOLD = 1.1
        self.env.change_parameters(param)
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.line_status[0] 
        assert obs.timestep_overflow[0] == 3 
        assert obs.thermal_limit[0] == 161
        assert obs.a_or[0] > 161
    
    def test_1point1_param_disco(self):
        """test line is indeed disconnected when its flow is above the threshold"""
        param = self.env.parameters
        param.SOFT_OVERFLOW_THRESHOLD = 1.1
        self.env.change_parameters(param)
        th_lim = self.env.get_thermal_limit()
        th_lim[0] /= 1.1
        self.env.set_thermal_limit(th_lim)
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        assert not obs.line_status[0] 

if __name__ == '__main__':
    unittest.main()
