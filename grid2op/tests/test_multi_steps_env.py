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
import numpy as np
import pdb
import os

from grid2op.tests.helper_path_test import *
    
import grid2op
import numpy as np


class MultiStepsForcaTester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make(os.path.join(PATH_DATA_TEST, "5bus_example_forecasts"), test=True)
        self.env.seed(0)
        self.env.set_id(0)
        self.dn = self.env.action_space()
    
    def _check_ok(self, f_obs, obs, h):
        sim_obs, *_ = obs.simulate(self.dn, h)
        assert np.all(sim_obs.load_p == f_obs.load_p), f"error for h={h}"
        assert np.all(sim_obs.load_q == f_obs.load_q), f"error for h={h}"
        assert np.all(sim_obs.gen_p == f_obs.gen_p), f"error for h={h}"
        assert np.all(sim_obs.gen_q == f_obs.gen_q), f"error for h={h}"
        assert np.all(sim_obs.rho == f_obs.rho), f"error for h={h}"
        
    def test_when_do_nothing(self):
        obs = self.env.reset()
        forecast_env = obs.get_forecast_env()
        f_obs = forecast_env.reset()
        self._check_ok(f_obs, obs, 0)
        for h in range(12):
            f_obs, *_ = forecast_env.step(self.dn)
            self._check_ok(f_obs, obs, h + 1)
    
    def test_soft_overflow(self):
        # the forecasted env should start with the same "values" for cooldowns, soft overflows etc.
        
        # get ready for the soft overflow
        a_or_first = np.array([442.308, 198.55365, 116.50534,  93.63006,
                               442.2703 , 110.96754, 110.96754,  92.05039])
        th_lim = a_or_first * 2.
        th_lim[5] /= 2.5
        self.env.set_thermal_limit(th_lim)
        param = self.env.parameters
        param.NO_OVERFLOW_DISCONNECTION = False
        param.NB_TIMESTEP_RECONNECTION = 4
        self.env.change_parameters(param)
        self.env.change_forecast_parameters(param)
        obs = self.env.reset()
        obs, *_ = self.env.step(self.dn)
        assert obs.timestep_overflow[5] == 1
        
        forecast_env = obs.get_forecast_env()
        f_obs = forecast_env.reset()
        assert f_obs.timestep_overflow[5] == 1
        f_obs2, *_ = forecast_env.step(self.dn)
        assert f_obs2.timestep_overflow[5] == 2
        
        f_obs3, *_ = forecast_env.step(self.dn)
        assert f_obs3.timestep_overflow[5] == 0
        assert not f_obs3.line_status[5]
        assert f_obs.time_before_cooldown_line[5] == 4
        
        f_obs4, *_ = forecast_env.step(self.dn)
        assert not f_obs4.line_status[5]
        assert f_obs4.time_before_cooldown_line[5] == 3
    
    def test_cooldown(self):
        act = self.env.action_space({"set_line_status": [(5, -1)]})
        un_act = self.env.action_space({"set_line_status": [(5, +1)]})
        obs = self.env.reset()
        obs, *_ = self.env.step(act)
        assert obs.time_before_cooldown_line[5] == 3
        
        forecast_env = obs.get_forecast_env()
        f_obs = forecast_env.reset()
        assert f_obs.time_before_cooldown_line[5] == 3
        f_obs2, *_ = forecast_env.step(self.dn)
        assert f_obs2.time_before_cooldown_line[5] == 2
        f_obs3, *_ = forecast_env.step(self.dn)
        assert f_obs3.time_before_cooldown_line[5] == 1
        f_obs4, r, done, info = forecast_env.step(un_act)
        assert f_obs4.time_before_cooldown_line[5] == 0
        assert info["is_illegal"]  # because cooldown is 1 when i took the action
        
        # now I can reco
        f_obs5, r, done, info = forecast_env.step(un_act)
        assert f_obs5.time_before_cooldown_line[5] == 3
        assert not info["is_illegal"]  # because cooldown is 1 when i took the action
        assert f_obs5.line_status[5]
        
    def test_maintenance(self):     
        reco = self.env.action_space({"set_line_status": [(5, +1)]})
          
        obs = self.env.reset()  # no maintenance
        obs = self.env.reset()  # maintenance
        assert obs.time_next_maintenance[5] == 6
        assert obs.duration_next_maintenance[5] == 4
        
        forecast_env = obs.get_forecast_env()
        f_obs = forecast_env.reset()
        assert f_obs.time_next_maintenance[5] == 6
        assert f_obs.duration_next_maintenance[5] == 4
        
        f_obs1, *_ = forecast_env.step(self.dn)
        f_obs2, *_ = forecast_env.step(self.dn)
        f_obs3, *_ = forecast_env.step(self.dn)
        f_obs4, *_ = forecast_env.step(self.dn)
        f_obs5, *_ = forecast_env.step(self.dn)
        
        f_obs6, *_ = forecast_env.step(self.dn)
        assert f_obs6.time_next_maintenance[5] == 0
        assert f_obs6.duration_next_maintenance[5] == 4
        assert f_obs6.time_before_cooldown_line[5] == 4
        assert not f_obs6.line_status[5]
        
        f_obs7, *_ = forecast_env.step(self.dn)
        assert f_obs7.time_next_maintenance[5] == 0
        assert f_obs7.duration_next_maintenance[5] == 3
        assert f_obs7.time_before_cooldown_line[5] == 3
        assert not f_obs7.line_status[5]
        
        f_obs8, *_ = forecast_env.step(self.dn)
        f_obs9, *_ = forecast_env.step(self.dn)
        
        # I cannot reco yet
        f_obs10, r, d, info = forecast_env.step(reco)
        assert f_obs10.time_next_maintenance[5] == -1
        assert f_obs10.duration_next_maintenance[5] == 0
        assert f_obs10.time_before_cooldown_line[5] == 0
        assert not f_obs10.line_status[5]
        assert info["is_illegal"]
        
        f_obs11, r, d, info = forecast_env.step(reco)
        assert f_obs11.time_next_maintenance[5] == -1
        assert f_obs11.duration_next_maintenance[5] == 0
        assert f_obs11.time_before_cooldown_line[5] == 3  # because i could act
        assert f_obs11.line_status[5]
        assert not info["is_illegal"]
    
# TODO same results as simulate / as expected:
# X regurlarly (do nothing)
#   X regurlarly (standard) 
#   X maintenance
#   X cooldown
#   X soft overflow
# > with some actions (carefull chain of simulate !)

# TODO test that backend is properly copied
# TODO test that whatever I do in the simulated env it has no impact "on the reality"             
# TODO test that whatever I do in the simulated env it has no impact "on simulate"
             
if __name__ == "__main__":
    unittest.main()
