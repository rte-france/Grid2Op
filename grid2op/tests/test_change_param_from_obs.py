# Copyright (c) 2023, RTE (https://www.rte-france.com)
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


class TestChangeParamFromObs(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def test_change_param_simulate(self):
        l_id = 3
        params = self.env.parameters
        params.HARD_OVERFLOW_THRESHOLD = 1.001
        th_lim = self.env.get_thermal_limit()
        th_lim[l_id] = 170
        self.env.set_thermal_limit(th_lim)
        self.env.change_parameters(params)
        self.env.change_forecast_parameters(params)
        obs = self.env.reset()
        # line 0 is connected
        assert obs.rho[l_id] > 0
        assert obs.line_status[0]
        
        # changes not done
        assert not obs._obs_env.parameters.NO_OVERFLOW_DISCONNECTION
        # sim_obs sees line 0 disconnected (flow above the hard threshold)
        sim_obs, *_ = obs.simulate(self.env.action_space())
        assert sim_obs.rho[l_id] == 0
        assert not sim_obs.line_status[0]
        
        # now do the change
        params.NO_OVERFLOW_DISCONNECTION = True
        obs.change_forecast_parameters(params)
        sim_obs, *_ = obs.simulate(self.env.action_space())
        assert obs._obs_env.parameters.NO_OVERFLOW_DISCONNECTION
        assert sim_obs.rho[l_id] > 1.
        assert sim_obs.line_status[0]
        
        
    def test_change_param_get_f_env(self):
        obs = self.env.reset()
        params = self.env.parameters
        params.NO_OVERFLOW_DISCONNECTION = True
        
        # before changing the parameters
        f_env = obs.get_forecast_env()
        assert not f_env.parameters.NO_OVERFLOW_DISCONNECTION
        
        # now change the parameters
        obs.change_forecast_parameters(params)
        f_env2 = obs.get_forecast_env()
        assert f_env2.parameters.NO_OVERFLOW_DISCONNECTION
        
        # check with simulate now
        sim_obs, *_ = obs.simulate(self.env.action_space())
        f_env3 = obs.get_forecast_env()
        assert f_env3.parameters.NO_OVERFLOW_DISCONNECTION
        
        
if __name__ == "__main__":
    unittest.main()
