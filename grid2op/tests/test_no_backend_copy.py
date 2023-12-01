# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings

from grid2op.Backend import PandaPowerBackend
from grid2op.Exceptions import (NoForecastAvailable,
                                EnvError,
                                BaseObservationError,
                                SimulatorError)
from grid2op.simulator import Simulator


class PPNoCpy(PandaPowerBackend):
    def copy(self):
        raise NotImplementedError("Not used for this class")

class PPNoCpyInCtor(PandaPowerBackend):
    def __init__(self,
                 detailed_infos_for_cascading_failures=False,
                 ligthsim2grid=False,
                 dist_slack=False,
                 max_iter=10):
        super().__init__(detailed_infos_for_cascading_failures,
                         ligthsim2grid,
                         dist_slack,
                         max_iter,
                         can_be_copied=False)
    
    
class NoCopyTester(unittest.TestCase):
    """test grid2op works when the backend cannot be copied."""
    def setUp(self) -> None:
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(env_name, test=True, backend=PPNoCpy(), _add_to_name=type(self).__name__)
            
    def tearDown(self) -> None:
        self.env.close()
        
    def test_env_correct_flags(self):
        assert not self.env.with_forecast
        assert self.env.get_obs()._obs_env is None
        assert not self.env.observation_space.with_forecast
        assert not self.env.backend._can_be_copied
        
    def test_no_backend_needs_copy(self):
        obs = self.env.reset()
        obs, reward, done, info = self.env.step(self.env.action_space()) 
    
    def test_cannot_reactivate_forecasts(self):
        with self.assertRaises(EnvError):
            self.env.reactivate_forecast()
            
    def test_cannot_use_simulate(self):
        obs = self.env.reset()
        with self.assertRaises(NoForecastAvailable):
            res = obs.simulate(self.env.action_space())
            
    def test_simulator_from_obs(self):
        obs = self.env.reset()
        with self.assertRaises(BaseObservationError):
            res = obs.get_simulator()
    
    def test_cannot_use_simulator(self):
        with self.assertRaises(SimulatorError):
            Simulator(backend=self.env.backend)
            
        with self.assertRaises(SimulatorError):
            Simulator(backend=None, env=self.env.backend)

class NoCopy2Tester(NoCopyTester):
    """test grid2op works when the backend cannot be copied."""
    def setUp(self) -> None:
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(env_name, test=True, backend=PPNoCpyInCtor(), _add_to_name=type(self).__name__)
            
            
if __name__ == "__main__":
    unittest.main()
