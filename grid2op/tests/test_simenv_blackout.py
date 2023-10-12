# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

import grid2op
from grid2op.Reward import BaseReward


class SimEnvRewardTester(BaseReward):
    def reset(self, env):
        self._sim_env = self.is_simulated_env(env)
        return super().reset(env) 
    
    def initialize(self, env):
        self._sim_env = self.is_simulated_env(env)
        return super().initialize(env)
    
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if self._sim_env:
            return -1.
        return 1.


class TestIsSimulatedEnv(unittest.TestCase):
    def setUp(self) -> None:
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(env_name, test=True, reward_class=SimEnvRewardTester,
                                    _add_to_name=type(self).__name__)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_simulate(self):
        obs = self.env.reset()
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert reward == 1., f"{reward} vs 1."
        sim_o, sim_r, *_ = obs.simulate(self.env.action_space())
        assert sim_r == -1., f"{reward} vs -1."
        sim_o, sim_r, sim_d, sim_i = obs.simulate(self.env.action_space({"set_bus": {"loads_id": [(0, -1)]}}))
        assert sim_d
        assert sim_r == -1., f"{reward} vs -1."
        
    def test_forecast_env(self):
        obs = self.env.reset()
        for_env = obs.get_forecast_env()
        for_d = False
        i = 0
        while not for_d:
            i += 1
            for_o, for_r, for_d, for_i = for_env.step(self.env.action_space())
            assert for_r == -1.0, f"{for_r} vs -1. for iter {i}"
        
        
if __name__ == "__main__":
    unittest.main()
