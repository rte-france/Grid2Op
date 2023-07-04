# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import grid2op
from grid2op.gym_compat import GymEnv, BoxGymActSpace, BoxGymObsSpace, DiscreteActSpace
import unittest
import warnings
import pdb

class L2RPNIDF2023Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_idf_2023", test=True)
        self.env.seed(0)
        self.env.set_id(0)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def legal_action_2subs(self):
        act12 = self.env.action_space({"set_bus": {"substations_id": [(3, (1, 2, 1, 2, 1)), (33, (1, 2, 1, 2, 1, 2))]}})
        act23 = self.env.action_space({"set_bus": {"substations_id": [(33, (1, 2, 1, 2, 1, 2)), (67, (1, 2, 1, 2))]}})
        act13 = self.env.action_space({"set_bus": {"substations_id": [(3, (1, 2, 1, 2, 1)), (67, (1, 2, 1, 2))]}})
        obs, reward, done, info = self.env.step(act12)
        assert not info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act13)
        assert not info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act23)
        assert not info["is_illegal"]
        self.env.reset()

    def test_illegal_action_2subs(self):
        # illegal actions
        act11 = self.env.action_space({"set_bus": {"substations_id": [(3, (1, 2, 1, 2, 1)), (4, (1, 2, 1, 2, 1))]}})
        act22 = self.env.action_space({"set_bus": {"substations_id": [(33, (1, 2, 1, 2, 1, 2)), (36, (1, 2, 1, 2, 1, 2))]}})
        act33 = self.env.action_space({"set_bus": {"substations_id": [(67, (1, 2, 1, 2)), (68, (1, 2, 1, 2, 1, 2, 1)) ]}})
        obs, reward, done, info = self.env.step(act11)
        assert info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act22)
        assert info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act33)
        assert info["is_illegal"]
        self.env.reset()

    def test_legal_action_2lines(self):
        # legal actions
        act12 = self.env.action_space({"set_line_status": [(0, -1), (110, -1)]})
        act23 = self.env.action_space({"set_line_status": [(110, -1), (3, -1)]})
        act13 = self.env.action_space({"set_line_status": [(0, -1), (3, -1)]})
        obs, reward, done, info = self.env.step(act12)
        assert not info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act13)
        assert not info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act23)
        assert not info["is_illegal"]
        self.env.reset()
    
    def test_illegal_action_2lines(self):
        # illegal actions
        act11 = self.env.action_space({"set_line_status": [(0, -1), (1, -1)]})
        act22 = self.env.action_space({"set_line_status": [(110, -1), (111, -1)]})
        act33 = self.env.action_space({"set_line_status": [(3, -1), (7, -1)]})
        obs, reward, done, info = self.env.step(act11)
        assert info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act22)
        assert info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act33)
        assert info["is_illegal"]
        self.env.reset()
            
    def test_to_gym(self):
        env_gym = GymEnv(self.env)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            box_act = BoxGymActSpace(self.env.action_space)
            bos_obs = BoxGymObsSpace(self.env.observation_space)
            discrete_act = BoxGymActSpace(self.env.action_space)
    
    def test_forecast_env(self):
        obs = self.env.reset()
        for_env = obs.get_forecast_env()
        assert for_env.max_episode_duration() == 13  # 12 + 1
        
    
if __name__ == '__main__':
    unittest.main()
