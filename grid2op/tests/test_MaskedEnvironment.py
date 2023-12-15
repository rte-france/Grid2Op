# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import unittest
import numpy as np

import grid2op
from grid2op.Environment import MaskedEnvironment
from grid2op.Runner import Runner
from grid2op.gym_compat import (GymEnv,
                                BoxGymActSpace,
                                BoxGymObsSpace,
                                DiscreteActSpace,
                                MultiDiscreteActSpace)
            
            
class TestMaskedEnvironment(unittest.TestCase):        
    def get_mask(self):
        mask = np.full(20, fill_value=False, dtype=bool)
        mask[[0, 1, 4, 2, 3, 6, 5]] = True  # THT part
        return mask
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_in = MaskedEnvironment(grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__),
                                            lines_of_interest=self.get_mask())
            self.env_out = MaskedEnvironment(grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__),
                                            lines_of_interest=~self.get_mask())
        self.line_id = 3
        th_lim = self.env_in.get_thermal_limit() * 2.  # avoid all problem in general
        th_lim[self.line_id] /= 10.  # make sure to get trouble in line 3
        # env_in: line is int the area
        self.env_in.set_thermal_limit(th_lim)
        # env_out: line is out of the area
        self.env_out.set_thermal_limit(th_lim)
        
        self._init_env(self.env_in)
        self._init_env(self.env_out)
    
    def _init_env(self, env):
        env.set_id(0)
        env.seed(0)
        env.reset()
        
    def tearDown(self) -> None:
        self.env_in.close()
        self.env_out.close()
        return super().tearDown()
    
    def test_right_type(self):
        assert isinstance(self.env_in, MaskedEnvironment)
        assert isinstance(self.env_out, MaskedEnvironment)
        assert hasattr(self.env_in, "_lines_of_interest")
        assert hasattr(self.env_out, "_lines_of_interest")
        assert self.env_in._lines_of_interest[self.line_id], "line_id should be in env_in"
        assert not self.env_out._lines_of_interest[self.line_id], "line_id should not be in env_out"
    
    def test_ok(self):
        act = self.env_in.action_space()
        for i in range(10):
            obs_in, reward, done, info = self.env_in.step(act)
            obs_out, reward, done, info = self.env_out.step(act)
            if i < 2:  # 2 : 2 full steps already
                assert obs_in.timestep_overflow[self.line_id] == i + 1, f"error for step {i}: {obs_in.timestep_overflow[self.line_id]}"
                assert obs_out.timestep_overflow[self.line_id] == i + 1, f"error for step {i}: {obs_in.timestep_overflow[self.line_id]}"
            else:
                # cooldown applied for line 3: 
                # - it disconnect stuff in `self.env_in`
                # - it does not affect anything in `self.env_out`
                assert not obs_in.line_status[self.line_id]
                assert obs_out.timestep_overflow[self.line_id] == i + 1, f"error for step {i}: {obs_in.timestep_overflow[self.line_id]}"
    
    def test_reset(self):
        # timestep_overflow should be 0 initially even if the flow is too high
        obs = self.env_in.reset()
        assert obs.timestep_overflow[self.line_id] == 0
        assert obs.rho[self.line_id] > 1.
        

class TestTimedOutEnvironmentCpy(TestMaskedEnvironment):
    def setUp(self) -> None:
        super().setUp()
        init_int = self.env_in.copy()
        init_out = self.env_out.copy()
        self.env0 = self.env_in.copy()
        self.env1 = self.env_out.copy()
        init_int.close()
        init_out.close()
        

# class TestTOEnvRunner(unittest.TestCase):
#     def get_timeout_ms(self):
#         return 200
    
#     def setUp(self) -> None:
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore")
#             self.env1 = TimedOutEnvironment(grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__),
#                                             time_out_ms=self.get_timeout_ms())
#         params = self.env1.parameters
#         params.NO_OVERFLOW_DISCONNECTION = True
#         self.env1.change_parameters(params)
#         self.cum_reward = 645.70208
#         self.max_iter = 10

#     def tearDown(self) -> None:
#         self.env1.close()
#         return super().tearDown()
        
#     def test_runner_can_make(self):
#         runner = Runner(**self.env1.get_params_for_runner())
#         env2 = runner.init_env()
#         assert isinstance(env2, TimedOutEnvironment)
#         assert env2.time_out_ms == self.get_timeout_ms()
    
#     def test_runner_noskip(self):
#         agent = AgentOK(self.env1)
#         runner = Runner(**self.env1.get_params_for_runner(),
#                         agentClass=None,
#                         agentInstance=agent)
#         res = runner.run(nb_episode=1,
#                          max_iter=self.max_iter)
#         _, _, cum_reward, timestep, max_ts = res[0]
#         assert abs(cum_reward - self.cum_reward) <= 1e-5
    
#     def test_runner_skip1(self):
#         agent = AgentKO(self.env1)
#         runner = Runner(**self.env1.get_params_for_runner(),
#                         agentClass=None,
#                         agentInstance=agent)
#         res = runner.run(nb_episode=1,
#                          max_iter=self.max_iter)
#         _, _, cum_reward, timestep, max_ts = res[0]
#         assert abs(cum_reward - self.cum_reward) <= 1e-5
    
#     def test_runner_skip2(self):
#         agent = AgentKO2(self.env1)
#         runner = Runner(**self.env1.get_params_for_runner(),
#                         agentClass=None,
#                         agentInstance=agent)
#         res = runner.run(nb_episode=1,
#                          max_iter=self.max_iter)
#         _, _, cum_reward, timestep, max_ts = res[0]
#         assert abs(cum_reward - self.cum_reward) <= 1e-5
    
#     def test_runner_skip2_2ep(self):
#         agent = AgentKO2(self.env1)
#         runner = Runner(**self.env1.get_params_for_runner(),
#                         agentClass=None,
#                         agentInstance=agent)
#         res = runner.run(nb_episode=2,
#                          max_iter=self.max_iter)
#         _, _, cum_reward, timestep, max_ts = res[0]
#         assert abs(cum_reward - self.cum_reward) <= 1e-5
#         _, _, cum_reward, timestep, max_ts = res[1]
#         assert abs(cum_reward - 648.90795) <= 1e-5
    

# class TestTOEnvGym(unittest.TestCase):
#     def get_timeout_ms(self):
#         return 400.
    
#     def setUp(self) -> None:
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore")
#             self.env1 = TimedOutEnvironment(grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__),
#                                             time_out_ms=self.get_timeout_ms())

#     def tearDown(self) -> None:
#         self.env1.close()
#         return super().tearDown()
    
#     def test_gym_with_step(self):
#         """test the step function also makes the 'do nothing'"""
#         self.skipTest("On docker execution time is too unstable")
#         env_gym = GymEnv(self.env1)
#         env_gym.reset()
        
#         agentok = AgentOK(env_gym)
#         for i in range(10):
#             act = agentok.act_gym(None, None, None)
#             for k in act:
#                 act[k][:] = 0
#             *_, info = env_gym.step(act)
#             assert info["nb_do_nothing"] == 0
#             assert info["nb_do_nothing_made"] == 0
#             assert env_gym.init_env._nb_dn_last == 0
            
#         env_gym.reset()
#         agentko = AgentKO1(env_gym)
#         for i in range(10):
#             act = agentko.act_gym(None, None, None)
#             for k in act:
#                 act[k][:] = 0
#             *_, info = env_gym.step(act)
#             assert info["nb_do_nothing"] == 1
#             assert info["nb_do_nothing_made"] == 1
#             assert env_gym.init_env._nb_dn_last == 1
            
#     def test_gym_normal(self):
#         """test I can create the gym env"""
#         env_gym = GymEnv(self.env1)
#         env_gym.reset()
    
#     def test_gym_box(self):
#         """test I can create the gym env with box ob space and act space"""
#         env_gym = GymEnv(self.env1)
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore")
#             env_gym.action_space = BoxGymActSpace(self.env1.action_space)
#             env_gym.observation_space = BoxGymObsSpace(self.env1.observation_space)
#         env_gym.reset()
    
#     def test_gym_discrete(self):
#         """test I can create the gym env with discrete act space"""
#         env_gym = GymEnv(self.env1)
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore")
#             env_gym.action_space = DiscreteActSpace(self.env1.action_space)
#         env_gym.reset()
    
#     def test_gym_multidiscrete(self):
#         """test I can create the gym env with multi discrete act space"""
#         env_gym = GymEnv(self.env1)
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore")
#             env_gym.action_space = MultiDiscreteActSpace(self.env1.action_space)
#         env_gym.reset()


if __name__ == "__main__":
    unittest.main()
