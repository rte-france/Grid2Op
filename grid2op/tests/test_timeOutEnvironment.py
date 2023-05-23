# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import time
import warnings
import unittest

import grid2op
from grid2op.Environment import TimedOutEnvironment
from grid2op.Agent import BaseAgent
from grid2op.Runner import Runner
from grid2op.gym_compat import (GymEnv,
                                BoxGymActSpace,
                                BoxGymObsSpace,
                                DiscreteActSpace,
                                MultiDiscreteActSpace)

class WaitAgent(BaseAgent):
    def __init__(self, action_space):
        super().__init__(action_space)
    
    def get_timeout(self, env):
        return env.time_out_ms if isinstance(env, TimedOutEnvironment) else env.init_env.time_out_ms
    
    def act(self, obs, reward, done):
        time.sleep(1e-3 * self.time_out_ms)
        return self.action_space()
    
    def act_gym(self, obs, reward, done):
        time.sleep(1e-3 * (self.time_out_ms  - 220.))
        return self.action_space.sample()
        

class AgentOK(WaitAgent):
    def __init__(self, env):
        super().__init__(env.action_space)
        self.time_out_ms = 0.9 * self.get_timeout(env)

    
class AgentKO(WaitAgent):
    def __init__(self, env):
        super().__init__(env.action_space)
        self.time_out_ms = 1.1 * self.get_timeout(env)
    
    
class AgentKO1(WaitAgent):
    def __init__(self, env):
        super().__init__(env.action_space)
        self.time_out_ms = 1.9 * self.get_timeout(env)
            
            
class AgentKO2(WaitAgent):
    def __init__(self, env):
        super().__init__(env.action_space)
        self.time_out_ms = 2.1 * self.get_timeout(env)
            
            
class TestTimedOutEnvironment100(unittest.TestCase):
    def get_timeout_ms(self):
        return 250
        
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env1 = TimedOutEnvironment(grid2op.make("l2rpn_case14_sandbox", test=True),
                                            time_out_ms=self.get_timeout_ms())
        params = self.env1.parameters
        params.NO_OVERFLOW_DISCONNECTION = True
        self.env1.change_parameters(params)

    def tearDown(self) -> None:
        self.env1.close()
        return super().tearDown()
    
    def test_no_dn(self):
        agentok = AgentOK(self.env1)
        obs = self.env1.reset()
        assert self.env1._nb_dn_last == 0
        for i in range(10):
            act = agentok.act(None, None, None)
            obs, reward, done, info = self.env1.step(act)
            assert info["nb_do_nothing"] == 0
            assert info["nb_do_nothing_made"] == 0
            assert self.env1._nb_dn_last == 0
            
    def test_one_dn(self):
        agentko = AgentKO(self.env1)
        obs = self.env1.reset()
        assert self.env1._nb_dn_last == 0
        for i in range(10):
            act = agentko.act(None, None, None)
            obs, reward, done, info = self.env1.step(act)
            assert info["nb_do_nothing"] == 1
            assert info["nb_do_nothing_made"] == 1
            assert self.env1._nb_dn_last == 1
            
    def test_one_dn2(self):
        agentko = AgentKO1(self.env1)
        obs = self.env1.reset()
        assert self.env1._nb_dn_last == 0
        for i in range(10):
            act = agentko.act(None, None, None)
            obs, reward, done, info = self.env1.step(act)
            assert info["nb_do_nothing"] == 1
            assert info["nb_do_nothing_made"] == 1
            assert self.env1._nb_dn_last == 1

    def test_two_dn(self):
        agentko2 = AgentKO2(self.env1)
        obs = self.env1.reset()
        assert self.env1._nb_dn_last == 0
        for i in range(10):
            act = agentko2.act(None, None, None)
            obs, reward, done, info = self.env1.step(act)
            assert info["nb_do_nothing"] == 2
            assert info["nb_do_nothing_made"] == 2
            assert self.env1._nb_dn_last == 2
    
    def test_diff_dn(self):
        agentok = AgentOK(self.env1)
        agentko = AgentKO(self.env1)
        agentko2 = AgentKO2(self.env1)
        obs = self.env1.reset()
        assert self.env1._nb_dn_last == 0
        for i, agent in enumerate([agentok, agentko, agentko2] * 2):
            act = agent.act(None, None, None)
            obs, reward, done, info = self.env1.step(act)
            assert info["nb_do_nothing"] == i % 3
            assert info["nb_do_nothing_made"] == i % 3
            assert self.env1._nb_dn_last == i % 3
        

class TestTimedOutEnvironment50(TestTimedOutEnvironment100):
    def get_timeout_ms(self):
        return 300


class TestTimedOutEnvironmentCpy(TestTimedOutEnvironment100):
    def setUp(self) -> None:
        super().setUp()
        self.env0 = self.env1
        self.env1 = self.env0.copy()
        
    def tearDown(self) -> None:
        self.env1.close()
        self.env0.close()
        

class TestTOEnvRunner(unittest.TestCase):
    def get_timeout_ms(self):
        return 200
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env1 = TimedOutEnvironment(grid2op.make("l2rpn_case14_sandbox", test=True),
                                            time_out_ms=self.get_timeout_ms())
        params = self.env1.parameters
        params.NO_OVERFLOW_DISCONNECTION = True
        self.env1.change_parameters(params)
        self.cum_reward = 645.70208
        self.max_iter = 10

    def tearDown(self) -> None:
        self.env1.close()
        return super().tearDown()
        
    def test_runner_can_make(self):
        runner = Runner(**self.env1.get_params_for_runner())
        env2 = runner.init_env()
        assert isinstance(env2, TimedOutEnvironment)
        assert env2.time_out_ms == self.get_timeout_ms()
    
    def test_runner_noskip(self):
        agent = AgentOK(self.env1)
        runner = Runner(**self.env1.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=agent)
        res = runner.run(nb_episode=1,
                         max_iter=self.max_iter)
        _, _, cum_reward, timestep, max_ts = res[0]
        assert abs(cum_reward - self.cum_reward) <= 1e-5
    
    def test_runner_skip1(self):
        agent = AgentKO(self.env1)
        runner = Runner(**self.env1.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=agent)
        res = runner.run(nb_episode=1,
                         max_iter=self.max_iter)
        _, _, cum_reward, timestep, max_ts = res[0]
        assert abs(cum_reward - self.cum_reward) <= 1e-5
    
    def test_runner_skip2(self):
        agent = AgentKO2(self.env1)
        runner = Runner(**self.env1.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=agent)
        res = runner.run(nb_episode=1,
                         max_iter=self.max_iter)
        _, _, cum_reward, timestep, max_ts = res[0]
        assert abs(cum_reward - self.cum_reward) <= 1e-5
    
    def test_runner_skip2_2ep(self):
        agent = AgentKO2(self.env1)
        runner = Runner(**self.env1.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=agent)
        res = runner.run(nb_episode=2,
                         max_iter=self.max_iter)
        _, _, cum_reward, timestep, max_ts = res[0]
        assert abs(cum_reward - self.cum_reward) <= 1e-5
        _, _, cum_reward, timestep, max_ts = res[1]
        assert abs(cum_reward - 648.90795) <= 1e-5
    

class TestTOEnvGym(unittest.TestCase):
    def get_timeout_ms(self):
        return 400.
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env1 = TimedOutEnvironment(grid2op.make("l2rpn_case14_sandbox", test=True),
                                            time_out_ms=self.get_timeout_ms())

    def tearDown(self) -> None:
        self.env1.close()
        return super().tearDown()
    
    def test_gym_with_step(self):
        """test the step function also makes the 'do nothing'"""
        env_gym = GymEnv(self.env1)
        env_gym.reset()
        
        agentok = AgentOK(env_gym)
        for i in range(10):
            act = agentok.act_gym(None, None, None)
            for k in act:
                act[k][:] = 0
            *_, info = env_gym.step(act)
            assert info["nb_do_nothing"] == 0
            assert info["nb_do_nothing_made"] == 0
            assert env_gym.init_env._nb_dn_last == 0
            
        env_gym.reset()
        agentko = AgentKO1(env_gym)
        for i in range(10):
            act = agentko.act_gym(None, None, None)
            for k in act:
                act[k][:] = 0
            *_, info = env_gym.step(act)
            assert info["nb_do_nothing"] == 1
            assert info["nb_do_nothing_made"] == 1
            assert env_gym.init_env._nb_dn_last == 1
            
    def test_gym_normal(self):
        """test I can create the gym env"""
        env_gym = GymEnv(self.env1)
        env_gym.reset()
    
    def test_gym_box(self):
        """test I can create the gym env with box ob space and act space"""
        env_gym = GymEnv(self.env1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_gym.action_space = BoxGymActSpace(self.env1.action_space)
            env_gym.observation_space = BoxGymObsSpace(self.env1.observation_space)
        env_gym.reset()
    
    def test_gym_discrete(self):
        """test I can create the gym env with discrete act space"""
        env_gym = GymEnv(self.env1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_gym.action_space = DiscreteActSpace(self.env1.action_space)
        env_gym.reset()
    
    def test_gym_multidiscrete(self):
        """test I can create the gym env with multi discrete act space"""
        env_gym = GymEnv(self.env1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_gym.action_space = MultiDiscreteActSpace(self.env1.action_space)
        env_gym.reset()


if __name__ == "__main__":
    unittest.main()
