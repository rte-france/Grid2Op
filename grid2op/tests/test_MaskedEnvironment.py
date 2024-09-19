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
    @staticmethod  
    def get_mask():
        mask = np.full(20, fill_value=False, dtype=bool)
        mask[[0, 1, 4, 2, 3, 6, 5]] = True  # THT part
        return mask
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_in = MaskedEnvironment(grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__),
                                            lines_of_interest=TestMaskedEnvironment.get_mask())
            self.env_out = MaskedEnvironment(grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__),
                                            lines_of_interest=~TestMaskedEnvironment.get_mask())
        self.line_id = 3
        th_lim = self.env_in.get_thermal_limit() * 2.  # avoid all problem in general
        th_lim[self.line_id] /= 10.  # make sure to get trouble in line 3
        # env_in: line is int the area
        self.env_in.set_thermal_limit(th_lim)
        # env_out: line is out of the area
        self.env_out.set_thermal_limit(th_lim)
        
        TestMaskedEnvironment._init_env(self.env_in)
        TestMaskedEnvironment._init_env(self.env_out)
        
    @staticmethod  
    def _init_env(env):
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
                assert obs_out.timestep_overflow[self.line_id] == i + 1, f"error for step {i}: {obs_out.timestep_overflow[self.line_id]}"
            else:
                # cooldown applied for line 3: 
                # - it disconnect stuff in `self.env_in`
                # - it does not affect anything in `self.env_out`
                assert not obs_in.line_status[self.line_id]
                assert obs_out.timestep_overflow[self.line_id] == i + 1, f"error for step {i}: {obs_out.timestep_overflow[self.line_id]}"
    
    def test_reset(self):
        # timestep_overflow should be 0 initially even if the flow is too high
        obs = self.env_in.reset()
        assert obs.timestep_overflow[self.line_id] == 0
        assert obs.rho[self.line_id] > 1.
        

class TestMaskedEnvironmentCpy(TestMaskedEnvironment):
    def setUp(self) -> None:
        super().setUp()
        init_int = self.env_in
        init_out = self.env_out
        self.env_in = self.env_in.copy()
        self.env_out = self.env_out.copy()
        init_int.close()
        init_out.close()
        

class TestMaskedEnvironmentRunner(unittest.TestCase):    
    def setUp(self) -> None:
        TestMaskedEnvironment.setUp(self)
        self.max_iter = 10

    def tearDown(self) -> None:
        self.env_in.close()
        self.env_out.close()
        return super().tearDown()
        
    def test_runner_can_make(self):
        runner = Runner(**self.env_in.get_params_for_runner())
        env2 = runner.init_env()
        assert isinstance(env2, MaskedEnvironment)
        assert (env2._lines_of_interest == self.env_in._lines_of_interest).all()
    
    def test_runner(self):
        # create the runner
        runner_in = Runner(**self.env_in.get_params_for_runner())
        runner_out = Runner(**self.env_out.get_params_for_runner())
        res_in, *_ = runner_in.run(nb_episode=1, max_iter=self.max_iter, env_seeds=[0], episode_id=[0], add_detailed_output=True)
        res_out, *_ = runner_out.run(nb_episode=1, max_iter=self.max_iter, env_seeds=[0], episode_id=[0], add_detailed_output=True)
        res_in2, *_ = runner_in.run(nb_episode=1, max_iter=self.max_iter, env_seeds=[0], episode_id=[0])
        # check correct results are obtained when agregated
        assert res_in[3] == 10
        assert res_in2[3] == 10
        assert res_out[3] == 10
        assert np.allclose(res_in[2], 645.4992065)
        assert np.allclose(res_in2[2], 645.4992065)
        assert np.allclose(res_out[2], 645.7020874)
        
        # check detailed results
        ep_data_in = res_in[-1]
        ep_data_out = res_out[-1]
        for i in range(self.max_iter + 1):
            obs_in = ep_data_in.observations[i]
            obs_out = ep_data_out.observations[i]
            if i < 3:
                assert obs_in.timestep_overflow[self.line_id] == i, f"error for step {i}: {obs_in.timestep_overflow[self.line_id]}"
                assert obs_out.timestep_overflow[self.line_id] == i, f"error for step {i}: {obs_out.timestep_overflow[self.line_id]}"
            else:
                # cooldown applied for line 3: 
                # - it disconnect stuff in `self.env_in`
                # - it does not affect anything in `self.env_out`
                assert not obs_in.line_status[self.line_id], f"error for step {i}: line is not disconnected"
                assert obs_out.timestep_overflow[self.line_id] == i, f"error for step {i}: {obs_out.timestep_overflow[self.line_id]}"
    
        
        
class TestMaskedEnvironmentGym(unittest.TestCase):
    def setUp(self) -> None:
        TestMaskedEnvironment.setUp(self)

    def tearDown(self) -> None:
        self.env_in.close()
        self.env_out.close()
        return super().tearDown()
    
    def _aux_run_envs(self, act, env_gym_in, env_gym_out):
        for i in range(10):
            obs_in, reward, done, truncated, info = env_gym_in.step(act)
            obs_out, reward, done, truncated, info = env_gym_out.step(act)
            if i < 2:  # 2 : 2 full steps already
                assert obs_in["timestep_overflow"][self.line_id] == i + 1, f"error for step {i}: {obs_in['timestep_overflow'][self.line_id]}"
                assert obs_out['timestep_overflow'][self.line_id] == i + 1, f"error for step {i}: {obs_out['timestep_overflow'][self.line_id]}"
            else:
                # cooldown applied for line 3: 
                # - it disconnect stuff in `self.env_in`
                # - it does not affect anything in `self.env_out`
                assert not obs_in["line_status"][self.line_id]
                assert obs_out["timestep_overflow"][self.line_id] == i + 1, f"error for step {i}: {obs_out['timestep_overflow'][self.line_id]}"
    
    def test_gym_with_step(self):
        """test the step function also disconnects (or not) the lines"""
        env_gym_in = GymEnv(self.env_in)
        env_gym_out = GymEnv(self.env_out)
        act = {}
        self._aux_run_envs(act, env_gym_in, env_gym_out)
        env_gym_in.reset()
        env_gym_out.reset()
        self._aux_run_envs(act, env_gym_in, env_gym_out)
            
    def test_gym_normal(self):
        """test I can create the gym env"""
        env_gym = GymEnv(self.env_in)
        env_gym.reset()
    
    def test_gym_box(self):
        """test I can create the gym env with box ob space and act space"""
        env_gym_in = GymEnv(self.env_in)
        env_gym_out = GymEnv(self.env_out)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_gym_in.action_space = BoxGymActSpace(self.env_in.action_space)
            env_gym_in.observation_space = BoxGymObsSpace(self.env_in.observation_space)
            env_gym_out.action_space = BoxGymActSpace(self.env_out.action_space)
            env_gym_out.observation_space = BoxGymObsSpace(self.env_out.observation_space)
        env_gym_in.reset()
        env_gym_out.reset()
    
    def test_gym_discrete(self):
        """test I can create the gym env with discrete act space"""
        env_gym_in = GymEnv(self.env_in)
        env_gym_out = GymEnv(self.env_out)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_gym_in.action_space = DiscreteActSpace(self.env_in.action_space)
            env_gym_out.action_space = DiscreteActSpace(self.env_out.action_space)
        env_gym_in.reset()
        env_gym_out.reset()
        act = 0
        self._aux_run_envs(act, env_gym_in, env_gym_out)
        
    def test_gym_multidiscrete(self):
        """test I can create the gym env with multi discrete act space"""
        env_gym_in = GymEnv(self.env_in)
        env_gym_out = GymEnv(self.env_out)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_gym_in.action_space = MultiDiscreteActSpace(self.env_in.action_space)
            env_gym_out.action_space = MultiDiscreteActSpace(self.env_out.action_space)
        env_gym_in.reset()
        env_gym_out.reset()
        act = env_gym_in.action_space.sample()
        act[:] = 0
        self._aux_run_envs(act, env_gym_in, env_gym_out)


if __name__ == "__main__":
    unittest.main()
