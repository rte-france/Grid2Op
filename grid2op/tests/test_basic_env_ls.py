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
import tempfile
import os
import json
import packaging
from packaging import version

import grid2op
from grid2op.Environment import Environment
from grid2op.Runner import Runner
from grid2op.gym_compat import (GymEnv,
                                BoxGymActSpace,
                                BoxGymObsSpace,
                                DiscreteActSpace,
                                MultiDiscreteActSpace)
from grid2op.Action import PlayableAction
from grid2op.Parameters import Parameters
from grid2op.Observation import CompleteObservation
from grid2op.Agent import RandomAgent
from grid2op.tests.helper_path_test import data_test_dir
from grid2op.Episode import EpisodeData

try:
    from lightsim2grid import LightSimBackend
    LS_AVAIL = True
except ImportError:
    LS_AVAIL = False
    pass
            
            
class TestEnvironmentBasic(unittest.TestCase):          
    def setUp(self) -> None:
        if not LS_AVAIL:
            self.skipTest("lightsim not installed")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox",
                                    test=True,
                                    _add_to_name=type(self).__name__,
                                    backend=LightSimBackend())
        self.line_id = 3
        th_lim = self.env.get_thermal_limit() * 2.  # avoid all problem in general
        th_lim[self.line_id] /= 10.  # make sure to get trouble in line 3
        self.env.set_thermal_limit(th_lim)
        
        TestEnvironmentBasic._init_env(self.env)
        
    @staticmethod  
    def _init_env(env):
        env.set_id(0)
        env.seed(0)
        env.reset()
        
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_right_type(self):
        assert isinstance(self.env, Environment)
        
    def test_ok(self):
        act = self.env.action_space()
        for i in range(10):
            obs_in, reward, done, info = self.env.step(act)
            if i < 2:  # 2 : 2 full steps already
                assert obs_in.timestep_overflow[self.line_id] == i + 1, f"error for step {i}: {obs_in.timestep_overflow[self.line_id]}"
            else:
                # cooldown applied for line 3: 
                # - it disconnect stuff in `self.env_in`
                # - it does not affect anything in `self.env_out`
                assert not obs_in.line_status[self.line_id]
    
    def test_reset(self):
        # timestep_overflow should be 0 initially even if the flow is too high
        obs = self.env.reset()
        assert obs.timestep_overflow[self.line_id] == 0
        assert obs.rho[self.line_id] > 1.
    
    def test_can_make_2_envs(self):
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(env_name, test=True, backend=LightSimBackend())

        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env1 = grid2op.make("educ_case14_storage",
                                test=True,
                                action_class=PlayableAction,
                                param=param,
                                backend=LightSimBackend())


class TestEnvironmentBasicCpy(TestEnvironmentBasic):
    def setUp(self) -> None:
        super().setUp()
        init_int = self.env
        self.env = self.env.copy()
        init_int.close()
        

class TestBasicEnvironmentRunner(unittest.TestCase):    
    def setUp(self) -> None:
        TestEnvironmentBasic.setUp(self)
        self.max_iter = 10

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
        
    def test_runner_can_make(self):
        runner = Runner(**self.env.get_params_for_runner())
        env2 = runner.init_env()
        assert isinstance(env2, Environment)
    
    def test_runner(self):
        # create the runner
        runner_in = Runner(**self.env.get_params_for_runner())
        res_in, *_ = runner_in.run(nb_episode=1, max_iter=self.max_iter, env_seeds=[0], episode_id=[0], add_detailed_output=True)
        res_in2, *_ = runner_in.run(nb_episode=1, max_iter=self.max_iter, env_seeds=[0], episode_id=[0])
        # check correct results are obtained when agregated
        assert res_in[3] == self.max_iter, f"{res_in[3]} vs {self.max_iter}"
        assert res_in2[3] == self.max_iter, f"{res_in[3]} vs {self.max_iter}"
        assert np.allclose(res_in[2], 645.4992065)
        assert np.allclose(res_in2[2], 645.4992065)
        
        # check detailed results
        ep_data_in = res_in[-1]
        for i in range(self.max_iter + 1):
            obs_in = ep_data_in.observations[i]
            if i < 3:
                assert obs_in.timestep_overflow[self.line_id] == i, f"error for step {i}: {obs_in.timestep_overflow[self.line_id]}"
            else:
                # cooldown applied for line 3: 
                # - it disconnect stuff in `self.env_in`
                # - it does not affect anything in `self.env_out`
                assert not obs_in.line_status[self.line_id], f"error for step {i}: line is not disconnected"
                
    def test_backward_compatibility(self):
        with warnings.catch_warnings(action="ignore"):
            # TODO copy paste from test_Runner
            backward_comp_version = [
                "1.6.4",  # minimum version for lightsim2grid
                "1.6.5",
                "1.7.0",
                "1.7.1",
                "1.7.2",
                "1.8.1",
                # "1.9.0",  # this one is bugy I don"t know why
                "1.9.1",
                "1.9.2",
                "1.9.3",
                "1.9.4",
                "1.9.5",
                "1.9.6",
                "1.9.7",
                "1.9.8",
                "1.10.0",
                "1.10.1",
            ]
            # first check a normal run
            curr_version = "test_version"
            PATH_PREVIOUS_RUNNER = os.path.join(data_test_dir, "runner_data")
            assert (
                "curtailment" in CompleteObservation.attr_list_vect
            ), "error at the beginning"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(
                    "rte_case5_example", test=True,
                    _add_to_name=type(self).__name__,
                    backend=LightSimBackend()
                ) as env, tempfile.TemporaryDirectory() as path:
                    runner = Runner(**env.get_params_for_runner(), agentClass=RandomAgent)
                    runner.run(
                        nb_episode=2,
                        path_save=os.path.join(path, curr_version),
                        pbar=False,
                        max_iter=100,
                        env_seeds=[1, 0],
                        agent_seeds=[42, 69],
                    )
                    # check that i can read this data generate for this runner
                    try:
                        self._aux_backward(path, curr_version, curr_version)
                    except Exception as exc_:
                        raise RuntimeError(f"error for {curr_version}") from exc_
            assert (
                "curtailment" in CompleteObservation.attr_list_vect
            ), "error after the first runner"

            # check that it raises a warning if loaded on the compatibility version
            grid2op_version = backward_comp_version[0]
            with self.assertWarns(UserWarning, msg=f"error for {grid2op_version}"):
                self._aux_backward(
                    PATH_PREVIOUS_RUNNER, f"res_agent_{grid2op_version}", grid2op_version
                )
            
            # now check the compat versions
            for grid2op_version in backward_comp_version:
                # check that i can read previous data stored from previous grid2Op version
                # can be loaded properly
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    try:
                        self._aux_backward(
                            PATH_PREVIOUS_RUNNER,
                            f"res_agent_{grid2op_version}",
                            grid2op_version,
                        )
                    except Exception as exc_:
                        raise RuntimeError(f"error for {grid2op_version}") from exc_
                assert "curtailment" in CompleteObservation.attr_list_vect, (
                    f"error after the legacy version " f"{grid2op_version}"
                )
 
    def _aux_backward(self, base_path, g2op_version_txt, g2op_version):
        # TODO copy paste from test_Runner
        episode_studied = EpisodeData.list_episode(
            os.path.join(base_path, g2op_version_txt)
        )
        for base_path, episode_path in episode_studied:
            assert "curtailment" in CompleteObservation.attr_list_vect, (
                f"error after the legacy version " f"{g2op_version}"
            )
            this_episode = EpisodeData.from_disk(base_path, episode_path)
            assert "curtailment" in CompleteObservation.attr_list_vect, (
                f"error after the legacy version " f"{g2op_version}"
            )
            full_episode_path = os.path.join(base_path, episode_path)
            with open(
                os.path.join(full_episode_path, "episode_meta.json"),
                "r",
                encoding="utf-8",
            ) as f:
                meta_data = json.load(f)
            nb_ts = int(meta_data["nb_timestep_played"])
            try:
                assert len(this_episode.actions) == nb_ts, (
                    f"wrong number of elements for actions for version "
                    f"{g2op_version_txt}: {len(this_episode.actions)} vs {nb_ts}"
                )
                assert len(this_episode.observations) == nb_ts + 1, (
                    f"wrong number of elements for observations "
                    f"for version {g2op_version_txt}: "
                    f"{len(this_episode.observations)} vs {nb_ts}"
                )
                assert len(this_episode.env_actions) == nb_ts, (
                    f"wrong number of elements for env_actions for "
                    f"version {g2op_version_txt}: "
                    f"{len(this_episode.env_actions)} vs {nb_ts}"
                )
            except Exception as exc_:
                raise exc_
            g2op_ver = ""
            try:
                g2op_ver = version.parse(g2op_version)
            except packaging.version.InvalidVersion:
                if g2op_version != "test_version":
                    g2op_ver = version.parse("0.0.1")
                else:
                    g2op_ver = version.parse("1.4.1")
            if g2op_ver <= version.parse("1.4.0"):
                assert (
                    EpisodeData.get_grid2op_version(full_episode_path) == "<=1.4.0"
                ), "wrong grid2op version stored (grid2op version <= 1.4.0)"
            elif g2op_version == "test_version":
                assert (
                    EpisodeData.get_grid2op_version(full_episode_path)
                    == grid2op.__version__
                ), "wrong grid2op version stored (test_version)"
            else:
                assert (
                    EpisodeData.get_grid2op_version(full_episode_path) == g2op_version
                ), "wrong grid2op version stored (>=1.5.0)"       
        
class TestBasicEnvironmentGym(unittest.TestCase):
    def setUp(self) -> None:
        TestEnvironmentBasic.setUp(self)

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def _aux_run_envs(self, act, env_gym):
        for i in range(10):
            obs_in, reward, done, truncated, info = env_gym.step(act)
            if i < 2:  # 2 : 2 full steps already
                assert obs_in["timestep_overflow"][self.line_id] == i + 1, f"error for step {i}: {obs_in['timestep_overflow'][self.line_id]}"
            else:
                # cooldown applied for line 3: 
                # - it disconnect stuff in `self.env_in`
                # - it does not affect anything in `self.env_out`
                assert not obs_in["line_status"][self.line_id]
    
    def test_gym_with_step(self):
        """test the step function also disconnects (or not) the lines"""
        env_gym = GymEnv(self.env)
        act = {}
        self._aux_run_envs(act, env_gym)
        env_gym.reset()
        self._aux_run_envs(act, env_gym)
            
    def test_gym_normal(self):
        """test I can create the gym env"""
        env_gym = GymEnv(self.env)
        env_gym.reset()
    
    def test_gym_box(self):
        """test I can create the gym env with box ob space and act space"""
        env_gym = GymEnv(self.env)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_gym.action_space = BoxGymActSpace(self.env.action_space)
            env_gym.observation_space = BoxGymObsSpace(self.env.observation_space)
        env_gym.reset()
    
    def test_gym_discrete(self):
        """test I can create the gym env with discrete act space"""
        env_gym = GymEnv(self.env)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_gym.action_space = DiscreteActSpace(self.env.action_space)
        env_gym.reset()
        act = 0
        self._aux_run_envs(act, env_gym)
        
    def test_gym_multidiscrete(self):
        """test I can create the gym env with multi discrete act space"""
        env_gym = GymEnv(self.env)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_gym.action_space = MultiDiscreteActSpace(self.env.action_space)
        env_gym.reset()
        act = env_gym.action_space.sample()
        act[:] = 0
        self._aux_run_envs(act, env_gym)


if __name__ == "__main__":
    unittest.main()
