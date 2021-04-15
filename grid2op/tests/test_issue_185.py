# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

import grid2op
from grid2op.gym_compat import GymEnv, BoxGymActSpace, BoxGymObsSpace, MultiDiscreteActSpace, DiscreteActSpace


class Issue185Tester(unittest.TestCase):
    """
    this test ensure that every "test" environment can be converted to gym

    this test suit goes beyond the simple error raised in the github issue.
    """
    def test_issue_185(self):
        for env_name in grid2op.list_available_test_env():
            if env_name == "blank":
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(env_name, test=True) as env:
                    gym_env = GymEnv(env)
                    obs_gym = gym_env.reset()
                    assert obs_gym["a_ex"].shape[0] == env.n_line, f"error for {env_name}"

    def test_issue_185_act_box_space(self):
        for env_name in grid2op.list_available_test_env():
            if env_name == "blank":
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(env_name, test=True) as env:
                    gym_env = GymEnv(env)
                    gym_env.action_space = BoxGymActSpace(gym_env.init_env.action_space)
                    gym_env.action_space.seed(0)
                    obs_gym = gym_env.reset()
                    act = gym_env.action_space.sample()
                    obs, reward, done, info = gym_env.step(act)

    def test_issue_185_obs_box_space(self):
        for env_name in grid2op.list_available_test_env():
            if env_name == "blank":
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(env_name, test=True) as env:
                    gym_env = GymEnv(env)
                    gym_env.observation_space = BoxGymObsSpace(gym_env.init_env.observation_space)
                    gym_env.action_space.seed(0)
                    obs_gym = gym_env.reset()
                    act = gym_env.action_space.sample()
                    obs, reward, done, info = gym_env.step(act)

    def test_issue_185_act_multidiscrete_space(self):
        for env_name in grid2op.list_available_test_env():
            if env_name == "blank":
                continue
            elif env_name == "l2rpn_neurips_2020_track1":
                # takes too much time
                continue
            elif env_name == "l2rpn_neurips_2020_track2":
                # takes too much time
                continue
            elif env_name == "rte_case118_example":
                # takes too much time
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(env_name, test=True) as env:
                    gym_env = GymEnv(env)
                    gym_env.action_space = MultiDiscreteActSpace(gym_env.init_env.action_space)
                    gym_env.action_space.seed(0)
                    obs_gym = gym_env.reset()
                    act = gym_env.action_space.sample()
                    obs, reward, done, info = gym_env.step(act)

    def test_issue_185_act_discrete_space(self):
        for env_name in grid2op.list_available_test_env():
            if env_name == "blank":
                continue
            elif env_name == "l2rpn_neurips_2020_track1":
                # takes too much time
                continue
            elif env_name == "l2rpn_neurips_2020_track2":
                # takes too much time
                continue
            elif env_name == "rte_case118_example":
                # takes too much time
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(env_name, test=True) as env:
                    gym_env = GymEnv(env)
                    gym_env.action_space = DiscreteActSpace(gym_env.init_env.action_space)
                    gym_env.action_space.seed(0)
                    obs_gym = gym_env.reset()
                    act = gym_env.action_space.sample()
                    obs, reward, done, info = gym_env.step(act)


if __name__ == "__main__":
    unittest.main()
