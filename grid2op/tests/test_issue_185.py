# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import os

from grid2op.tests.helper_path_test import *
import grid2op
from grid2op.gym_compat import (
    GymEnv,
    BoxGymActSpace,
    BoxGymObsSpace,
    MultiDiscreteActSpace,
    DiscreteActSpace,
)

import pdb


ENV_WITH_ALARM_NAME = os.path.join(
    PATH_DATA_TEST, "l2rpn_neurips_2020_track1_with_alarm"
)


class Issue185Tester(unittest.TestCase):
    """
    this test ensure that every "test" environment can be converted to gym

    this test suit goes beyond the simple error raised in the github issue.
    """

    def get_list_env(self):
        res = grid2op.list_available_test_env()
        res.append(ENV_WITH_ALARM_NAME)
        return res

    def test_issue_185(self):
        for env_name in self.get_list_env():
            if env_name == "blank":
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(env_name, test=True) as env:
                    try:
                        gym_env = GymEnv(env)
                        # gym_env.seed(0)
                        # gym_env.observation_space.seed(0)
                        # gym_env.action_space.seed(0)
                        gym_env.action_space.seed(0)
                        obs_gym, *_ = gym_env.reset(seed=0)  # reset and seed
                        assert (
                            obs_gym["a_ex"].shape[0] == env.n_line
                        ), f"error for {env_name}"
                        # if obs_gym not in gym_env.observation_space:
                        for k in gym_env.observation_space.spaces.keys():
                            assert (
                                obs_gym[k] in gym_env.observation_space[k]
                            ), f"error for {env_name}, for key={k}"
                    finally:
                        gym_env.close()

    def test_issue_185_act_box_space(self):
        for env_name in self.get_list_env():
            if env_name == "blank":
                continue
            if env_name == "rte_case5_example":
                # no action to perform for this env ! (no redispatch, curtail, storage)
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(env_name, test=True) as env:
                    try:
                        gym_env = GymEnv(env)
                        try:
                            gym_env.action_space = BoxGymActSpace(gym_env.init_env.action_space)
                        except Exception as exc_:
                            raise AssertionError(f"Error for {env_name}: {exc_}") from exc_
                        # gym_env.seed(0)
                        # gym_env.observation_space.seed(0)
                        # gym_env.action_space.seed(0)
                        gym_env.action_space.seed(0)
                        obs_gym, *_ = gym_env.reset(seed=0)  # reset and seed
                        assert isinstance(obs_gym, dict), "probably a wrong gym version"
                        # "was_alert_used_after_attack"
                        # "was_alarm_used_after_game_over"
                        for key in gym_env.observation_space.keys():
                            assert key in obs_gym, f"error for {env_name} for {key}"
                            assert obs_gym[key] in gym_env.observation_space[key], f"error for {env_name} for {key}"
                        act = gym_env.action_space.sample()
                        assert act in gym_env.action_space, f"error for {env_name}"
                        obs, reward, done, truncated, info = gym_env.step(act)
                        assert isinstance(obs_gym, dict)
                        for key in gym_env.observation_space.keys():
                            assert key in obs_gym, f"error for {env_name} for {key}"
                            assert obs_gym[key] in gym_env.observation_space[key], f"error for {env_name} for {key}"
                    finally:
                        gym_env.close()

    def test_issue_185_obs_box_space(self):
        for env_name in self.get_list_env():
            if env_name == "blank":
                continue
            # if env_name != "l2rpn_neurips_2020_track1":
            #     continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(env_name, test=True) as env:
                    try:
                        gym_env = GymEnv(env)
                        gym_env.observation_space.close()
                        gym_env.observation_space = BoxGymObsSpace(
                            gym_env.init_env.observation_space
                        )
                        # gym_env.seed(0)
                        # gym_env.observation_space.seed(0)
                        # gym_env.action_space.seed(0)
                        gym_env.action_space.seed(0)
                        obs_gym, *_ = gym_env.reset(seed=0)  # reset and seed
                        if obs_gym not in gym_env.observation_space:
                            raise AssertionError(f"error for {env_name}: \n{gym_env.observation_space.low}\n{obs_gym}\n{gym_env.observation_space.high}")
                        act = gym_env.action_space.sample()
                        assert act in gym_env.action_space, f"error for {env_name}"
                        obs, reward, done, truncated, info = gym_env.step(act)
                        if obs not in gym_env.observation_space:
                            raise AssertionError(f"error for {env_name}: \n{gym_env.observation_space.low}\n{obs}\n{gym_env.observation_space.high}")
                    finally:
                        gym_env.close()

    def test_issue_185_act_multidiscrete_space(self):
        for env_name in self.get_list_env():
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
            elif env_name == ENV_WITH_ALARM_NAME:
                # takes too much time
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(env_name, test=True) as env:
                    try:
                        gym_env = GymEnv(env)
                        gym_env.action_space = MultiDiscreteActSpace(
                            gym_env.init_env.action_space
                        )
                        # gym_env.seed(0)
                        # gym_env.observation_space.seed(0)
                        gym_env.action_space.seed(0)
                        obs_gym, *_ = gym_env.reset(seed=0)
                        assert obs_gym in gym_env.observation_space, f"error for {env_name}"
                        act = gym_env.action_space.sample()
                        assert act in gym_env.action_space, f"error for {env_name}"
                        obs, reward, done, truncated, info = gym_env.step(act)
                        assert obs in gym_env.observation_space, f"error for {env_name}"
                    finally:
                        gym_env.close()

    def test_issue_185_act_discrete_space(self):
        for env_name in self.get_list_env():
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
            elif env_name == ENV_WITH_ALARM_NAME:
                # takes too much time
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(env_name, test=True) as env:
                    try:
                        gym_env = GymEnv(env)
                        gym_env.action_space = DiscreteActSpace(
                            gym_env.init_env.action_space
                        )
                        # gym_env.seed(0)
                        # gym_env.observation_space.seed(0)
                        # gym_env.action_space.seed(0)
                        gym_env.action_space.seed(0)
                        obs_gym, *_ = gym_env.reset(seed=0)
                        assert obs_gym in gym_env.observation_space, f"error for {env_name}"
                        act = gym_env.action_space.sample()
                        assert act in gym_env.action_space, f"error for {env_name}"
                        obs, reward, done, truncated, info = gym_env.step(act)
                        if obs not in gym_env.observation_space:
                            for k in obs:
                                if not obs[k] in gym_env.observation_space[k]:
                                    raise RuntimeError(
                                        f"Error for key {k} for env {env_name}"
                                    )
                    finally:
                        gym_env.close()


if __name__ == "__main__":
    unittest.main()
