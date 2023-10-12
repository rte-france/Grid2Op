# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import unittest

from grid2op.tests.helper_path_test import *

PATH_ADN_CHRONICS_FOLDER = os.path.abspath(
    os.path.join(PATH_CHRONICS, "test_multi_chronics")
)
PATH_PREVIOUS_RUNNER = os.path.join(data_test_dir, "runner_data")

import grid2op
from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from grid2op.dtypes import dt_float

warnings.simplefilter("error")


class TestRunner(HelperTests, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.init_grid_path = os.path.join(PATH_DATA_TEST_PP, "test_case14.json")
        self.path_chron = PATH_ADN_CHRONICS_FOLDER
        self.parameters_path = None
        self.max_iter = 10
        self.real_reward = dt_float(7748.425 / 12.)
        self.real_reward_li = [self.real_reward, dt_float(7786.8955 / 12.)]  # 7786.89599609375

        self.all_real_rewards = [
            dt_float(el / 12.)
            for el in [
                761.3295,
                768.10144,
                770.2673,
                767.767,
                768.69,
                768.71246,
                779.1029,
                783.2737,
                788.7833,
                792.39764,
            ]
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True)
        self.runner = Runner(**self.env.get_params_for_runner())

    def test_one_episode(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _, cum_reward, timestep, max_ts = self.runner.run_one_episode(
            max_iter=self.max_iter
        )
        assert int(timestep) == self.max_iter
        assert np.abs(cum_reward - self.real_reward) <= self.tol_one, f"{cum_reward} != {self.real_reward}"

    def test_one_episode_detailed(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _, cum_reward, timestep, max_ts, episode_data = self.runner.run_one_episode(
            max_iter=self.max_iter, detailed_output=True
        )
        assert int(timestep) == self.max_iter
        assert np.abs(cum_reward - self.real_reward) <= self.tol_one
        for j in range(len(self.all_real_rewards)):
            assert (
                np.abs(episode_data.rewards[j] - self.all_real_rewards[j])
                <= self.tol_one
            ), f"{episode_data.rewards[j]} != {self.all_real_rewards[j]}"

    def test_2episode(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner._run_sequential(nb_episode=2, max_iter=self.max_iter)
        assert len(res) == 2
        for i, (stuff, _, cum_reward, timestep, total_ts) in enumerate(res):
            assert int(timestep) == self.max_iter
            assert np.abs(cum_reward - self.real_reward_li[i]) <= self.tol_one, f"for iter {i}: {cum_reward} != {self.real_reward_li[i]}"

    def test_init_from_env(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case14_test", test=True) as env:
                runner = Runner(**env.get_params_for_runner())
                res = runner.run(nb_episode=1, max_iter=self.max_iter)
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == self.max_iter, f"{timestep} != {self.max_iter}"

    def test_seed_seq(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case14_test", test=True) as env:
                runner = Runner(**env.get_params_for_runner())
                res = runner.run(
                    nb_episode=1, max_iter=self.max_iter, env_seeds=[1], agent_seeds=[2]
                )
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == self.max_iter, f"{timestep} != {self.max_iter}"

    def test_seed_par(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case14_test", test=True) as env:
                runner = Runner(**env.get_params_for_runner())
                res = runner.run(
                    nb_episode=2,
                    nb_process=2,
                    max_iter=self.max_iter,
                    env_seeds=[1, 2],
                    agent_seeds=[3, 4],
                )
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == self.max_iter


if __name__ == "__main__":
    unittest.main()
