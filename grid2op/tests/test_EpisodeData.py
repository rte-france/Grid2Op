# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import tempfile
import pdb

from grid2op.tests.helper_path_test import *

from grid2op.Exceptions import *
from grid2op.Chronics import Multifolder
from grid2op.Reward import L2RPNReward
from grid2op.Backend import PandaPowerBackend
from grid2op.Runner import Runner
from grid2op.EpisodeData import EpisodeData

DEBUG = True
PATH_ADN_CHRONICS_FOLDER = os.path.abspath(os.path.join(PATH_CHRONICS, "test_multi_chronics"))


class TestEpisodeData(unittest.TestCase):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.max_iter = 10
        self.real_reward = 199.99800

        self.init_grid_path = os.path.join(
            PATH_DATA_TEST_PP, "test_case14.json")
        self.path_chron = PATH_ADN_CHRONICS_FOLDER
        self.parameters_path = None
        self.names_chronics_to_backend = {"loads": {"2_C-10.61": 'load_1_0', "3_C151.15": 'load_2_1',
                                                    "14_C63.6": 'load_13_2', "4_C-9.47": 'load_3_3',
                                                    "5_C201.84": 'load_4_4',
                                                    "6_C-6.27": 'load_5_5', "9_C130.49": 'load_8_6',
                                                    "10_C228.66": 'load_9_7',
                                                    "11_C-138.89": 'load_10_8', "12_C-27.88": 'load_11_9',
                                                    "13_C-13.33": 'load_12_10'},
                                          "lines": {'1_2_1': '0_1_0', '1_5_2': '0_4_1', '9_10_16': '8_9_2',
                                                    '9_14_17': '8_13_3',
                                                    '10_11_18': '9_10_4', '12_13_19': '11_12_5', '13_14_20': '12_13_6',
                                                    '2_3_3': '1_2_7', '2_4_4': '1_3_8', '2_5_5': '1_4_9',
                                                    '3_4_6': '2_3_10',
                                                    '4_5_7': '3_4_11', '6_11_11': '5_10_12', '6_12_12': '5_11_13',
                                                    '6_13_13': '5_12_14', '4_7_8': '3_6_15', '4_9_9': '3_8_16',
                                                    '5_6_10': '4_5_17',
                                                    '7_8_14': '6_7_18', '7_9_15': '6_8_19'},
                                          "prods": {"1_G137.1": 'gen_0_4', "3_G36.31": "gen_2_1", "6_G63.29": "gen_5_2",
                                                    "2_G-56.47": "gen_1_0", "8_G40.43": "gen_7_3"},
                                          }
        self.gridStateclass = Multifolder
        self.backendClass = PandaPowerBackend
        self.runner = Runner(init_grid_path=self.init_grid_path,
                             path_chron=self.path_chron,
                             parameters_path=self.parameters_path,
                             names_chronics_to_backend=self.names_chronics_to_backend,
                             gridStateclass=self.gridStateclass,
                             backendClass=self.backendClass,
                             rewardClass=L2RPNReward,
                             other_rewards={"test": L2RPNReward},
                             max_iter=self.max_iter)

    def test_one_episode_with_saving(self):
        f = tempfile.mkdtemp()
        episode_name, cum_reward, timestep = self.runner.run_one_episode(path_save=f)
        episode_data = EpisodeData.from_disk(agent_path=f, name=episode_name)
        assert int(episode_data.meta["chronics_max_timestep"]) == self.max_iter
        assert len(episode_data.other_rewards) == self.max_iter
        for other, real in zip(episode_data.other_rewards, episode_data.rewards):
            assert np.abs(other["test"] - real) <= self.tol_one
        assert np.abs(float(episode_data.meta["cumulative_reward"]) - self.real_reward) <= self.tol_one

    def test_3_episode_with_saving(self):
        f = tempfile.mkdtemp()
        res = self.runner.run_sequential(nb_episode=3, path_save=f)
        for i, episode_name, cum_reward, timestep, total_ts in res:
            episode_data = EpisodeData.from_disk(agent_path=f, name=episode_name)
            assert int(episode_data.meta["chronics_max_timestep"]) == self.max_iter
            assert np.abs(
                float(episode_data.meta["cumulative_reward"]) - self.real_reward) <= self.tol_one

    def test_3_episode_3process_with_saving(self):
        f = tempfile.mkdtemp()
        res = self.runner.run_parrallel(nb_episode=3, nb_process=2, path_save=f)
        assert len(res) == 3
        for i, episode_name, cum_reward, timestep, total_ts in res:
            episode_data = EpisodeData.from_disk(agent_path=f, name=episode_name)
            assert int(episode_data.meta["chronics_max_timestep"]) == self.max_iter
            assert np.abs(
                float(episode_data.meta["cumulative_reward"]) - self.real_reward) <= self.tol_one


if __name__ == "__main__":
    unittest.main()
