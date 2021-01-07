# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import tempfile
import pdb

from grid2op.tests.helper_path_test import *
PATH_ADN_CHRONICS_FOLDER = os.path.abspath(os.path.join(PATH_CHRONICS, "test_multi_chronics"))
PATH_PREVIOUS_RUNNER = os.path.join(data_dir, "runner_data")

import grid2op
from grid2op.Chronics import Multifolder
from grid2op.Reward import L2RPNReward
from grid2op.Backend import PandaPowerBackend
from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from grid2op.dtypes import dt_float
from grid2op.Agent import RandomAgent
from grid2op.Episode import EpisodeData
import json

import warnings
warnings.simplefilter("error")


class TestRunner(HelperTests):
    def setUp(self):
        self.init_grid_path = os.path.join(PATH_DATA_TEST_PP, "test_case14.json")
        self.path_chron = PATH_ADN_CHRONICS_FOLDER
        self.parameters_path = None
        self.max_iter = 10
        self.real_reward = dt_float(199.99800)
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
                             max_iter=self.max_iter,
                             name_env="test_runner_env")

    def test_one_episode(self):
        _, cum_reward, timestep = self.runner.run_one_episode(max_iter=self.max_iter)
        assert int(timestep) == self.max_iter
        assert np.abs(cum_reward - self.real_reward) <= self.tol_one

    def test_one_process_par(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = Runner._one_process_parrallel(self.runner, [0], 0, None, None, self.max_iter)
        assert len(res) == 1
        _, el1, el2, el3, el4 = res[0]
        assert el1 == "1"
        assert np.abs(el2 - 199.9980010986328) <= self.tol_one
        assert el3 == 10
        assert el4 == 10

    def test_2episode(self):
        res = self.runner._run_sequential(nb_episode=2, max_iter=self.max_iter)
        assert len(res) == 2
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == self.max_iter
            assert np.abs(cum_reward - self.real_reward) <= self.tol_one

    def test_2episode_2process(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner._run_parrallel(nb_episode=2, nb_process=2, max_iter=self.max_iter)
        assert len(res) == 2
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == self.max_iter
            assert np.abs(cum_reward - self.real_reward) <= self.tol_one

    def test_complex_agent(self):
        nb_episode = 4
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env:
                f = tempfile.mkdtemp()
                runner_params = env.get_params_for_runner()
                runner = Runner(**runner_params)
                res = runner.run(path_save=f,
                                 nb_episode=nb_episode,
                                 nb_process=2,
                                 max_iter=self.max_iter)
        test_ = set()
        for id_chron, name_chron, cum_reward, nb_time_step, max_ts in res:
            test_.add(name_chron)
        assert len(test_) == nb_episode

    def test_init_from_env(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case14_test", test=True) as env:
                runner = Runner(**env.get_params_for_runner())
        res = runner.run(nb_episode=1, max_iter=self.max_iter)
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == self.max_iter

    def test_init_from_env_with_other_reward(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case14_test", test=True, other_rewards={"test": L2RPNReward}) as env:
                runner = Runner(**env.get_params_for_runner())
        res = runner.run(nb_episode=1, max_iter=self.max_iter)
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == self.max_iter

    def test_seed_seq(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case14_test", test=True) as env:
                runner = Runner(**env.get_params_for_runner())
        res = runner.run(nb_episode=1, max_iter=self.max_iter, env_seeds=[1], agent_seeds=[2])
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == self.max_iter

    def test_seed_par(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case14_test", test=True) as env:
                runner = Runner(**env.get_params_for_runner())
        res = runner.run(nb_episode=2, nb_process=2, max_iter=self.max_iter, env_seeds=[1, 2], agent_seeds=[3, 4])
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == self.max_iter

    def test_seed_properly_set(self):
        class TestSuitAgent(RandomAgent):
            def __init__(self, *args, **kwargs):
                RandomAgent.__init__(self, *args, **kwargs)
                self.seeds = []

            def seed(self, seed):
                super().seed(seed)
                self.seeds.append(seed)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case14_test", test=True) as env:
                my_agent = TestSuitAgent(env.action_space)
                runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=my_agent)

        # test that the right seeds are assigned to the agent
        res = runner.run(nb_episode=3, max_iter=self.max_iter, env_seeds=[1, 2, 3], agent_seeds=[5, 6, 7])
        assert np.all(my_agent.seeds == [5, 6, 7])

        # test that is no seeds are set, then the "seed" function of the agent is not called.
        my_agent.seeds = []
        res = runner.run(nb_episode=3, max_iter=self.max_iter, env_seeds=[1, 2, 3])
        assert my_agent.seeds == []

    def test_always_same_order(self):
        # test that a call to "run" will do always the same chronics in the same order
        # regardless of the seed or the parallelism or the number of call to runner.run
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case14_test", test=True) as env:
                runner = Runner(**env.get_params_for_runner())
        res = runner.run(nb_episode=2, nb_process=2, max_iter=self.max_iter, env_seeds=[1, 2], agent_seeds=[3, 4])
        first_ = [el[0] for el in res]
        res = runner.run(nb_episode=2, nb_process=1, max_iter=self.max_iter, env_seeds=[1, 2], agent_seeds=[3, 4])
        second_ = [el[0] for el in res]
        res = runner.run(nb_episode=2, nb_process=1, max_iter=self.max_iter, env_seeds=[9, 10])
        third_ = [el[0] for el in res]
        res = runner.run(nb_episode=2, nb_process=2, max_iter=self.max_iter, env_seeds=[1, 2], agent_seeds=[3, 4])
        fourth_ = [el[0] for el in res]
        assert np.all(first_ == second_)
        assert np.all(first_ == third_)
        assert np.all(first_ == fourth_)

    def test_nomaxiter(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case14_test", test=True) as env:
                runner = Runner(**env.get_params_for_runner())
        runner.gridStateclass_kwargs["max_iter"] = 2*self.max_iter
        runner.chronics_handler.set_max_iter(2*self.max_iter)
        res = runner.run(nb_episode=1)
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == 2*self.max_iter

    def test_nomaxiter_par(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case14_test", test=True) as env:
                dict_ = env.get_params_for_runner()
                dict_["max_iter"] = -1
                sub_dict = dict_["gridStateclass_kwargs"]
                sub_dict["max_iter"] = 2*self.max_iter
                runner = Runner(**dict_)
        res = runner.run(nb_episode=2, nb_process=2)
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == 2*self.max_iter

    def _aux_backward(self, base_path, g2op_version):
        episode_studied = EpisodeData.list_episode(os.path.join(base_path, g2op_version))
        for base_path, episode_path in episode_studied:
            this_episode = EpisodeData.from_disk(base_path, episode_path)
            with open(os.path.join(os.path.join(base_path, episode_path), "episode_meta.json"), "r",
                      encoding="utf-8") as f:
                meta_data = json.load(f)
            nb_ts = int(meta_data["nb_timestep_played"])
            try:
                assert len(this_episode.actions) == nb_ts, f"wrong number of elements for actions for version " \
                                                           f"{g2op_version}: {len(this_episode.actions)} vs {nb_ts}"
                assert len(this_episode.observations) == nb_ts + 1, f"wrong number of elements for observations " \
                                                                    f"for version {g2op_version}: " \
                                                                    f"{len(this_episode.observations)} vs {nb_ts}"
                assert len(this_episode.env_actions) == nb_ts, f"wrong number of elements for env_actions for " \
                                                               f"version {g2op_version}: " \
                                                               f"{len(this_episode.env_actions)} vs {nb_ts}"
            except:
                import pdb
                pdb.set_trace()

    def test_backward_compatibility(self):
        backward_comp_version = ["1.0.0", "1.1.0", "1.1.1", "1.2.0", "1.2.1", "1.2.2", "1.2.3", "1.3.0"]
        curr_version = "current_version"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env, \
                 tempfile.TemporaryDirectory() as path:
                runner = Runner(**env.get_params_for_runner(), agentClass=RandomAgent)
                runner.run(nb_episode=2,
                           path_save=os.path.join(path, curr_version),
                           pbar=False,
                           max_iter=100,
                           env_seeds=[1, 0],
                           agent_seeds=[42, 69])
                # check that i can read this data generate for this runner
                self._aux_backward(path, curr_version)

        for grid2op_version in backward_comp_version:
            # check that i can read previous data stored from previous grid2Op version
            self._aux_backward(PATH_PREVIOUS_RUNNER, f"res_agent_{grid2op_version}")


if __name__ == "__main__":
    unittest.main()
