# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import tempfile
import json
import unittest
import pdb
import packaging
from packaging import version
import inspect

from grid2op.tests.helper_path_test import *

PATH_ADN_CHRONICS_FOLDER = os.path.abspath(
    os.path.join(PATH_CHRONICS, "test_multi_chronics")
)
PATH_PREVIOUS_RUNNER = os.path.join(data_test_dir, "runner_data")

import grid2op
from grid2op.Agent import BaseAgent
from grid2op.Chronics import Multifolder, ChangeNothing
from grid2op.Reward import L2RPNReward, N1Reward
from grid2op.Backend import PandaPowerBackend
from grid2op.Runner.aux_fun import _aux_one_process_parrallel
from grid2op.Runner import Runner
from grid2op.dtypes import dt_float
from grid2op.Agent import RandomAgent
from grid2op.Episode import EpisodeData
from grid2op.Observation import BaseObservation, CompleteObservation

    
class AgentTestLegalAmbiguous(BaseAgent):
    def act(self, observation: BaseObservation, reward: float, done: bool = False):
        if observation.current_step == 1:
            return self.action_space({"set_line_status": [(0, -1)], "change_line_status": [0]})  # ambiguous
        if observation.current_step == 2:
            return self.action_space({"set_line_status": [(0, -1), (1, -1)]})  # illegal
        return super().act(observation, reward, done)
            
            
class TestRunner(HelperTests, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.init_grid_path = os.path.join(PATH_DATA_TEST_PP, "test_case14.json")
        self.path_chron = PATH_ADN_CHRONICS_FOLDER
        self.parameters_path = None
        self.max_iter = 10
        # self.real_reward = dt_float(199.99800)
        self.real_reward = dt_float(179.99818)
        self.all_real_rewards = [
            19.999783,
            19.999786,
            19.999784,
            19.999794,
            19.9998,
            19.999804,
            19.999804,
            19.999817,
            19.999823,
            0.0,
        ]
        self.names_chronics_to_backend = {
            "loads": {
                "2_C-10.61": "load_1_0",
                "3_C151.15": "load_2_1",
                "14_C63.6": "load_13_2",
                "4_C-9.47": "load_3_3",
                "5_C201.84": "load_4_4",
                "6_C-6.27": "load_5_5",
                "9_C130.49": "load_8_6",
                "10_C228.66": "load_9_7",
                "11_C-138.89": "load_10_8",
                "12_C-27.88": "load_11_9",
                "13_C-13.33": "load_12_10",
            },
            "lines": {
                "1_2_1": "0_1_0",
                "1_5_2": "0_4_1",
                "9_10_16": "8_9_2",
                "9_14_17": "8_13_3",
                "10_11_18": "9_10_4",
                "12_13_19": "11_12_5",
                "13_14_20": "12_13_6",
                "2_3_3": "1_2_7",
                "2_4_4": "1_3_8",
                "2_5_5": "1_4_9",
                "3_4_6": "2_3_10",
                "4_5_7": "3_4_11",
                "6_11_11": "5_10_12",
                "6_12_12": "5_11_13",
                "6_13_13": "5_12_14",
                "4_7_8": "3_6_15",
                "4_9_9": "3_8_16",
                "5_6_10": "4_5_17",
                "7_8_14": "6_7_18",
                "7_9_15": "6_8_19",
            },
            "prods": {
                "1_G137.1": "gen_0_4",
                "3_G36.31": "gen_2_1",
                "6_G63.29": "gen_5_2",
                "2_G-56.47": "gen_1_0",
                "8_G40.43": "gen_7_3",
            },
        }
        self.gridStateclass = Multifolder
        self.backendClass = PandaPowerBackend
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore"
            )  # silence the warning about missing layout
            self.runner = Runner(
                init_grid_path=self.init_grid_path,
                init_env_path=self.init_grid_path,
                path_chron=self.path_chron,
                parameters_path=self.parameters_path,
                names_chronics_to_backend=self.names_chronics_to_backend,
                gridStateclass=self.gridStateclass,
                backendClass=self.backendClass,
                rewardClass=L2RPNReward,
                max_iter=self.max_iter,
                name_env="test_runner_env",
            )

    # def test_one_episode(self):  # tested in the runner fast
    # def test_one_episode_detailed(self):  # tested in the runner fast
    # def test_2episode(self):  # tested in the runner fast
    # def test_init_from_env(self):  # tested in the runner fast
    # def test_seed_seq(self):  # tested in the runner fast
    # def test_seed_par(self):  # tested in the runner fast

    def test_one_process_par(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = _aux_one_process_parrallel(
                self.runner,
                [0],
                0,
                env_seeds=None,
                agent_seeds=None,
                max_iter=self.max_iter,
            )
        assert len(res) == 1
        _, el1, el2, el3, el4 = res[0]
        assert el1 == "1"
        assert np.abs(el2 - self.real_reward) <= self.tol_one
        assert el3 == 10
        assert el4 == 10

    def test_2episode_2process(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner._run_parrallel(
                nb_episode=2, nb_process=2, max_iter=self.max_iter
            )
        assert len(res) == 2
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == self.max_iter
            assert np.abs(cum_reward - self.real_reward) <= self.tol_one

    def test_2episode_2process_with_id(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res_1 = self.runner._run_parrallel(
                nb_episode=2, nb_process=2, episode_id=[0, 1], max_iter=self.max_iter
            )
        assert len(res_1) == 2
        assert res_1[0][1] == "1"
        assert res_1[1][1] == "2"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res_2 = self.runner._run_parrallel(
                nb_episode=2, nb_process=2, episode_id=[1, 0], max_iter=self.max_iter
            )
        assert len(res_2) == 2
        assert res_2[0][1] == "2"
        assert res_2[1][1] == "1"

    def test_2episodes_with_id(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res_1 = self.runner.run(
                nb_episode=2, episode_id=[0, 1], max_iter=self.max_iter
            )
        assert len(res_1) == 2
        assert res_1[0][1] == "1"
        assert res_1[1][1] == "2"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res_2 = self.runner.run(
                nb_episode=2, episode_id=[1, 0], max_iter=self.max_iter
            )
        assert len(res_2) == 2
        assert res_2[0][1] == "2"
        assert res_2[1][1] == "1"

    def test_2episodes_with_id_str(self):
        env = self.runner.init_env()
        subpaths = env.chronics_handler.subpaths
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res_1 = self.runner.run(
                nb_episode=2,
                episode_id=[subpaths[0], subpaths[1]],
                max_iter=self.max_iter,
            )
        assert len(res_1) == 2
        assert res_1[0][1] == "1"
        assert res_1[1][1] == "2"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res_2 = self.runner.run(
                nb_episode=2,
                episode_id=[subpaths[1], subpaths[0]],
                max_iter=self.max_iter,
            )
        assert len(res_2) == 2
        assert res_2[0][1] == "2"
        assert res_2[1][1] == "1"

    def test_2episode_2process_detailed(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner.run(
                nb_episode=2,
                nb_process=2,
                max_iter=self.max_iter,
                add_detailed_output=True,
            )
        assert len(res) == 2
        for i, _, cum_reward, timestep, total_ts, episode_data in res:
            assert int(timestep) == self.max_iter
            assert np.abs(cum_reward - self.real_reward) <= self.tol_one
            for j in range(len(self.all_real_rewards)):
                assert (
                    np.abs(episode_data.rewards[j] - self.all_real_rewards[j])
                    <= self.tol_one
                )

    def test_add_detailed_output_first_obs(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner.run(
                nb_episode=1,
                nb_process=1,
                max_iter=self.max_iter,
                add_detailed_output=True,
            )
        assert res[0][-1].observations[0] is not None

    def test_multiprocess_windows_no_fail(self):
        """test that i can run multiple times parallel run of the same env (breaks on windows)"""
        nb_episode = 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__) as env:
                f = tempfile.mkdtemp()
                runner_params = env.get_params_for_runner()
                runner = Runner(**runner_params)
                res1 = runner.run(
                    path_save=f,
                    nb_episode=nb_episode,
                    nb_process=2,
                    max_iter=self.max_iter,
                )
                res2 = runner.run(
                    path_save=f,
                    nb_episode=nb_episode,
                    nb_process=1,
                    max_iter=self.max_iter,
                )
                res3 = runner.run(
                    path_save=f,
                    nb_episode=nb_episode,
                    nb_process=2,
                    max_iter=self.max_iter,
                )
        test_ = set()
        for id_chron, name_chron, cum_reward, nb_time_step, max_ts in res1:
            test_.add(name_chron)
        assert len(test_) == nb_episode
        test_ = set()
        for id_chron, name_chron, cum_reward, nb_time_step, max_ts in res2:
            test_.add(name_chron)
        assert len(test_) == nb_episode
        test_ = set()
        for id_chron, name_chron, cum_reward, nb_time_step, max_ts in res3:
            test_.add(name_chron)
        assert len(test_) == nb_episode

    def test_complex_agent(self):
        nb_episode = 4
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__) as env:
                f = tempfile.mkdtemp()
                runner_params = env.get_params_for_runner()
                runner = Runner(**runner_params)
                res = runner.run(
                    path_save=f,
                    nb_episode=nb_episode,
                    nb_process=2,
                    max_iter=self.max_iter,
                )
        test_ = set()
        for id_chron, name_chron, cum_reward, nb_time_step, max_ts in res:
            test_.add(name_chron)
        assert len(test_) == nb_episode

    def test_init_from_env_with_other_reward(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "rte_case14_test", test=True, other_rewards={"test": L2RPNReward},
                _add_to_name=type(self).__name__
            ) as env:
                runner = Runner(**env.get_params_for_runner())
        res = runner.run(nb_episode=1, max_iter=self.max_iter)
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
            with grid2op.make("rte_case14_test", test=True, _add_to_name=type(self).__name__) as env:
                my_agent = TestSuitAgent(env.action_space)
                runner = Runner(
                    **env.get_params_for_runner(),
                    agentClass=None,
                    agentInstance=my_agent,
                )

        # test that the right seeds are assigned to the agent
        res = runner.run(
            nb_episode=3,
            max_iter=self.max_iter,
            env_seeds=[1, 2, 3],
            agent_seeds=[5, 6, 7],
        )
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
            with grid2op.make("rte_case14_test", test=True, _add_to_name=type(self).__name__) as env:
                runner = Runner(**env.get_params_for_runner())
        res = runner.run(
            nb_episode=2,
            nb_process=2,
            max_iter=self.max_iter,
            env_seeds=[1, 2],
            agent_seeds=[3, 4],
        )
        first_ = [el[0] for el in res]
        res = runner.run(
            nb_episode=2,
            nb_process=1,
            max_iter=self.max_iter,
            env_seeds=[1, 2],
            agent_seeds=[3, 4],
        )
        second_ = [el[0] for el in res]
        res = runner.run(
            nb_episode=2, nb_process=1, max_iter=self.max_iter, env_seeds=[9, 10]
        )
        third_ = [el[0] for el in res]
        res = runner.run(
            nb_episode=2,
            nb_process=2,
            max_iter=self.max_iter,
            env_seeds=[1, 2],
            agent_seeds=[3, 4],
        )
        fourth_ = [el[0] for el in res]
        assert np.all(first_ == second_)
        assert np.all(first_ == third_)
        assert np.all(first_ == fourth_)

    def test_nomaxiter(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case14_test", test=True, _add_to_name=type(self).__name__) as env:
                env.set_max_iter(2 * self.max_iter)
                runner = Runner(**env.get_params_for_runner())
        res = runner.run(nb_episode=1)
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == 2 * self.max_iter

    def test_nomaxiter_par(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case14_test", test=True, _add_to_name=type(self).__name__) as env:
                dict_ = env.get_params_for_runner()
                dict_["max_iter"] = -1
                sub_dict = dict_["gridStateclass_kwargs"]
                sub_dict["max_iter"] = 2 * self.max_iter
                runner = Runner(**dict_)
        res = runner.run(nb_episode=2, nb_process=2)
        for i, _, cum_reward, timestep, total_ts in res:
            assert int(timestep) == 2 * self.max_iter

    def _aux_backward(self, base_path, g2op_version_txt, g2op_version):
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

    def test_backward_compatibility(self):
        backward_comp_version = [
            "1.0.0",
            "1.1.0",
            "1.1.1",
            "1.2.0",
            "1.2.1",
            "1.2.2",
            "1.2.3",
            "1.3.0",
            "1.3.1",
            "1.4.0",
            "1.5.0",
            "1.5.1",
            "1.5.1.post1",
            "1.5.2",
            "1.6.0",
            "1.6.0.post1",
            "1.6.1",
            "1.6.2",
            "1.6.2.post1",
            "1.6.3",
            "1.6.4",
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
            "1.10.2",
            "1.10.3",
            "1.10.4",
            "1.10.5",
        ]
        curr_version = "test_version"
        assert (
            "curtailment" in CompleteObservation.attr_list_vect
        ), "error at the beginning"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "rte_case5_example", test=True,
                _add_to_name=type(self).__name__
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

    def test_reward_as_object(self):
        L_ID = 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "l2rpn_case14_sandbox", reward_class=N1Reward(l_id=L_ID), test=True,
                _add_to_name=type(self).__name__
            )
        runner = Runner(**env.get_params_for_runner())
        runner.run(nb_episode=1, max_iter=10)
        env.close()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "l2rpn_case14_sandbox",
                other_rewards={f"line_{l_id}": N1Reward(l_id=l_id) for l_id in [0, 1]},
                test=True,
                _add_to_name=type(self).__name__
            )

        runner = Runner(**env.get_params_for_runner())
        runner.run(nb_episode=1, max_iter=10)
        env.close()

    def test_legal_ambiguous_regular(self):            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)   
                     
        runner = Runner(**env.get_params_for_runner(), agentClass=AgentTestLegalAmbiguous)
        env.close()
        res, *_ = runner.run(nb_episode=1, max_iter=10, add_detailed_output=True)
        ep_data = res[-1]
        # test the "legal" part
        assert ep_data.legal[0]
        assert ep_data.legal[1]
        assert not ep_data.legal[2]
        assert ep_data.legal[3]
        # test the ambiguous part
        assert not ep_data.ambiguous[0]
        assert ep_data.ambiguous[1]
        assert not ep_data.ambiguous[2]
        assert not ep_data.ambiguous[3]

    def test_legal_ambiguous_nofaststorage(self):            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True, chronics_class=ChangeNothing,
                               _add_to_name=type(self).__name__)   
                     
            runner = Runner(**env.get_params_for_runner(), agentClass=AgentTestLegalAmbiguous)
            env.close()
            res, *_ = runner.run(nb_episode=1, max_iter=10, add_detailed_output=True)
        ep_data = res[-1]
        # test the "legal" part
        assert ep_data.legal[0]
        assert ep_data.legal[1]
        assert not ep_data.legal[2]
        assert ep_data.legal[3]
        # test the ambiguous part
        assert not ep_data.ambiguous[0]
        assert ep_data.ambiguous[1]
        assert not ep_data.ambiguous[2]
        assert not ep_data.ambiguous[3]
        
    def test_get_params(self):
        """test the runner._get_params() function (used in multiprocessing context)
        can indeed make a runner with all its arguments modified (proper 'copy' of the runner)
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True, chronics_class=ChangeNothing,
                               _add_to_name=type(self).__name__)   
                     
            runner = Runner(**env.get_params_for_runner(), agentClass=AgentTestLegalAmbiguous)
        made_params = runner._get_params()
        ok_params =  inspect.signature(Runner.__init__).parameters
        for k in made_params.keys():
            assert k in ok_params, f"params {k} is returned in runner._get_params() but cannot be used to make a runner"
            
        for k in ok_params.keys():
            if k == "self":
                continue
            assert k in made_params, f"params {k} is used to make a runner but is not returned in runner._get_params()"
        
        


if __name__ == "__main__":
    unittest.main()
