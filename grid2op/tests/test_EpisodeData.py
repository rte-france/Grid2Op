# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import tempfile
import warnings
import pdb
import unittest

import grid2op
from grid2op.Agent import OneChangeThenNothing
from grid2op.tests.helper_path_test import *
from grid2op.Chronics import Multifolder
from grid2op.Reward import L2RPNReward
from grid2op.Backend import PandaPowerBackend
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData
from grid2op.dtypes import dt_float
from grid2op.Agent import BaseAgent
from grid2op.Action import TopologyAction
from grid2op.Parameters import Parameters
from grid2op.Opponent.baseActionBudget import BaseActionBudget
from grid2op.Opponent import RandomLineOpponent


DEBUG = True
PATH_ADN_CHRONICS_FOLDER = os.path.abspath(
    os.path.join(PATH_CHRONICS, "test_multi_chronics")
)


class TestEpisodeData(unittest.TestCase):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        self.tolvect = dt_float(1e-2)
        self.tol_one = dt_float(1e-5)
        self.max_iter = 10
        # self.real_reward = dt_float(199.99800)
        self.real_reward = dt_float(179.99818)

        self.init_grid_path = os.path.join(PATH_DATA_TEST_PP, "test_case14.json")
        self.path_chron = PATH_ADN_CHRONICS_FOLDER
        self.parameters_path = None
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
        self.runner = Runner(
            init_grid_path=self.init_grid_path,
            init_env_path=self.init_grid_path,
            path_chron=self.path_chron,
            parameters_path=self.parameters_path,
            names_chronics_to_backend=self.names_chronics_to_backend,
            gridStateclass=self.gridStateclass,
            backendClass=self.backendClass,
            rewardClass=L2RPNReward,
            other_rewards={"test": L2RPNReward},
            max_iter=self.max_iter,
            name_env="test_episodedata_env",
        )

    def test_load_ambiguous(self):
        f = tempfile.mkdtemp()

        class TestSuitAgent(BaseAgent):
            def __init__(self, *args, **kwargs):
                BaseAgent.__init__(self, *args, **kwargs)

            def act(self, observation, reward, done=False):
                # do a ambiguous action
                return self.action_space(
                    {"set_line_status": [(0, 1)], "change_line_status": [0]}
                )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case14_test", test=True, _add_to_name=type(self).__name__) as env:
                my_agent = TestSuitAgent(env.action_space)
                runner = Runner(
                    **env.get_params_for_runner(),
                    agentClass=None,
                    agentInstance=my_agent
                )

                # test that the right seeds are assigned to the agent
                res = runner.run(nb_episode=1, max_iter=self.max_iter, path_save=f)
            episode_data = EpisodeData.from_disk(agent_path=f, name=res[0][1])
        assert int(episode_data.meta["chronics_max_timestep"]) == self.max_iter
        assert len(episode_data.actions) == self.max_iter
        assert len(episode_data.observations) == self.max_iter + 1
        assert len(episode_data.env_actions) == self.max_iter
        assert len(episode_data.attacks) == self.max_iter

    def test_one_episode_with_saving(self):
        f = tempfile.mkdtemp()
        (
            ep_id,
            episode_name,
            cum_reward,
            timestep,
            max_ts
        ) = self.runner.run_one_episode(path_save=f)
        episode_data = EpisodeData.from_disk(agent_path=f, name=episode_name)
        assert int(episode_data.meta["chronics_max_timestep"]) == self.max_iter
        assert len(episode_data.other_rewards) == self.max_iter
        for other, real in zip(episode_data.other_rewards, episode_data.rewards):
            assert dt_float(np.abs(other["test"] - real)) <= self.tol_one
        assert (
            np.abs(dt_float(episode_data.meta["cumulative_reward"]) - self.real_reward)
            <= self.tol_one
        )

    def test_collection_wrapper_after_run(self):
        OneChange = OneChangeThenNothing.gen_next(
            {"set_bus": {"lines_or_id": [(1, -1)]}}
        )
        runner = Runner(
            init_grid_path=self.init_grid_path,
            init_env_path=self.init_grid_path,
            path_chron=self.path_chron,
            parameters_path=self.parameters_path,
            names_chronics_to_backend=self.names_chronics_to_backend,
            gridStateclass=self.gridStateclass,
            backendClass=self.backendClass,
            rewardClass=L2RPNReward,
            other_rewards={"test": L2RPNReward},
            max_iter=self.max_iter,
            name_env="test_episodedata_env",
            agentClass=OneChange,
        )
        _, _, cum_reward, timestep, max_ts, episode_data = runner.run_one_episode(
            max_iter=self.max_iter, detailed_output=True
        )
        # Check that the type of first action is set bus
        assert episode_data.actions[0].get_types()[2]

    def test_len(self):
        """test i can use the function "len" of the episode data"""
        f = tempfile.mkdtemp()
        (
            ep_id,
            episode_name,
            cum_reward,
            timestep,
            max_ts
        ) = self.runner.run_one_episode(path_save=f)
        episode_data = EpisodeData.from_disk(agent_path=f, name=episode_name)
        len(episode_data)

    def test_3_episode_with_saving(self):
        f = tempfile.mkdtemp()
        res = self.runner._run_sequential(nb_episode=3, path_save=f)
        for i, episode_name, cum_reward, timestep, total_ts in res:
            episode_data = EpisodeData.from_disk(agent_path=f, name=episode_name)
            assert int(episode_data.meta["chronics_max_timestep"]) == self.max_iter
            assert (
                np.abs(
                    dt_float(episode_data.meta["cumulative_reward"]) - self.real_reward
                )
                <= self.tol_one
            )

    def test_3_episode_3process_with_saving(self):
        f = tempfile.mkdtemp()
        nb_episode = 2
        res = self.runner._run_parrallel(
            nb_episode=nb_episode, nb_process=2, path_save=f
        )
        assert len(res) == nb_episode
        for i, episode_name, cum_reward, timestep, total_ts in res:
            episode_data = EpisodeData.from_disk(agent_path=f, name=episode_name)
            assert int(episode_data.meta["chronics_max_timestep"]) == self.max_iter
            assert (
                np.abs(
                    dt_float(episode_data.meta["cumulative_reward"]) - self.real_reward
                )
                <= self.tol_one
            )

    def test_with_opponent(self):
        init_budget = 1000
        opponent_attack_duration = 15
        opponent_attack_cooldown = 30
        opponent_budget_per_ts = 0.0
        opponent_action_class = TopologyAction

        LINES_ATTACKED = ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]

        p = Parameters()
        p.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "rte_case14_realistic",
                test=True,
                param=p,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=opponent_budget_per_ts,
                opponent_attack_cooldown=opponent_attack_cooldown,
                opponent_attack_duration=opponent_attack_duration,
                opponent_action_class=opponent_action_class,
                opponent_budget_class=BaseActionBudget,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__,
            )
        env.seed(0)
        runner = Runner(**env.get_params_for_runner())

        f = tempfile.mkdtemp()
        res = runner.run(
            nb_episode=1,
            env_seeds=[4],
            agent_seeds=[0],
            max_iter=opponent_attack_cooldown - 1,
            path_save=f,
        )

        episode_data = EpisodeData.from_disk(agent_path=f, name=res[0][1])
        lines_impacted, subs_impacted = episode_data.attacks[0].get_topological_impact()
        assert lines_impacted[3]


if __name__ == "__main__":
    unittest.main()
