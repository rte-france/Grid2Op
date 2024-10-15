# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import time
import warnings
import pandapower as pp
import unittest

from grid2op.tests.helper_path_test import *

import grid2op
from grid2op.Exceptions import *
from grid2op.Agent import (
    PowerLineSwitch,
    TopologyGreedy,
    DoNothingAgent,
    RecoPowerlineAgent,
    FromActionsListAgent,
)
from grid2op.Parameters import Parameters
from grid2op.dtypes import dt_float
from grid2op.Agent import RandomAgent

import pdb

DEBUG = False

if DEBUG:
    print("pandapower version : {}".format(pp.__version__))


class TestAgent(HelperTests, unittest.TestCase):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        super().setUp()
        param = Parameters()
        param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("rte_case14_redisp", test=True, param=param, _add_to_name=type(self).__name__)

    def tearDown(self):
        self.env.close()
        super().tearDown()

    def _aux_test_agent(self, agent, i_max=30):
        done = False
        i = 0
        beg_ = time.perf_counter()
        cum_reward = dt_float(0.0)
        obs = self.env.get_obs()
        reward = 0.0
        time_act = 0.0
        all_acts = []
        while not done:
            # print("_______________")
            beg__ = time.perf_counter()
            act = agent.act(obs, reward, done)
            all_acts.append(act)
            end__ = time.perf_counter()
            obs, reward, done, info = self.env.step(
                act
            )  # should load the first time stamp
            time_act += end__ - beg__
            cum_reward += reward
            i += 1
            if i > i_max:
                break

        end_ = time.perf_counter()
        if DEBUG:
            li_text = [
                "Env: {:.2f}s",
                "\t - apply act {:.2f}s",
                "\t - run pf: {:.2f}s",
                "\t - env update + observation: {:.2f}s",
                "\t - time env obs space: {:.2f}s",
                "BaseAgent: {:.2f}s",
                "Total time: {:.2f}s",
                "Cumulative reward: {:1f}",
            ]
            msg_ = "\n".join(li_text)
            print(
                msg_.format(
                    self.env._time_apply_act
                    + self.env._time_powerflow
                    + self.env._time_extract_obs,  # env
                    self.env._time_apply_act,  # apply act
                    self.env._time_powerflow,  # run pf
                    self.env._time_extract_obs,  # env update + obs
                    self.env.observation_space._update_env_time,  # time get topo vect
                    time_act,
                    end_ - beg_,
                    cum_reward,
                )
            )
        return i, cum_reward, all_acts

    def test_0_donothing(self):
        agent = DoNothingAgent(self.env.action_space)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            i, cum_reward, all_acts = self._aux_test_agent(agent)
        assert i == 31, "The powerflow diverged before step 30 for do nothing"
        expected_reward = dt_float(35140.027)
        expected_reward = dt_float(35140.03125 / 12.)
        assert (
            np.abs(cum_reward - expected_reward, dtype=dt_float) <= self.tol_one
        ), f"The reward has not been properly computed {cum_reward} instead of {expected_reward}"

    def test_1_powerlineswitch(self):
        agent = PowerLineSwitch(self.env.action_space)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            i, cum_reward, all_acts = self._aux_test_agent(agent)
        assert (
            i == 31
        ), "The powerflow diverged before step 30 for powerline switch agent"
        # switch to using df_float in the reward, change then the results
        expected_reward = dt_float(35147.55859375)  
        expected_reward = dt_float(35147.7685546 / 12.)
        assert (
            np.abs(cum_reward - expected_reward) <= self.tol_one
        ), f"The reward has not been properly computed {cum_reward} instead of {expected_reward}"

    def test_2_busswitch(self):
        agent = TopologyGreedy(self.env.action_space)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            i, cum_reward, all_acts = self._aux_test_agent(agent, i_max=10)
        assert i == 11, "The powerflow diverged before step 10 for greedy agent"
        # i have more actions now, so this is not correct (though it should be..
        # yet a proof that https://github.com/Grid2Op/grid2op/issues/86 is grounded
        expected_reward = dt_float(12075.389)
        expected_reward = dt_float(12277.632)
        expected_reward = dt_float(12076.35644531 / 12.)
        # 1006.363037109375
        #: Breaking change in 1.10.0: topology are not in the same order
        expected_reward = dt_float(1006.34924)  
        assert (
            np.abs(cum_reward - expected_reward) <= self.tol_one
        ), f"The reward has not been properly computed {cum_reward} instead of {expected_reward}"


class TestMake2Agents(HelperTests, unittest.TestCase):
    def test_2random(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__)
            env2 = grid2op.make("rte_case14_realistic", test=True, _add_to_name=type(self).__name__)
        agent = RandomAgent(env.action_space)
        agent2 = RandomAgent(env2.action_space)
        # test i can reset the env
        obs = env.reset()
        obs2 = env2.reset()
        # test the agent can act
        act = agent.act(obs, 0.0, False)
        act2 = agent2.act(obs2, 0.0, False)
        # test the env can step
        _ = env.step(act)
        _ = env2.step(act2)
        env.close()
        env2.close()


class TestSeeding(HelperTests, unittest.TestCase):
    def test_random(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__) as env:
                obs = env.reset()
                my_agent = RandomAgent(env.action_space)
                my_agent.seed(0)
                nb_test = 100
                res = np.zeros(nb_test, dtype=int)
                res2 = np.zeros(nb_test, dtype=int)
                res3 = np.zeros(nb_test, dtype=int)
                for i in range(nb_test):
                    res[i] = my_agent.my_act(obs, 0.0, False)
                my_agent.seed(0)
                for i in range(nb_test):
                    res2[i] = my_agent.my_act(obs, 0.0, False)
                my_agent.seed(1)
                for i in range(nb_test):
                    res3[i] = my_agent.my_act(obs, 0.0, False)

                # the same seeds should produce the same sequence
                assert np.all(res == res2)
                # different seeds should produce different sequence
                assert np.any(res != res3)


class TestRecoPowerlineAgent(HelperTests, unittest.TestCase):
    def test_reco_simple(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        param.NB_TIMESTEP_COOLDOWN_LINE = 1
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, param=param, _add_to_name=type(self).__name__) as env:
                my_agent = RecoPowerlineAgent(env.action_space)
                obs = env.reset()
                assert np.sum(obs.time_before_cooldown_line) == 0
                obs, reward, done, info = env.step(
                    env.action_space({"set_line_status": [(1, -1)]})
                )
                assert np.sum(obs.time_before_cooldown_line) == 1
                # the agent should do nothing, as the line is still in cooldown
                act = my_agent.act(obs, reward, done)
                assert not act.as_dict()
                obs, reward, done, info = env.step(act)
                # now cooldown is over
                assert np.sum(obs.time_before_cooldown_line) == 0
                act2 = my_agent.act(obs, reward, done)
                ddict = act2.as_dict()
                assert "set_line_status" in ddict
                assert "nb_connected" in ddict["set_line_status"]
                assert "connected_id" in ddict["set_line_status"]
                assert ddict["set_line_status"]["nb_connected"] == 1
                assert ddict["set_line_status"]["connected_id"][0] == 1

    def test_reco_more_difficult(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        param.NB_TIMESTEP_COOLDOWN_LINE = 3
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, param=param, _add_to_name=type(self).__name__) as env:
                my_agent = RecoPowerlineAgent(env.action_space)
                obs = env.reset()
                obs, reward, done, info = env.step(
                    env.action_space({"set_line_status": [(1, -1)]})
                )
                obs, reward, done, info = env.step(
                    env.action_space({"set_line_status": [(2, -1)]})
                )

                # the agent should do nothing, as the line is still in cooldown
                act = my_agent.act(obs, reward, done)
                assert not act.as_dict()
                obs, reward, done, info = env.step(act)
                act = my_agent.act(obs, reward, done)
                assert not act.as_dict()
                obs, reward, done, info = env.step(act)
                # now in theory i can reconnect the first one
                act2 = my_agent.act(obs, reward, done)
                ddict = act2.as_dict()
                assert "set_line_status" in ddict
                assert "nb_connected" in ddict["set_line_status"]
                assert "connected_id" in ddict["set_line_status"]
                assert ddict["set_line_status"]["nb_connected"] == 1
                assert ddict["set_line_status"]["connected_id"][0] == 1

                # but i will not implement it on the grid
                obs, reward, done, info = env.step(env.action_space())

                act3 = my_agent.act(obs, reward, done)
                ddict3 = act3.as_dict()
                assert len(my_agent.tested_action) == 2
                # and it turns out i need to reconnect the first one first
                assert "set_line_status" in ddict3
                assert "nb_connected" in ddict3["set_line_status"]
                assert "connected_id" in ddict3["set_line_status"]
                assert ddict3["set_line_status"]["nb_connected"] == 1
                assert ddict3["set_line_status"]["connected_id"][0] == 1

                obs, reward, done, info = env.step(act3)

                act4 = my_agent.act(obs, reward, done)
                ddict4 = act4.as_dict()
                assert len(my_agent.tested_action) == 1
                # and it turns out i need to reconnect the first one first
                assert "set_line_status" in ddict4
                assert "nb_connected" in ddict4["set_line_status"]
                assert "connected_id" in ddict4["set_line_status"]
                assert ddict4["set_line_status"]["nb_connected"] == 1
                assert ddict4["set_line_status"]["connected_id"][0] == 2


class TestFromList(HelperTests, unittest.TestCase):
    def test_agentfromlist_empty(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        param.NB_TIMESTEP_COOLDOWN_LINE = 3
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, param=param, _add_to_name=type(self).__name__) as env:
                agent = FromActionsListAgent(env.action_space, action_list=[])
                obs = env.reset()

                # should do nothing
                act = agent.act(obs, 0.0, False)
                obs, reward, done, info = env.step(act)
                assert act.can_affect_something() is False

                act = agent.act(obs, 0.0, False)
                obs, reward, done, info = env.step(act)
                assert act.can_affect_something() is False

    def test_agentfromlist(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        param.NB_TIMESTEP_COOLDOWN_LINE = 3
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, param=param, _add_to_name=type(self).__name__) as env:
                agent = FromActionsListAgent(
                    env.action_space,
                    action_list=[env.action_space({"set_line_status": [(0, +1)]})],
                )
                obs = env.reset()

                # should do nothing
                act = agent.act(obs, 0.0, False)
                obs, reward, done, info = env.step(act)
                assert act == env.action_space({"set_line_status": [(0, +1)]})

                act = agent.act(obs, 0.0, False)
                obs, reward, done, info = env.step(act)
                assert act.can_affect_something() is False

    def test_agentfromlist_creation_fails(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        param.NB_TIMESTEP_COOLDOWN_LINE = 3
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, param=param, _add_to_name=type(self).__name__) as env:
                with self.assertRaises(AgentError):
                    # action_list should be an iterable
                    agent = FromActionsListAgent(env.action_space, action_list=1)
                with self.assertRaises(AgentError):
                    # action_list should contain only actions
                    agent = FromActionsListAgent(env.action_space, action_list=[1])

                with grid2op.make(
                    "l2rpn_case14_sandbox", test=True, param=param,
                    _add_to_name=type(self).__name__
                ) as env2:
                    with self.assertRaises(AgentError):
                        # action_list should contain only actions from a compatible environment
                        agent = FromActionsListAgent(
                            env.action_space,
                            action_list=[
                                env2.action_space({"set_line_status": [(0, +1)]})
                            ],
                        )

                with grid2op.make(
                    "rte_case5_example", test=True, param=param, _add_to_name="toto"
                ) as env3:
                    # this should work because it's the same underlying grid
                    agent = FromActionsListAgent(
                        env.action_space,
                        action_list=[env3.action_space({"set_line_status": [(0, +1)]})],
                    )


if __name__ == "__main__":
    unittest.main()
