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

from grid2op.tests.helper_path_test import *

import grid2op
from grid2op.Exceptions import *
from grid2op.MakeEnv import make
from grid2op.Agent import PowerLineSwitch, TopologyGreedy, DoNothingAgent
from grid2op.Parameters import Parameters
from grid2op.dtypes import dt_float
from grid2op.Agent import RandomAgent

import pdb

DEBUG = False

if DEBUG:
    print("pandapower version : {}".format(pp.__version__))


class TestAgent(HelperTests):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        param = Parameters()
        param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("rte_case14_redisp", test=True, param=param)

    def tearDown(self):
        self.env.close()

    def _aux_test_agent(self, agent, i_max=30):
        done = False
        i = 0
        beg_ = time.time()
        cum_reward = dt_float(0.0)
        obs = self.env.get_obs()
        reward = 0.
        time_act = 0.
        all_acts = []
        while not done:
            # print("_______________")
            beg__ = time.time()
            act = agent.act(obs, reward, done)
            all_acts.append(act)
            end__ = time.time()
            obs, reward, done, info = self.env.step(act)  # should load the first time stamp
            time_act += end__ - beg__
            cum_reward += reward
            # print("reward: {}".format(reward))
            # print("_______________")
            # if reward <= 0 or np.any(obs.prod_p < 0):
            #     pdb.set_trace()
            i += 1
            if i > i_max:
                break

        end_ = time.time()
        if DEBUG:
            li_text = ["Env: {:.2f}s",
                       "\t - apply act {:.2f}s",
                       "\t - run pf: {:.2f}s",
                       "\t - env update + observation: {:.2f}s",
                       "\t - time get topo vect: {:.2f}s",
                       "\t - time env obs space: {:.2f}s",
                       "BaseAgent: {:.2f}s", "Total time: {:.2f}s",
                       "Cumulative reward: {:1f}"]
            msg_ = "\n".join(li_text)
            print(msg_.format(
                self.env._time_apply_act+self.env._time_powerflow+self.env._time_extract_obs,  # env
                self.env._time_apply_act,  # apply act
                self.env._time_powerflow,  # run pf
                self.env._time_extract_obs,  # env update + obs
                self.env.backend._time_topo_vect,  # time get topo vect
                self.env.observation_space._update_env_time,  # time get topo vect
                time_act, end_-beg_, cum_reward))
        return i, cum_reward, all_acts

    def test_0_donothing(self):
        agent = DoNothingAgent(self.env.helper_action_player)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            i, cum_reward, all_acts = self._aux_test_agent(agent)
        assert i == 31, "The powerflow diverged before step 30 for do nothing"
        expected_reward = dt_float(35140.027)
        assert np.abs(cum_reward - expected_reward, dtype=dt_float) <= self.tol_one, "The reward has not been properly computed"

    def test_1_powerlineswitch(self):
        agent = PowerLineSwitch(self.env.helper_action_player)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            i, cum_reward, all_acts = self._aux_test_agent(agent)
        assert i == 31, "The powerflow diverged before step 30 for powerline switch agent"
        expected_reward = dt_float(35147.55859375)  # switch to using df_float in the reward, change then the results
        expected_reward = dt_float(35147.76)
        assert np.abs(cum_reward - expected_reward) <= self.tol_one, "The reward has not been properly computed"

    def test_2_busswitch(self):
        agent = TopologyGreedy(self.env.helper_action_player)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            i, cum_reward, all_acts = self._aux_test_agent(agent, i_max=10)
        assert i == 11, "The powerflow diverged before step 10 for greedy agent"
        expected_reward = dt_float(12075.389)  # i have more actions now, so this is not correct (though it should be..
        # yet a proof that https://github.com/rte-france/Grid2Op/issues/86 is grounded
        expected_reward = dt_float(12277.632)
        # 12076.356
        # 12076.191
        expected_reward = dt_float(12076.356)
        assert np.abs(cum_reward - expected_reward) <= self.tol_one, "The reward has not been properly computed"


class TestMake2Agents(HelperTests):
    def test_2random(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            env2 = grid2op.make("rte_case14_realistic", test=True)
        agent = RandomAgent(env.action_space)
        agent2 = RandomAgent(env2.action_space)
        # test i can reset the env
        obs = env.reset()
        obs2 = env2.reset()
        # test the agent can act
        act = agent.act(obs, 0., False)
        act2 = agent2.act(obs2, 0., False)
        # test the env can step
        _ = env.step(act)
        _ = env2.step(act2)


if __name__ == "__main__":
    unittest.main()
