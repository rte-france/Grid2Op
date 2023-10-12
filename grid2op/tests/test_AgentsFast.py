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

from grid2op.Exceptions import *
from grid2op.Agent import DoNothingAgent, BaseAgent
from grid2op.Parameters import Parameters
from grid2op.dtypes import dt_float

import pdb

DEBUG = False

if DEBUG:
    print("pandapower version : {}".format(pp.__version__))


class RandomTestAgent(BaseAgent):
    def act(self, observation, reward, done=False):
        return self.action_space.sample()


class TestAgentFaster(HelperTests, unittest.TestCase):
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
        super().tearDown()
        self.env.close()

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
            # print("reward: {}".format(reward))
            # print("_______________")
            # if reward <= 0 or np.any(obs.prod_p < 0):
            #     pdb.set_trace()
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
                "\t - time get topo vect: {:.2f}s",
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
                    self.env.backend._time_topo_vect,  # time get topo vect
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
        expected_reward = dt_float(35140.027 / 12.)
        expected_reward = dt_float(35140.03125 / 12.)
        assert (
            np.abs(cum_reward - expected_reward, dtype=dt_float) <= self.tol_one
        ), f"The reward has not been properly computed {cum_reward} instead of {expected_reward}"

    def test_1_random(self):
        agent = RandomTestAgent(self.env.action_space)
        seed = 72  # don't change that !
        agent.seed(seed)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            nb_steps, cum_reward, all_acts = self._aux_test_agent(agent)
        assert nb_steps == 16, "The powerflow diverged before step 16 for RandomTestAgent"
        expected_reward = dt_float(16441.488)
        expected_reward = dt_float(16331.4873046875 / 12.)
        expected_reward = dt_float(16331.54296875 / 12.)
        assert (
            np.abs(cum_reward - expected_reward, dtype=dt_float) <= self.tol_one
        ), f"The reward has not been properly computed {cum_reward} instead of {expected_reward}"
