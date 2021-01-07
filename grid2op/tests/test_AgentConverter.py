# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import json
import re
import warnings
import pdb
from abc import ABC, abstractmethod

from grid2op.tests.helper_path_test import *
from grid2op.Agent import AgentWithConverter, MLAgent
from grid2op.Converter import IdToAct
from grid2op.Rules import AlwaysLegal
from grid2op import make
from grid2op.Parameters import Parameters

import warnings
warnings.simplefilter("error")


class TestAgent(AgentWithConverter):
    def __init__(self, action_space, env_name, action_space_converter=IdToAct, **kwargs_converter):
        AgentWithConverter.__init__(self, action_space, action_space_converter=action_space_converter, **kwargs_converter)
        self.action_space.all_actions = []

        # do nothing
        all_actions_tmp = [action_space()]

        # powerline switch: disconnection
        for i in range(action_space.n_line):
            if env_name == "case14_realistic":
                if i == 18:
                    continue
            if env_name == "case5_example":
                pass
            all_actions_tmp.append(action_space.disconnect_powerline(line_id=i))

        # other type of actions
        all_actions_tmp += action_space.get_all_unitary_topologies_set(action_space)
        # self.action_space.all_actions += action_space.get_all_unitary_redispatch(action_space)

        if env_name == "case14_realistic":
            # remove action that makes the powerflow diverge
            breaking_acts = [action_space({"set_bus": {"lines_or_id": [(7,2), (8,1), (9,1)],
                                                       "lines_ex_id": [(17,2)],
                                                       "generators_id": [(2,2)],
                                                       "loads_id": [(4,1)]}}),
                             action_space({"set_bus": {"lines_or_id": [(10, 2), (11, 1), (19,2)],
                                                       "lines_ex_id": [(16, 2)],
                                                       "loads_id": [(5, 1)]}}),
                             action_space({"set_bus": {"lines_or_id": [(5, 1)],
                                                       "lines_ex_id": [(2, 2)],
                                                       "generators_id": [(1, 2)],
                                                       "loads_id": [(1, 1)]}}),
                             action_space({"set_bus": {"lines_or_id": [(6, 2), (15, 2), (16, 1)],
                                                       "lines_ex_id": [(3, 2), (5, 2)],
                                                       "loads_id": [(2, 1)]}})
            ]
        else:
            breaking_acts = [action_space({"set_bus": {"lines_or_id": [(0,2), (1,2), (2,2), (3,1)],
                                                       "generators_id": [(0,1)],
                                                       "loads_id": [(0,1)]}}),
                             ]

        # filter out actions that break everything
        all_actions = []
        for el in all_actions_tmp:
            if not el in breaking_acts:
                all_actions.append(el)

        # set the action to the action space
        self.action_space.all_actions = all_actions

        # add the action "reset everything to 1 bus"
        self.action_space.all_actions.append(action_space({"set_bus": np.ones(action_space.dim_topo, dtype=np.int),
                                                           "set_line_status": np.ones(action_space.n_line, dtype=np.int)}))
        self.nb_act_done = 0
        self.act_this = True

    def my_act(self, transformed_obs, reward, done=False):
        if self.act_this:
            res = self.nb_act_done
            self.nb_act_done += 1
            self.nb_act_done %= len(self.action_space.all_actions)
            self.act_this = False
        else:
            res = -1
            self.act_this = True
        return res


class TestBasicConverter(unittest.TestCase):
    def test_create_id2act(self):
        param = Parameters()
        param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("rte_case5_example", param=param, gamerules_class=AlwaysLegal, test=True)
        my_agent = TestAgent(env.action_space, "rte_case5_example")
        obs = env.reset()
        for i in range(10):
            act = my_agent.act(obs, 0, False)
            obs, reward, done, info = env.step(act)
        env.close()

    def test_create_to_vect(self):
        param = Parameters()
        param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("rte_case5_example", param=param, gamerules_class=AlwaysLegal, test=True)
        my_agent = MLAgent(env.action_space)
        obs = env.reset()
        for i in range(10):
            act = my_agent.act(obs, 0, False)
            obs, reward, done, info = env.step(act)
        env.close()


if __name__ == "__main__":
    unittest.main()