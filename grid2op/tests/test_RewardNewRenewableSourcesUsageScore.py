# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import numpy as np
import unittest

import grid2op
from grid2op.Reward import _NewRenewableSourcesUsageScore
from grid2op.Agent import DoNothingAgent

class TestNewRenewableSourcesUsageScore(unittest.TestCase):
    
    def setUp(self) -> None:
        env_name = "l2rpn_case14_sandbox"
        self.env = grid2op.make(env_name, reward_class = _NewRenewableSourcesUsageScore)
        self.nres_id = np.arange(self.env.n_gen)[self.env.gen_renewable]
        
    def test_surlinear_function(self):
        #for recalls, use nres_ratio percentages between 50 and 100
        delta_x = 0.5
        x = np.arange(start=50, stop=100, step=delta_x)
        f_x = _NewRenewableSourcesUsageScore._surlinear_func_curtailment(x)
        gradient_f = (f_x[1:] - f_x[:-1]) / delta_x
        return all(gradient_f > 1)    
    
    def test_capitalization(self):

        my_agent = DoNothingAgent(self.env.action_space)
        done = False
        reward = self.env.reward_range[0]
        gen_res_p_curtailed_array = np.zeros(self.env.chronics_handler.max_timestep())
        gen_res_p_before_curtail_array = np.zeros(self.env.chronics_handler.max_timestep())
        obs = self.env.reset()
        while True:
            gen_res_p_curtailed_array[self.env.nb_time_step] = np.sum(obs.gen_p[self.env.gen_renewable])
            gen_res_p_before_curtail_array[self.env.nb_time_step] = np.sum(obs.gen_p_before_curtail[self.env.gen_renewable])
            action = my_agent.act(obs, reward, done)
            obs, reward, done, _ = self.env.step(action)
            if done:
                break
            
        return all(
            [
                all(self.env._reward_helper.template_reward.gen_res_p_curtailed_list == gen_res_p_curtailed_array),
                all(self.env._reward_helper.template_reward.gen_res_p_before_curtail_list == gen_res_p_before_curtail_array),
                reward == _NewRenewableSourcesUsageScore._surlinear_func_curtailment(100 * np.sum(gen_res_p_curtailed_array[1:]) / np.sum(gen_res_p_before_curtail_array[1:]))
            ]
        )
            
    
        
    