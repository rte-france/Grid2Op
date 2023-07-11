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
from grid2op.Agent import DoNothingAgent, BaseAgent

class CurtailTrackerAgent(BaseAgent):
    def __init__(self, action_space, gen_renewable, gen_pmax, curtail_level=1.):
        super().__init__(action_space)
        self.gen_renewable = gen_renewable
        self.gen_pmax = gen_pmax[gen_renewable]
        self.curtail_level = curtail_level
        
    def act(self, obs, reward, done):
        curtail_target = self.curtail_level * obs.gen_p_before_curtail[self.gen_renewable] / self.gen_pmax
        act = self.action_space(
            {"curtail": [(el, ratio) for el, ratio in zip(np.arange(len(self.gen_renewable))[self.gen_renewable], curtail_target)]}
        )
        return act
    
class DoNothingSimulatorAgent(DoNothingAgent):
    def __init__(self, action_space, nres_id, gen_pmax):
        super().__init__(action_space)
        self.nres_id = nres_id
        self.gen_pmax = gen_pmax
        
    def act(self, obs, reward, done):
        curtail_target = 0.5 * obs.gen_p_before_curtail[self.nres_id] / self.gen_pmax[self.nres_id]
        act = self.action_space(
            {"curtail": [(el, ratio) for el, ratio in zip(self.nres_id, curtail_target)]}
        )
        sim_obs_1, *_ = obs.simulate(act, time_step=1)
        return super().act(obs, reward, done)

class TestNewRenewableSourcesUsageScore(unittest.TestCase):
    
    def setUp(self) -> None:
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(env_name,
                                    reward_class = _NewRenewableSourcesUsageScore,
                                    test=True
                                )
            self.env.set_max_iter(20)
            self.env.parameters.NO_OVERFLOW_DISCONNECTION = True
        self.nres_id = np.arange(self.env.n_gen)[self.env.gen_renewable]
        
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
        
    def test_surlinear_function(self):
        #for recalls, use nres_ratio percentages between 50 and 100
        delta_x = 0.5
        x = np.arange(start=50, stop=100, step=delta_x)
        f_x = _NewRenewableSourcesUsageScore._surlinear_func_curtailment(x)
        gradient_f = (f_x[1:] - f_x[:-1]) / delta_x
        assert all(gradient_f > 1 / 50)
        assert all(np.equal(np.argsort(gradient_f),np.arange(len(gradient_f), dtype=int)))    
    
    def test_capitalization_score(self):

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
                np.array_equal(self.env._reward_helper.template_reward.gen_res_p_curtailed_list, gen_res_p_curtailed_array),
                np.array_equal(self.env._reward_helper.template_reward.gen_res_p_before_curtail_list, gen_res_p_before_curtail_array),
                reward == _NewRenewableSourcesUsageScore._surlinear_func_curtailment(100 * np.sum(gen_res_p_curtailed_array[1:]) / np.sum(gen_res_p_before_curtail_array[1:]))
            ]
        )
        
    def test_reward_after_blackout(self):
        for blackout_time_step in [1,3,10]:
            my_agent = DoNothingAgent(self.env.action_space)
            done = False
            reward = self.env.reward_range[0]
            obs = self.env.reset()
            while True:
                if self.env.nb_time_step + 1 > blackout_time_step:
                    blackout_act = {"set_bus": {"generators_id": (0,-1)}}
                    action = self.env.action_space(blackout_act)
                else:
                    action = my_agent.act(obs, reward, done)
                obs, reward, done, _ = self.env.step(action)
                if done:
                    break
            assert reward == 1.
        
    def test_reward_value(self):
        for curtail_target, ratio_curtail_expected in [
            (0.5, 50.84402431116107),
            (0.65, 66.09722918973647),
            (0.8, 81.35044050770632),
            (0.9, 91.51924150630187),
            (1., 99.96623954270511)
            ]:
            my_agent = CurtailTrackerAgent(self.env.action_space,
                                           gen_renewable = self.env.gen_renewable,
                                           gen_pmax=self.env.gen_pmax,
                                           curtail_level = curtail_target)
            self.env.seed(0)
            self.env.set_id(0)
            obs = self.env.reset()
            done = False
            reward = self.env.reward_range[0]      
            while True:
                action = my_agent.act(obs, reward, done)
                obs, reward, done, _ = self.env.step(action)
                if done:
                    break
            assert reward == _NewRenewableSourcesUsageScore._surlinear_func_curtailment(ratio_curtail_expected)
        
    def test_simulate_ignored(self):
        my_agent = DoNothingSimulatorAgent(self.env.action_space,
                                           nres_id = np.arange(self.env.n_gen)[self.env.gen_renewable],
                                           gen_pmax=self.env.gen_pmax,)
        done = False
        reward = self.env.reward_range[0]
        obs = self.env.reset()
        while True:
            action = my_agent.act(obs, reward, done)
            obs, reward, done, _ = self.env.step(action)
            if done:
                break
            
        return reward == 1.
    
    def test_simulate_blackout_ignored(self):
        obs = self.env.reset()
        obs, reward, done, _ = self.env.step(self.env.action_space())
        go_act = self.env.action_space({"set_bus": {"generators_id": [(0, -1)]}})
        simO, simr, simd, simi = obs.simulate(go_act)
        assert simr == 0., f"{simr} vs 0."
        assert simd
    
    def test_simulated_env(self):
        obs = self.env.reset()
        f_env = obs.get_forecast_env()
        forD = False
        while not forD:
            forO, forR, forD, forI = f_env.step(self.env.action_space())
            assert forR == 0.
            
    
if __name__ == "__main__":
    unittest.main()        
