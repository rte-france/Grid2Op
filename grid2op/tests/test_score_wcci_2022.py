# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import numpy as np

import grid2op
from grid2op.Agent.baseAgent import BaseAgent
from grid2op.Agent.doNothing import DoNothingAgent
from grid2op.Reward import L2RPNWCCI2022ScoreFun
from grid2op.utils import ScoreL2RPN2022

import pdb


class AgentTester(BaseAgent):
    def act(self, observation, reward, done):
        if observation.current_step == 0:
            return self.action_space()
        if observation.current_step >= 13:
            return self.action_space()
        return self.action_space({"set_storage": [(0, 1.), (1, -1.)]})
    
    
class WCCI2022Tester(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = 0
        self.scen_id = 0
        self.nb_scenario = 2
        self.max_iter = 13
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage", test=True, reward_class=L2RPNWCCI2022ScoreFun)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def _aux_reset_env(self):    
        self.env.seed(self.seed)
        self.env.set_id(self.scen_id)
        obs = self.env.reset()
        return obs
    
    def test_storage_cost(self):
        """basic tests for L2RPNWCCI2022ScoreFun"""
        score_fun = L2RPNWCCI2022ScoreFun()
        score_fun.initialize(self.env)
        th_val = 10. * 10. / 12.
        
        obs = self._aux_reset_env()
        act = self.env.action_space({"set_storage": [(0, -5.), (1, 5.)]})
        obs, reward, done, info = self.env.step(act)
        rew = score_fun(act, self.env, False, False, False, False)
        margin_cost =  score_fun._get_marginal_cost(self.env)
        assert margin_cost == 70.
        storage_cost = score_fun._get_storage_cost(self.env, margin_cost)
        assert abs(storage_cost - th_val) <= 1e-5  # (10 MWh )* (10 € / MW )* (1/12. step / h)
        gen_p = 1.0 * obs.gen_p
        
        _ = self._aux_reset_env()
        obs, reward_dn, done, info = self.env.step(self.env.action_space())
        gen_p_dn = 1.0 * obs.gen_p
        
        assert reward >= reward_dn
        assert abs(reward - (reward_dn + storage_cost + (gen_p.sum() - gen_p_dn.sum()) * margin_cost / 12. )) <= 1e-6
    
    def test_storage_cost_2(self):
        """basic tests for L2RPNWCCI2022ScoreFun, when changin storage cost"""
        storage_cost = 100.
        self.env.close()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage", test=True,
                                    reward_class=L2RPNWCCI2022ScoreFun(storage_cost=storage_cost))
        score_fun = L2RPNWCCI2022ScoreFun(storage_cost=storage_cost)
        score_fun.initialize(self.env)
        th_val = storage_cost * 10. / 12.
        
        obs = self._aux_reset_env()
        act = self.env.action_space({"set_storage": [(0, -5.), (1, 5.)]})
        obs, reward, done, info = self.env.step(act)
        rew = score_fun(act, self.env, False, False, False, False)
        margin_cost =  score_fun._get_marginal_cost(self.env)
        assert margin_cost == 70.
        storage_cost = score_fun._get_storage_cost(self.env, margin_cost)
        assert abs(storage_cost - th_val) <= 1e-5  # (10 MWh )* (storage_cost € / MW )* (1/12. step / h)
        gen_p = 1.0 * obs.gen_p
        
        _ = self._aux_reset_env()
        obs, reward_dn, done, info = self.env.step(self.env.action_space())
        gen_p_dn = 1.0 * obs.gen_p
        
        assert reward >= reward_dn
        assert abs(reward - (reward_dn + storage_cost + (gen_p.sum() - gen_p_dn.sum()) * margin_cost / 12. )) <= 1e-6
        
    def test_score_helper(self):
        """basic tests for ScoreL2RPN2022 class"""        
        my_score = ScoreL2RPN2022(self.env,
                                  nb_scenario=self.nb_scenario,
                                  env_seeds=[0 for _ in range(self.nb_scenario)],
                                  agent_seeds=[0 for _ in range(self.nb_scenario)],
                                  max_step=self.max_iter,
                                  )
        try:
            res_dn = my_score.get(DoNothingAgent(self.env.action_space))
            res_agent = my_score.get(AgentTester(self.env.action_space))
            for scen_id, (score_dn, score_agent) in enumerate(zip(res_dn[0], res_agent[0])):
                assert score_agent < score_dn, f"error for scenario id {scen_id}"
            assert np.all(np.abs(np.array(res_agent[0]) - np.array([-0.007520790059641119, -0.00823946207038134])) <= 1e-6)
        finally:
            my_score.clear_all()
        
    def test_score_helper_2(self):
        """basic tests for ScoreL2RPN2022 class when changing storage cost"""
        storage_cost = 100.
        my_score = ScoreL2RPN2022(self.env,
                                  nb_scenario=self.nb_scenario,
                                  env_seeds=[0 for _ in range(self.nb_scenario)],
                                  agent_seeds=[0 for _ in range(self.nb_scenario)],
                                  max_step=self.max_iter,
                                  scores_func=L2RPNWCCI2022ScoreFun(storage_cost=storage_cost)
                                  )
        
        try:
            res_dn = my_score.get(DoNothingAgent(self.env.action_space))
            res_agent = my_score.get(AgentTester(self.env.action_space))
            for scen_id, (score_dn, score_agent) in enumerate(zip(res_dn[0], res_agent[0])):
                assert score_agent < score_dn, f"error for scenario id {scen_id}"
            assert np.all(np.abs(np.array(res_agent[0]) - np.array([-0.07931602, -0.08532347])) <= 1e-6)
        finally:
            my_score.clear_all()
        
        
if __name__ == "__main__":
    unittest.main()        
