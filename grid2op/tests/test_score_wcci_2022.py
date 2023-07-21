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
from grid2op.Agent import (BaseAgent, DoNothingAgent)
from grid2op.Reward import L2RPNWCCI2022ScoreFun
from grid2op.utils import ScoreL2RPN2022


class AgentTester(BaseAgent):
    def act(self, observation, reward, done):
        if observation.current_step == 0:
            return self.action_space()
        if observation.current_step >= 13:
            return self.action_space()
        return self.action_space({"set_storage": [(0, 1.), (1, -1.)]})
    
    
class WCCI2022Tester(unittest.TestCase):
    """tests are focused on the storage units for this class"""
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

class TestL2RPNWCCI2022ScoreFun(unittest.TestCase): 
    """test curtailment and redispatching scores to make sure they match the description in the html of the competition.
    
    (storage are tested in the class just above, so i don't retest them here)
    """
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_wcci_2022", test=True)
        self.env.seed(0)
        self.env.reset()
        self.score_fun = L2RPNWCCI2022ScoreFun()
        self.score_fun.initialize(self.env)   
        
        # run the env without actions
        self.env.set_id(0)
        obs = self.env.reset()
        dn_ = self.env.action_space()
        self.obs_ref, reward, done, info = self.env.step(dn_)
        obs = self.obs_ref
        self.score_ref = self.score_fun(env=self.env, action=dn_, is_done=False, has_error=False, is_illegal=False, is_ambiguous=False)
        self.losses_ref = np.sum(obs.gen_p) - np.sum(obs.load_p)
        self.pt_ref = obs.gen_cost_per_MW[obs.gen_p > 0].max()
        
        # test that the score, in this case, is the losses
        assert np.abs(self.score_ref - self.losses_ref * self.pt_ref / 12.) <= 1e-4
             
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_unary_curtailment(self):
        gen_id = 0
        # now apply a curtailment action and compute the score
        for E_curt in [0.5, 1., 2., 3.]:
            self.env.set_id(0)
            obs = self.env.reset()
            # curtail a certain amount of MWh and check the formula is correct
            act = self.env.action_space({"curtail": [(gen_id,(self.obs_ref.gen_p[gen_id] - 12. * E_curt) / obs.gen_pmax[gen_id])]})
            obs, reward, done, info = self.env.step(act)
            assert not info['exception']
            score = self.score_fun(env=self.env, action=act, is_done=False, has_error=False, is_illegal=False, is_ambiguous=False)
            losses = np.sum(obs.gen_p) - np.sum(obs.load_p)
            pt = obs.gen_cost_per_MW[obs.gen_p > 0].max()
            assert np.abs(pt - self.pt_ref) <= 1e-5, f"wrong marginal price for {E_curt:.2f}"
            assert np.abs(score - (losses * pt / 12. + 2 * E_curt * pt)) <= 1e-4, f"error for {E_curt:.2f}"
        
    def test_unary_redisp(self):
        gen_id = 8  # ramps is 11.2
        # now apply a redispatching action and compute the score
        for E_redisp in [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]:
            self.env.set_id(0)
            obs = self.env.reset()
            # curtail 12MW for 5 mins => 1 MWh (next step gen 0 should produce 41.6 MW)
            act = self.env.action_space({"redispatch": [(gen_id, 12. * E_redisp)]})
            obs, reward, done, info = self.env.step(act)
            assert not info['exception'], f'one error occured for {E_redisp:.2f}'
            score = self.score_fun(env=self.env, action=act, is_done=False, has_error=False, is_illegal=False, is_ambiguous=False)
            losses = np.sum(obs.gen_p) - np.sum(obs.load_p)
            pt = obs.gen_cost_per_MW[obs.gen_p > 0].max()
            assert np.abs(pt - self.pt_ref) <= 1e-5, f"wrong marginal price for {E_redisp:.2f}"
            assert np.abs(score - (losses * pt / 12. + 2 * np.abs(E_redisp) * pt)) <= 2e-4, f"error for {E_redisp:.2f}"
        
        
if __name__ == "__main__":
    unittest.main()        
