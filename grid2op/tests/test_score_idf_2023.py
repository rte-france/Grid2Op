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
from grid2op.Action import ActionSpace, BaseAction
from grid2op.utils import ScoreL2RPN2023
from grid2op.Observation import BaseObservation
from grid2op.Agent.doNothing import DoNothingAgent, BaseAgent
from grid2op.Chronics import FromHandlers
from grid2op.Chronics.handlers import CSVHandler, PerfectForecastHandler
from grid2op.Reward import _NewRenewableSourcesUsageScore


class CurtailTrackerAgent(BaseAgent):
    def __init__(self, action_space, gen_renewable, gen_pmax, curtail_level=1.):
        super().__init__(action_space)
        self.gen_renewable = gen_renewable
        self.gen_pmax = gen_pmax[gen_renewable]
        self.curtail_level = curtail_level
        
    def act(self, obs: BaseObservation, reward, done):
        curtail_target = self.curtail_level * obs.gen_p[self.gen_renewable] / self.gen_pmax
        act = self.action_space(
            {"curtail": [(el, ratio) for el, ratio in zip(np.arange(len(self.gen_renewable))[self.gen_renewable], curtail_target)]}
        )
        return act

class CurtailAgent(BaseAgent):
    def __init__(self, action_space: ActionSpace, curtail_level=1.):
        self.curtail_level = curtail_level
        super().__init__(action_space)
        
    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        next_gen_p = observation.simulate(self.action_space())[0].gen_p_before_curtail
        curtail = self.curtail_level * next_gen_p / observation.gen_pmax
        curtail[~observation.gen_renewable] = -1
        act = self.action_space({"curtail": curtail})
        return act
    
class TestScoreL2RPN2023(unittest.TestCase):
    
    def setUp(self) -> None:
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(env_name,
                                    test=True,
                                    data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                                          "gen_p_handler": CSVHandler("prod_p"),
                                                          "load_p_handler": CSVHandler("load_p"),
                                                          "gen_v_handler": CSVHandler("prod_v"),
                                                          "load_q_handler": CSVHandler("load_q"),
                                                          "h_forecast": (1, ),
                                                          "gen_p_for_handler": PerfectForecastHandler("prod_p_forecasted"),
                                                          "gen_v_for_handler": PerfectForecastHandler("prod_v_forecasted"),
                                                          "load_p_for_handler": PerfectForecastHandler("load_p_forecasted"),
                                                          "load_q_for_handler": PerfectForecastHandler("load_q_forecasted"),
                                                        },) 
            self.env.set_max_iter(20)
            params = self.env.parameters
            params.NO_OVERFLOW_DISCONNECTION = True
            params.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        self.seed = 0
        self.scen_id = 0
        self.nb_scenario = 2
        self.max_iter = 10
        
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_score_helper(self):
        """basic tests for ScoreL2RPN2023 class"""
        self.env.reset() 
        my_score = ScoreL2RPN2023(
            self.env,
            nb_scenario=self.nb_scenario,
            env_seeds=[0 for _ in range(self.nb_scenario)],
            agent_seeds=[0 for _ in range(self.nb_scenario)],
            max_step=self.max_iter,
            weight_op_score=0.8,
            weight_assistant_score=0,
            weight_nres_score=0.2,
            scale_nres_score=100,
            scale_assistant_score=100)
        
        # test do nothing indeed gets 100.
        res_dn = my_score.get(DoNothingAgent(self.env.action_space))
        for scen_id, (ep_score, op_score, nres_score, assistant_confidence_score, assistant_cost_score) in enumerate(res_dn[0]):
            assert nres_score == 100.
            assert ep_score == 0.8 * op_score + 0.2 * nres_score
            
        # now test that the score decrease fast "at beginning" and slower "at the end"
        # ie from 1. to 0.95 bigger difference than from 0.8 to 0.7
        res_agent0 = my_score.get(CurtailTrackerAgent(self.env.action_space,
                                                     gen_renewable = self.env.gen_renewable,
                                                     gen_pmax=self.env.gen_pmax,
                                                     curtail_level = 0.95))
        # assert np.allclose(res_agent0[0][0][2], 81.83611011377577)
        # assert np.allclose(res_agent0[0][1][2], 68.10026022372575)
        assert np.allclose(res_agent0[0][0][0], 0.8 * res_agent0[0][0][1] + 0.2 * res_agent0[0][0][2])
        assert np.allclose(res_agent0[0][0][2], 16.73128726588182)
        assert np.allclose(res_agent0[0][1][2], -26.02070223995034)
        
        res_agent1 = my_score.get(CurtailTrackerAgent(self.env.action_space,
                                                     gen_renewable = self.env.gen_renewable,
                                                     gen_pmax=self.env.gen_pmax,
                                                     curtail_level = 0.9))
        # assert np.allclose(res_agent1[0][0][2], 56.256863965501466)
        # assert np.allclose(res_agent1[0][1][2], 43.370607328810415)
        assert np.allclose(res_agent1[0][0][2], -49.61104170080321)
        assert np.allclose(res_agent1[0][1][2], -78.00216266500183)
        
        # decrease
        assert 100. - res_agent0[0][0][2] >= res_agent0[0][0][2] - res_agent1[0][0][2]
        assert 100. - res_agent0[0][1][2] >= res_agent0[0][1][2] - res_agent1[0][1][2]
            
        res_agent2 = my_score.get(CurtailTrackerAgent(self.env.action_space,
                                                     gen_renewable = self.env.gen_renewable,
                                                     gen_pmax=self.env.gen_pmax,
                                                     curtail_level = 0.8))
        
        assert np.allclose(res_agent2[0][0][2], -100)
        assert np.allclose(res_agent2[0][1][2], -100)
        # decrease
        assert 100. - res_agent1[0][0][2] >= res_agent1[0][0][2] - res_agent2[0][0][2]
        assert 100. - res_agent1[0][1][2] >= res_agent1[0][1][2] - res_agent2[0][1][2]
        
        res_agent3 = my_score.get(CurtailTrackerAgent(self.env.action_space,
                                                     gen_renewable = self.env.gen_renewable,
                                                     gen_pmax=self.env.gen_pmax,
                                                     curtail_level = 0.7))
        assert np.allclose(res_agent3[0][0][2], -100)
        assert np.allclose(res_agent3[0][1][2], -100)
        assert res_agent1[0][0][2] - res_agent2[0][0][2] >= res_agent2[0][0][2] - res_agent2[0][0][2]
        assert res_agent1[0][1][2] - res_agent2[0][1][2] >= res_agent2[0][1][2] - res_agent2[0][1][2]
        my_score.clear_all() 

    def test_spec(self):
        """ spec are: 100pts for 0 curtailment, 0 pts for 80% renewable (20% curtailment) and -100 pts for 50% renewable"""
        
        # test function without actual data
        assert _NewRenewableSourcesUsageScore._surlinear_func_curtailment(100.) == 1.
        assert _NewRenewableSourcesUsageScore._surlinear_func_curtailment(80.) == 0.
        assert _NewRenewableSourcesUsageScore._surlinear_func_curtailment(50.) == -1.
        assert _NewRenewableSourcesUsageScore._surlinear_func_curtailment(0.) < _NewRenewableSourcesUsageScore._surlinear_func_curtailment(50.)
        
        # now test with "real" data
        my_score = ScoreL2RPN2023(
            self.env,
            nb_scenario=self.nb_scenario,
            env_seeds=[0 for _ in range(self.nb_scenario)],
            agent_seeds=[0 for _ in range(self.nb_scenario)],
            max_step=self.max_iter,
            weight_op_score=0.8,
            weight_assistant_score=0,
            weight_nres_score=0.2)
        
        # test do nothing indeed gets 100.
        res_dn = my_score.get(DoNothingAgent(self.env.action_space))
        for scen_id, (ep_score, op_score, nres_score, assistant_confidence_score, assistant_cost_score) in enumerate(res_dn[0]):
            assert nres_score
            
        # test 80% gets indeed close to 0
        res_80 = my_score.get(CurtailAgent(self.env.action_space, 0.8))
        for scen_id, (ep_score, op_score, nres_score, assistant_confidence_score, assistant_cost_score) in enumerate(res_80[0]):
            assert abs(nres_score) <= 7
            
        # test 50% gets indeed close to -100
        res_50 = my_score.get(CurtailAgent(self.env.action_space, 0.5))
        for scen_id, (ep_score, op_score, nres_score, assistant_confidence_score, assistant_cost_score) in enumerate(res_50[0]):
            assert abs(nres_score + 100.) <= 7
        
        # test bellow 50% still gets close to -100
        res_30 = my_score.get(CurtailAgent(self.env.action_space, 0.3))
        for scen_id, (ep_score, op_score, nres_score, assistant_confidence_score, assistant_cost_score) in enumerate(res_30[0]):
            assert abs(nres_score + 100.) <= 7
        my_score.clear_all()

        
if __name__ == "__main__":
    unittest.main()   