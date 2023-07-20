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
from grid2op.utils import ScoreL2RPN2023, ScoreL2RPN2022, ScoreICAPS2021, ScoreL2RPN2020
from grid2op.Observation import BaseObservation
from grid2op.Agent import DoNothingAgent, BaseAgent
from grid2op.Chronics import FromHandlers
from grid2op.Chronics.handlers import CSVHandler, PerfectForecastHandler, DoNothingHandler


class _TesterSimulateAgent(BaseAgent):
    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        observation.simulate(self.action_space())
        observation.simulate(self.action_space())
        return super().act(observation, reward, done)


class TestHighResSimCountInScore:
    def _score_fun(self):
        raise RuntimeError()
    
    def _env_name(self):
        return "l2rpn_case14_sandbox"
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self._env_name(),
                                    test=True,
                                    data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                                          "gen_p_handler": CSVHandler("prod_p"),
                                                          "load_p_handler": CSVHandler("load_p"),
                                                          "gen_v_handler": DoNothingHandler("gen_v"),
                                                          "load_q_handler": CSVHandler("load_q"),
                                                          "h_forecast": (5,),
                                                          "gen_p_for_handler": PerfectForecastHandler("prod_p_forecasted", quiet_warnings=True),
                                                        #   "gen_v_for_handler": PerfectForecastHandler("prod_v_forecasted", quiet_warnings=True),
                                                          "load_p_for_handler": PerfectForecastHandler("load_p_forecasted", quiet_warnings=True),
                                                          "load_q_for_handler": PerfectForecastHandler("load_q_forecasted", quiet_warnings=True),
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
        my_score = self._score_fun()(
            self.env,
            nb_scenario=self.nb_scenario,
            env_seeds=[0 for _ in range(self.nb_scenario)],
            agent_seeds=[0 for _ in range(self.nb_scenario)],
            max_step=self.max_iter,
            add_nb_highres_sim=True)
        try:
            # test do nothing indeed gets 0
            res_dn = my_score.get(DoNothingAgent(self.env.action_space))
            assert len(res_dn) == 4
            all_scores, ts_survived, total_ts, nb_highres_sim = res_dn
            assert nb_highres_sim == [0] * self.nb_scenario, f"do nothing does not have 0 but {nb_highres_sim}"
            
            # test do nothing indeed gets 2 x .
            res_tester = my_score.get(_TesterSimulateAgent(self.env.action_space))
            assert len(res_tester) == 4
            all_scores, ts_survived, total_ts, nb_highres_sim = res_tester
            assert nb_highres_sim == [2 * self.max_iter] * self.nb_scenario, f"_TesterSimulateAgent does not have 2x but {nb_highres_sim}"
            
        finally:
            my_score.clear_all() 
    
class TestHighResSimCountInScore2023(TestHighResSimCountInScore, unittest.TestCase):
    def _score_fun(self):
        return ScoreL2RPN2023
    
    def _env_name(self):
        return "l2rpn_idf_2023"
    
class TestHighResSimCountInScore2022(TestHighResSimCountInScore, unittest.TestCase):
    def _score_fun(self):
        return ScoreL2RPN2022
    
    def _env_name(self):
        return "l2rpn_case14_sandbox"
    
class TestHighResSimCountInScore2021(TestHighResSimCountInScore, unittest.TestCase):
    def _score_fun(self):
        return ScoreICAPS2021
    
    def _env_name(self):
        return "l2rpn_icaps_2021"
    
class TestHighResSimCountInScore2020(TestHighResSimCountInScore, unittest.TestCase):
    def _score_fun(self):
        return ScoreL2RPN2020
    
    def _env_name(self):
        return "l2rpn_case14_sandbox"
    
if __name__ == "__main__":
    unittest.main()   