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
from grid2op.utils import ScoreL2RPN2023
from grid2op.Agent.doNothing import DoNothingAgent, BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction
from grid2op.Chronics import FromHandlers
from grid2op.Chronics.handlers import CSVHandler, PerfectForecastHandler
from _aux_opponent_for_test_alerts import TestOpponent

class Alert_Blackout_Agent(BaseAgent):
    def __init__(self, action_space,do_Alerts=False,blackout_step=None):
        super().__init__(action_space)
        self.do_Alerts = do_Alerts
        self.blackout_step = blackout_step

    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        act=self.action_space({})

        if self.do_Alerts:
            act+=self.action_space({"raise_alert": [i for i in range(len(observation.alertable_line_ids))]})#we don't know which line will get attacked, so we raise all alerts to be sure to raise an alert for the line attacked

        if((self.blackout_step is not None) and (observation.current_step == self.blackout_step)):
            blackout_action = self.action_space({})
            blackout_action.gen_set_bus = [(0, -1)]
            act+=blackout_action

        return act


def get_blackout(self, env):
    blackout_action = env.action_space({})
    blackout_action.gen_set_bus = [(0, -1)]
    return blackout_action

class TestScoreL2RPN2023Assist(unittest.TestCase):
    """test the "assistant" part of the l2rpn_idf_2023"""
    def setUp(self) -> None:
        env_name = "l2rpn_idf_2023"
        ATTACKED_LINE = "48_50_136"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE],
                                   duration=3,
                                   steps_attack=[2])
            opponent_class = TestOpponent,
            #kwargs_opponent = kwargs_opponent,

            self.env = grid2op.make(env_name,
                                    test=True,
                                    #Error currently at doing make with kwargs below
                                    #data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                    #                      "gen_p_handler": CSVHandler("prod_p"),
                                    #                      "load_p_handler": CSVHandler("load_p"),
                                    #                      "gen_v_handler": CSVHandler("prod_v"),
                                    #                      "load_q_handler": CSVHandler("load_q"),
                                    #                      "h_forecast": (5,),
                                    #                      "gen_p_for_handler": PerfectForecastHandler("prod_p_forecasted", quiet_warnings=True),
                                    #                      "gen_v_for_handler": PerfectForecastHandler("prod_v_forecasted", quiet_warnings=True),
                                    #                      "load_p_for_handler": PerfectForecastHandler("load_p_forecasted", quiet_warnings=True),
                                    #                      "load_q_for_handler": PerfectForecastHandler("load_q_forecasted", quiet_warnings=True),
                                    #                    },
                                    )
            self.env.set_max_iter(30)
            params = self.env.parameters
            params.NO_OVERFLOW_DISCONNECTION = True
            params.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        self.seed = 0
        self.scen_id = 0
        self.nb_scenario = 2
        self.max_iter = 30 #if attacks are at timestep 13, needs at least 26 timsteps to get beyond the alert time window and see some relevant score in case of no blackout
        
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    

    def test_score_helper(self):
        """basic tests for ScoreL2RPN2023 class for assistant score with nres score set to 0"
        With those seeds, we observe an attack in both episodes at timestep 13"""
        self.env.reset() 
        my_score = ScoreL2RPN2023(
                    self.env,
                    nb_scenario=self.nb_scenario,
                    env_seeds=[0 for _ in range(self.nb_scenario)],
                    agent_seeds=[0 for _ in range(self.nb_scenario)],
                    max_step=self.max_iter,
                    weight_op_score=0.6,
                    weight_assistant_score=0.4,
                    weight_nres_score=0.,
                    scale_nres_score=100,
                    scale_assistant_score=100,
                    min_nres_score=-100.,
                    min_assistant_score=-300)
        try:

            # test do nothing indeed gets 100.
            res_dn = my_score.get(DoNothingAgent(self.env.action_space))
            for scen_id, (ep_score, op_score, nres_score, assistant_score) in enumerate(res_dn[0]):
                assert assistant_score == 100. #no blackout with no disconnections
                assert ep_score == 0.6 * op_score + 0.4 * assistant_score

            #raising alerts for attack but it should not as it gets no blackout. With the score L2RPN IDF parametrization, it gives a score of -300
            res_agent = my_score.get(Alert_Blackout_Agent(self.env.action_space, do_Alerts=True))#attacks are at timestep 13 for both scenarios with those seeds
            for scen_id, (ep_score, op_score, nres_score, assistant_score) in enumerate(res_agent[0]):
                assert(assistant_score == -300)
                assert ep_score == 0.6 * op_score + 0.4 * assistant_score

            #raising no alert for attack and it gets no blackout. With the score L2RPN IDF parametrization, it gives a score of 100
            res_agent = my_score.get(Alert_Blackout_Agent(self.env.action_space, do_Alerts=False))#attacks are at timestep 13 for both scenarios with those seeds
            for scen_id, (ep_score, op_score, nres_score, assistant_score) in enumerate(res_agent[0]):
                assert(assistant_score == 100)
                assert ep_score == 0.6 * op_score + 0.4 * assistant_score

            #raising alert for attack and it gets a blackout. With the score L2RPN IDF parametrization, it gives a score of 100
            res_agent = my_score.get(Alert_Blackout_Agent(self.env.action_space, do_Alerts=True,blackout_step=15))#attacks are at timestep 13 for both scenarios with those seeds
            for scen_id, (ep_score, op_score, nres_score, assistant_score) in enumerate(res_agent[0]):
                assert(assistant_score == 100)
                assert ep_score == 0.6 * op_score + 0.4 * assistant_score


        finally:
            my_score.clear_all()

    def test_min_score(self):
        """test the score does not go bellow the minimum in input
        With those seeds, we observe an attack in both episodes at timestep 13"""

        try:
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
                scale_assistant_score=100,
                min_nres_score=-100.,
                min_assistant_score=-300)

            res_agent = my_score.get(Alert_Blackout_Agent(self.env.action_space, blackout_step=14))
            for scen_id, (ep_score, op_score, nres_score, assistant_score) in enumerate(res_agent[0]):
                assert(assistant_score == -300)#gets minimum score because blackout after attack with no alerts
        finally:
            my_score.clear_all()

 
if __name__ == "__main__":
    unittest.main()   