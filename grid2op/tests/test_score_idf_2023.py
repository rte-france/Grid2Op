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

class TestScoreL2RPN2023(unittest.TestCase):
    
    def setUp(self) -> None:
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(env_name,
                                    test=True
                                ) 
            self.env.set_max_iter(20)
            self.env.parameters.NO_OVERFLOW_DISCONNECTION = True
        self.seed = 0
        self.scen_id = 0
        self.nb_scenario = 2
        self.max_iter = 10
        
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_score_helper(self):
        """basic tests for ScoreL2RPN2022 class"""
        self.env.reset() 
        my_score = ScoreL2RPN2023(
            self.env,
            nb_scenario=self.nb_scenario,
            env_seeds=[0 for _ in range(self.nb_scenario)],
            agent_seeds=[0 for _ in range(self.nb_scenario)],
            max_step=self.max_iter,
            weight_op_score=0.8, weight_assistant_score=0, weight_nres_score=0.2,)
        
        res_dn = my_score.get(DoNothingAgent(self.env.action_space))
        res_agent = my_score.get(CurtailTrackerAgent(self.env.action_space, gen_renewable = self.env.gen_renewable,
                                           gen_pmax=self.env.gen_pmax,
                                           curtail_level = 0.8))
        for scen_id, (score_dn, score_agent) in enumerate(zip(res_dn[0], res_agent[0])):
            assert score_agent < score_dn, f"error for scenario id {scen_id}"
            
        my_score.clear_all() 

if __name__ == "__main__":
    unittest.main()   