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
from grid2op.Agent.doNothing import DoNothingAgent

class TestScoreL2RPN2023(unittest.TestCase):

    def setUp(self) -> None:
        env_name = "l2rpn_idf_2023"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(env_name,
                                    test=True,
                                    _add_to_name=type(self).__name__)

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
            weight_op_score=0.6,
            weight_assistant_score=0.25,
            weight_nres_score=0.15,
            scale_nres_score=100,
            scale_assistant_score=100,
            min_nres_score=-100.,
            min_assistant_score=-300)
        try:
            # test do nothing indeed gets 100.
            res_dn = my_score.get(DoNothingAgent(self.env.action_space))
            for scen_id, (ep_score, op_score, nres_score, assistant_score) in enumerate(res_dn[0]):
                assert nres_score == 100.
                assert ep_score == 0.6 * op_score + 0.15 * nres_score + 0.25 * assistant_score
                assert assistant_score == 100.  # no blackout with no disconnections
                assert op_score == 0


        finally:
            my_score.clear_all()



if __name__ == "__main__":
    unittest.main()   