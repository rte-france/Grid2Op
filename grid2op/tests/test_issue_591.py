# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import warnings
import unittest
import grid2op
from grid2op.utils import ScoreL2RPN2020
from grid2op.Agent import DoNothingAgent


class Issue591Tester(unittest.TestCase):
    def setUp(self) -> None:
        self.max_iter = 10
        return super().setUp()
    
    def test_issue_591(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            
        ch_patterns = env.chronics_handler.reset()
        ch_patterns = ch_patterns.tolist()
        ch_patterns = ch_patterns[17:19]

        nb_scenario = len(ch_patterns)
        agent = DoNothingAgent(env.action_space)
        handler = env.chronics_handler
        handler.set_filter(lambda path: path in ch_patterns)
        chronics = handler.reset()


        scorer_2020 = ScoreL2RPN2020(
            env,
            max_step=1,
            nb_scenario=1,
            env_seeds=[0 for _ in range(1)],
            agent_seeds=[0 for _ in range(1)],
        )
        scorer_2020.clear_all()
        scorer_2020 = ScoreL2RPN2020(
            env,
            max_step=self.max_iter,
            nb_scenario=nb_scenario,
            env_seeds=[0 for _ in range(nb_scenario)],
            agent_seeds=[0 for _ in range(nb_scenario)],
        )
        try:
            score_2020 = scorer_2020.get(agent)
        finally:
            scorer_2020.clear_all()
        for scen_path, score, ts_survived, total_ts in zip(ch_patterns, *score_2020):
            assert total_ts == self.max_iter, f"wrong number of ts {total_ts} vs {self.max_iter}"

            
if __name__ == "__main__":
    unittest.main()
