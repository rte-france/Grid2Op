# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import unittest

import grid2op
from grid2op.tests.helper_path_test import *
from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import WeightedRandomOpponent, BaseActionBudget


class Issue224Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_nm = "l2rpn_case14_sandbox"
            lines_attacked = ["6_7_18"]
            rho_normalization = [1.0]
            opponent_attack_cooldown = 2  # need to be at least 1
            opponent_attack_duration = 12
            opponent_budget_per_ts = 9999.0
            opponent_init_budget = 9999.0
            self.env = grid2op.make(
                env_nm,
                test=True,
                _add_to_name=type(self).__name__,
                opponent_attack_cooldown=opponent_attack_cooldown,
                opponent_attack_duration=opponent_attack_duration,
                opponent_budget_per_ts=opponent_budget_per_ts,
                opponent_init_budget=opponent_init_budget,
                opponent_action_class=PowerlineSetAction,
                opponent_class=WeightedRandomOpponent,
                opponent_budget_class=BaseActionBudget,
                kwargs_opponent={
                    "lines_attacked": lines_attacked,
                    "rho_normalization": rho_normalization,
                    "attack_period": opponent_attack_cooldown,
                },
            )
            self.env.seed(0)
            self.env.reset()

    def test_opponent(self):
        obs = self.env.reset()
        assert obs.line_status[18], "line 18 should be connected"
        # TODO this test does not really test it...
        # TODO but i would need an env with a connected powerline with exactly 0 flow on it !
        # assert obs.rho[18] == 0., "line 18 should not have any flow"

        res = self.env._opponent.attack(
            obs, self.env.action_space(), self.env.action_space(), 100, False
        )
        assert len(res) == 2, "it should return something of length 2"
