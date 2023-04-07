# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import tempfile
import warnings
import grid2op
from grid2op.Opponent.OpponentSpace import OpponentSpace
from grid2op.tests.helper_path_test import *
from grid2op.Chronics import ChangeNothing
from grid2op.Opponent import (
    GeometricOpponentMultiArea
)
from grid2op.Action import TopologyAction
from grid2op.MakeEnv import make
from grid2op.Opponent.BaseActionBudget import BaseActionBudget
from grid2op.dtypes import dt_int
from grid2op.Parameters import Parameters
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData
from grid2op.Environment import SingleEnvMultiProcess
from grid2op.Exceptions import OpponentError
import pdb

LINES_ATTACKED = ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]


class TestMultiAreaOpponent(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("l2rpn_case14_sandbox", test=True)
        self.opponent = GeometricOpponentMultiArea(self.env.action_space)
        self.opponent.init(self.env,
                           lines_attacked=[LINES_ATTACKED[:3],LINES_ATTACKED[3:]],
                           attack_every_xxx_hour=24,
                           average_attack_duration_hour=4,
                           minimum_attack_duration_hour=2,
                           pmax_pmin_ratio=4)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_seed(self):
        self.opponent.seed(0)
        obs = self.env.reset()
        initial_budget = 250
        self.opponent.reset(initial_budget)
        assert np.all(self.opponent.list_opponents[0]._attack_times == [160])
        assert np.all(self.opponent.list_opponents[1]._attack_times == [182, 467])
        
if __name__ == "__main__":
    unittest.main()