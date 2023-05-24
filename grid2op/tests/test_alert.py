# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import numpy as np
import unittest
import os
import tempfile
from grid2op.tests.helper_path_test import *

from grid2op.operator_attention import LinearAttentionBudgetByLine
from grid2op import make
from grid2op.Reward import RedispReward, _AlarmScore
from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner
from grid2op.Environment import Environment
from grid2op.Episode import EpisodeData
from grid2op.Opponent import GeometricOpponent, GeometricOpponentMultiArea

LINES_ATTACKED = [
            "62_58_180",
            "62_63_160",
            "48_50_136",
            "48_53_141",
            "41_48_131",
            "39_41_121",
            "43_44_125",
            "44_45_126",
            "34_35_110",
            "54_58_154",
        ] 

class TestAlert(unittest.TestCase):
    """test the basic bahavior of the assistant alert feature"""

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make(self.env_nm, test=True, difficulty="1")
        self.env.seed(0)
        self.env.reset()

    def test_init_default_param(self) -> None : 
        
        assert isinstance(self.env.parameters.ALERT_TIME_WINDOW, np.int32)
        assert isinstance(self.env._attention_budget, LinearAttentionBudgetByLine)
        assert self.env._opponent_class == GeometricOpponent
        assert self.env.parameters.ALERT_TIME_WINDOW > 0

        param = self.env.parameters
        param.init_from_dict({
            "ALERT_TIME_WINDOW": -1, 
            "ASSISTANT_WARNING_TYPE": "BY_LINE"})
        
        negative_value_invalid = False
        try: 
            self.env.change_parameters(param)
            self.env.reset()
        except : 
            negative_value_invalid = True 

        assert negative_value_invalid


    def test_init_observation(self) -> None :
        true_alertable_lines = LINES_ATTACKED
        
        assert isinstance(self.env.alertable_line_names, list)
        assert sorted(self.env.alertable_line_names) == sorted(true_alertable_lines)
        assert self.env.dim_alerts == len(true_alertable_lines)


    def test_raise_alert_action(self) -> None : 
        # raise alert on line number 0
        act = self.env.action_space()
        act.raise_alert = [0]


        obs, reward, done, info = self.env.step(act)

    def test_assistant_reward_value(self) -> None : 
        
        # 
        score = 2
        # SI on en prédit 2/3 on aurait - 2
        assert score == -2.0

        # SI on en prédit 1/3 on aurait - 6
        assert score == -6

        # SI on en prédit 1/3 on aurait - 10 
        assert score == -10


@unittest.skip("Not implemented yet")
class TestAlertMultiArea(unittest.TestCase): 
    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make(self.env_nm, test=True, difficulty="1")

        self.opponent = GeometricOpponentMultiArea(self.env.action_space)
        self.opponent.init(self.env,
                           lines_attacked=[LINES_ATTACKED[:3],LINES_ATTACKED[3:]],
                           attack_every_xxx_hour=24,
                           average_attack_duration_hour=4,
                           minimum_attack_duration_hour=2,
                           pmax_pmin_ratio=4)
        
        self.env.seed(0)
        self.env.reset()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()