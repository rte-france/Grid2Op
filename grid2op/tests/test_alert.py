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

from grid2op.operator_attention import LinearAttentionBudget
from grid2op import make
from grid2op.Reward import RedispReward, _AlarmScore
from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner
from grid2op.Environment import Environment
from grid2op.Episode import EpisodeData

class TestAlert(unittest.TestCase):
    """test the basic bahavior of the assistant alert feature"""

    def test_init_default_param(self) -> None : 
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make(self.env_nm, test=True)

        
        self.env.seed(0)
        self.env.reset()
        assert isinstance(self.env.parameters.ALERT_TIME_WINDOW, np.int32)
        assert self.env.parameters.ALERT_TIME_WINDOW > 0

        param = self.env.parameters
        param.init_from_dict({"ALERT_TIME_WINDOW": -1})
        
        negative_value_invalid = False
        try: 
            self.env.change_parameters(param)
            self.env.reset()
        except : 
            negative_value_invalid = True 

        assert negative_value_invalid



    
    

