# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import numpy as np
import unittest
import itertools
import warnings


class TestIsolatedLoad(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True) # , backend=LightSimBackend())
        param = self.env.parameters
        param.ENV_DC = True  # force the computation of the powerflow in DC mode
        param.MAX_LINE_STATUS_CHANGED = 99999
        param.MAX_SUB_CHANGED = 99999
        param.NO_OVERFLOW_DISCONNECTION = True
        self.env.change_parameters(param)
        self.env.change_forecast_parameters(param)
        time_series_id = 0
        self.env.set_id(time_series_id)
        self.obs = self.env.reset()
        self.obs_start, *_  = self.env.step(self.env.action_space({}))
        self.tol = 3e-5
        return super().setUp()
    
    def test_specific_action_load(self):
        act = self.env.action_space({'set_bus': {'loads_id': [(0, 1), (1, 2)],
                                                 'generators_id': [(0, 2), (1, 1)],
                                                 'lines_or_id': [(2, 2), (3, 1), (4, 2), (5, -1)],
                                                 'lines_ex_id': [(0, 1), (2, 1), (5, -1)]}}
                                    )
        obs, reward, done, info = self.env.step(act)
        assert done
        assert info["exception"]
        
    def test_specific_action_gen(self):
        act = self.env.action_space({'set_bus': {'loads_id': [(0, 1), (1, 2)],
                                                 'generators_id': [(0, 2), (1, 1)],
                                                 'lines_or_id': [(2, 2), (3, 1), (4, 2), (5, -1)],
                                                 'lines_ex_id': [(0, 1), (2, 1), (5, -1)]}}
                                    )
        obs, reward, done, info = self.env.step(act)
        assert done
        assert info["exception"]
    
if __name__ == "__main__":
    unittest.main()
