# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import warnings
import unittest
import grid2op
from grid2op.Exceptions import Grid2OpException
import numpy as np


class Issue340Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_wcci_2022", test=True, _add_to_name=type(self).__name__)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_obs(self):
        obs = self.env.reset()
        obs.state_of(storage_id=0)
        
    def test_act_add(self):
        act0 = self.env.action_space({"curtail": [(0, 0.5)]})
        act1 = self.env.action_space({"curtail": [(1, 0.5)]})
        res = act0 + act1
        assert np.all(res.curtail[[0,1]] == 0.5)
  

if __name__ == "__main__":
    unittest.main()
    