# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/rte-france/Grid2Op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import grid2op
import unittest
import warnings

import re

from grid2op.Chronics import MultifolderWithCache
from grid2op.tests.helper_path_test import *


class Issue367Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Creation of the environment
            self.env = grid2op.make("l2rpn_wcci_2022_dev",
                                    chronics_class=MultifolderWithCache,
                                    _add_to_name=type(self).__name__,
                                    test=True) # No bug wihout the MultifolderWithCache argument here
            self.env.chronics_handler.real_data.set_filter(lambda x: re.match(".*2050-02-14_0", x) is not None) # We test on a randomly chosen chronic for the example
            self.env.chronics_handler.reset()
            
        self.env.set_id(0) 
        self.env.seed(0)
        param = self.env.parameters
        param.NO_OVERFLOW_DISCONNECTION = True
        self.env.change_parameters(param)
        self.obs = self.env.reset()

    def test_date(self):
        assert self.obs.year == 2050
        assert self.obs.month == 2
        assert self.obs.day == 14

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    

if __name__ == "__main__":
    unittest.main()
