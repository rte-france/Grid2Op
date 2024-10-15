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


class Issue331Tester(unittest.TestCase):
    def test_seed(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
            
        with self.assertRaises(Grid2OpException):
            env.seed(2735729614)  # crashes !
            
        env.seed(2147483647)  # just the limit
        
        with self.assertRaises(Grid2OpException):
            env.seed(2147483648)  # crashes !


if __name__ == "__main__":
    unittest.main()
    