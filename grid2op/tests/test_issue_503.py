# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.Parameters import Parameters
import warnings
import unittest


class Issue503Tester(unittest.TestCase):    
    def test_only_kwargs(self):
        params = Parameters()
        params.NO_OVERFLOW_DISCONNECTION = True
        with self.assertRaises(TypeError):
            _ = grid2op.make("l2rpn_case14_sandbox", params, test=True, _add_to_name=type(self).__name__)
            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True, param=params, _add_to_name=type(self).__name__)
        env.close()


if __name__ == '__main__':
    unittest.main()
