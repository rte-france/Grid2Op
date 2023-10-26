# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import warnings
import unittest


class BaseTestPlot(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)

    def tearDown(self):
        self.env.close()
    
    def test_render(self):
        obs = self.env.reset()
        arr_ = self.env.render()
        assert arr_.shape == (720, 1280, 3)
        assert arr_.min() == 0
        assert arr_.max() == 255

if __name__ == "__main__":
    unittest.main()