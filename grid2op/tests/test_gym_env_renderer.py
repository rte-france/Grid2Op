# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/rte-france/Grid2Op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings
    
import grid2op
from grid2op.gym_compat import GymEnv
import numpy as np


class TestGymEnvRenderer(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)

    def test_render_signature(self):
        genv = GymEnv(self.env)
        _ = genv.reset()
        array = genv.render()
        assert array.shape == (720, 1280, 3)
        with self.assertRaises(TypeError):
            array = genv.render(render_mode="rgb_array")
        
        
if __name__ == "__main__":
    unittest.main()