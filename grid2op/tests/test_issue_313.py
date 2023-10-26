# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend
from grid2op.Action import PlayableAction
import warnings
import numpy as np


class Issu313Tester(unittest.TestCase):
    def setUp(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                backend=LightSimBackend(),
                                action_class=PlayableAction,
                                param=param,
                                test=True,
                                _add_to_name=type(self).__name__)
        self.env.set_id(2)
        self.env.seed(0)
        self.env.reset()
        self.env.fast_forward_chronics(215)

    def tearDown(self):
        self.env.close()

    def test_gen_margin_up(self):
        """test that the runner works ok"""

        act_prev = self.env.action_space()
        act_prev.storage_p = [4.9, 9.9]
        obs, reward, done, info = self.env.step(act_prev)
        obs, reward, done, info = self.env.step(act_prev)
        obs, reward, done, info = self.env.step(act_prev)
        obs, reward, done, info = self.env.step(act_prev)

        assert np.all(obs.gen_margin_up >= 0.)
        # this is because the last generator (slack bus) has a pmax of 100. and a value of 100.05 ...


if __name__ == "__main__":
    unittest.main()
