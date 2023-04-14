# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import time
import warnings

from grid2op.tests.helper_path_test import *

from grid2op.Exceptions import *
from grid2op.Environment import Environment
from grid2op.MakeEnv import make
from grid2op.dtypes import dt_float


class TestTimedOutEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # TODO : Comment on fait avec un time out ?
            self.env1 = make("rte_case14_test", test=True, time_out_ms=1e3)
            

    def tearDown(self) -> None:
        self.env1.close()

    def test_action_b4_time_out(self):
        obs = self.env1.reset()
        obs, reward, done, info = self.env1.step(self.env1.action_space())
        # TODO : comment on récupère le time_step
        assert self.obs.current_step==1

    def test_action_after_1_time_out(self):
        obs = self.env1.reset()
        time.sleep(1.2 * self.env1.time_out_ms/1e3)                                 # sleep in seconds
        obs, reward, done, info = self.env1.step(self.env1.action_space())
        # TODO : comment on récupère le time_step
        assert self.obs.current_step==2

    def test_action_after_3_time_out(self):
        obs = self.env1.reset()
        time.sleep(3.2 * self.env1.time_out_ms/1e3)                                 # sleep in seconds
        obs, reward, done, info = self.env1.step(self.env1.action_space())
        # TODO : comment on récupère le time_step
        assert self.obs.current_step==4


if __name__ == "__main__":
    unittest.main()
