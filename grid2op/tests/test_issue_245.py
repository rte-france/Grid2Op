# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import unittest

import grid2op
from grid2op.tests.helper_path_test import *
from grid2op.Action import CompleteAction


class Issue245Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_nm = "l2rpn_case14_sandbox"
            self.env = grid2op.make(env_nm, test=True, action_class=CompleteAction)
            self.env.seed(0)
            self.env.reset()

    def test_simulate(self):
        # change the thermal limit to disconnect powerline 3, and only this one, after 3 steps
        self.env.set_thermal_limit(
            2.0
            * np.array(
                [
                    162.49806,
                    127.40141,
                    107.29365,
                    110.0 / 2.0,
                    107.90302,
                    48.58299,
                    135.80658,
                    434.52097,
                    252.74222,
                    649.2642,
                    56.013855,
                    183.57967,
                    324.4969,
                    79.42569,
                    319.6616,
                    100.69238,
                    46.787445,
                    73.96375,
                    895.7249,
                    563.3386,
                ],
                dtype=dt_float,
            )
        )
        # perform the 3 steps
        obs = self.env.reset()
        assert obs.line_status[3]
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.line_status[3]
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.line_status[3]
        obs, *_ = self.env.step(self.env.action_space())
        assert not obs.line_status[3]
        # line 3 is now disconnected, i try to simulate an action that reconnects it (it should not work) !
        act = self.env.action_space({"set_bus": {"lines_or_id": [(3, 1)]}})
        *_, sim_info = obs.simulate(act)
        assert (
            len(sim_info["exception"]) > 0
        ), "there should be an exception because the action is illegal"
        *_, info = self.env.step(act)
        assert (
            len(info["exception"]) > 0
        ), "there should be an exception because the action is illegal"
