# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import grid2op
import warnings
from grid2op.Action import CompleteAction
from grid2op.gym_compat import GymEnv


class Issue281Tester(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage", test=True, action_class=CompleteAction,
                _add_to_name=type(self).__name__
            )

    def tearDown(self):
        self.env.close()

    def test_can_make(self):
        """test that the opponent state is correctly copied"""
        gym_env = GymEnv(self.env)


if __name__ == "__main__":
    unittest.main()
