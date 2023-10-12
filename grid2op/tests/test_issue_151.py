# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import numpy as np
import os
from grid2op.tests.helper_path_test import PATH_CHRONICS

import grid2op
import unittest
from grid2op.Parameters import Parameters
import warnings
import pdb


class Issue151Tester(unittest.TestCase):
    def test_issue_151(self):
        """
        The rule "Prevent Reconnection" was not properly applied, this was because the
        observation of the _ObsEnv was not properly updated.
        """

        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        param.NB_TIMESTEP_COOLDOWN_SUB = 3
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_realistic", test=True, _add_to_name=type(self).__name__)
        do_nothing = env.action_space({})
        obs, reward, done, info = env.step(do_nothing)
        obs.line_status = (
            obs.line_status / 1
        )  # do some weird things to the vector "line_status"

        # the next line of cod
        _, _, _, _ = env.step(do_nothing)


if __name__ == "__main__":
    unittest.main()
