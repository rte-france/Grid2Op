# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

import grid2op
import unittest
from grid2op.Parameters import Parameters
import warnings


class Issue153Tester(unittest.TestCase):
    def test_issue_153(self):
        """
        The rule "Prevent Reconnection" was not properly applied, this was because the
        observation of the _ObsEnv was not properly updated.
        """

        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        param.NB_TIMESTEP_COOLDOWN_SUB = 3
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "rte_case14_realistic", test=True, _add_to_name=type(self).__name__+"test_issue_153"
            )
        env.gen_max_ramp_up[:] = env.gen_pmax
        env.gen_max_ramp_down[:] = env.gen_pmax
        env.action_space.gen_max_ramp_up[:] = env.gen_pmax
        env.action_space.gen_max_ramp_down[:] = env.gen_pmax
        env.action_space.actionClass.gen_max_ramp_up[:] = env.gen_pmax
        env.action_space.actionClass.gen_max_ramp_down[:] = env.gen_pmax
        obs = env.reset()
        # prod 1 do [74.8, 77. , 75.1, 76.4, 76.3, 75. , 74.5, 74.2, 73. , 72.6]
        for i in range(3):
            obs, reward, done, info = env.step(env.action_space())

        # now generator 1 decreases: 76.3, 75. , 74.5, 74.2, 73. , 72.6
        action = env.action_space({"redispatch": [(0, -76)]})
        obs, reward, done, info = env.step(action)
        # should be at 0.3
        assert np.abs(obs.prod_p[0] - 0.3) <= 1e-2, "wrong data"

        # I do an illegal action
        obs, reward, done, info = env.step(action)
        # and the redispatching was negative (this was the issue)
        assert obs.prod_p[0] >= -env._tol_poly, "generator should be positive"


if __name__ == "__main__":
    unittest.main()
