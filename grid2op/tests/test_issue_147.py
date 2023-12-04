# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
from grid2op.Parameters import Parameters
import warnings
import pdb


class Issue147Tester(unittest.TestCase):
    def test_issue_147(self):
        """
        The rule "Prevent Reconnection" was not properly applied, this was because the
        observation of the _ObsEnv was not properly updated.
        """

        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        param.NB_TIMESTEP_COOLDOWN_SUB = 3
        param.NB_TIMESTEP_COOLDOWN_LINE = 3
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_realistic", test=True, param=param, _add_to_name=type(self).__name__)

        action = env.action_space(
            {"set_bus": {"substations_id": [(1, [2, 2, 1, 1, 2, 2])]}}
        )

        obs, reward, done, info = env.step(
            env.action_space({"set_line_status": [(0, -1)]})
        )
        env.step(env.action_space())
        sim_o, sim_r, sim_d, info = obs.simulate(env.action_space())
        env.step(env.action_space())
        sim_o, sim_r, sim_d, info = obs.simulate(env.action_space())
        env.step(env.action_space())
        sim_o, sim_r, sim_d, info = obs.simulate(env.action_space())
        obs, reward, done, info = env.step(
            env.action_space({"set_line_status": [(0, 1)]})
        )
        assert obs.time_before_cooldown_line[0] == 3
        sim_o, sim_r, sim_d, sim_info = obs.simulate(action)
        assert not sim_d
        assert not sim_info[
            "is_illegal"
        ]  # this was declared as "illegal" due to an issue with updating
        # the line status in the observation of the _ObsEnv
        obs, reward, done, info = obs.simulate(action)
        assert not info["is_illegal"]


if __name__ == "__main__":
    unittest.main()
