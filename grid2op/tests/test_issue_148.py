# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
from grid2op.tests.helper_path_test import PATH_CHRONICS

import grid2op
import unittest
from grid2op.Parameters import Parameters
import warnings
import pdb


class Issue148Tester(unittest.TestCase):
    def test_issue_148(self):
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
                os.path.join(PATH_CHRONICS, "env_14_test_maintenance"),
                test=True,
                param=param,
                _add_to_name=type(self).__name__
            )

        ID_MAINT = 11  # in maintenance at the second time step
        obs = env.reset()
        # check i can "simulate" properly if a maintenance happens next
        sim_o, sim_r, sim_d, sim_i = obs.simulate(env.action_space())
        assert not sim_d
        assert (
            sim_o.time_next_maintenance[ID_MAINT] == 0
        )  # the stuff have been properly updated
        assert not sim_o.line_status[ID_MAINT]
        oo_, rr_, dd_, ii_ = env.step(env.action_space())
        assert not dd_
        assert oo_.time_next_maintenance[ID_MAINT] == 0
        assert not oo_.line_status[ID_MAINT]

        # check once the maintenance is performed, it stays this way
        sim_o, sim_r, sim_d, sim_i = oo_.simulate(env.action_space())
        assert not sim_d
        assert (
            sim_o.time_next_maintenance[ID_MAINT] == 0
        )  # the stuff have been properly updated
        assert not sim_o.line_status[ID_MAINT]
        oo_, rr_, dd_, ii_ = env.step(env.action_space())
        assert not dd_
        assert oo_.time_next_maintenance[ID_MAINT] == 0
        assert not oo_.line_status[ID_MAINT]

        # now test the cooldown
        action = env.action_space(
            {"set_bus": {"substations_id": [(1, [1, 1, 1, 1, 1, 1])]}}
        )
        oo_, rr_, dd_, ii_ = env.step(action)
        assert oo_.time_before_cooldown_sub[1] == 3
        oo_, rr_, dd_, ii_ = env.step(env.action_space())
        oo_, rr_, dd_, ii_ = env.step(action)
        assert oo_.time_before_cooldown_sub[1] == 1
        assert ii_["is_illegal"]

        # still illegal (cooldown of 1 now so 0 later)
        ooo_, rr_, dd_, ii_ = oo_.simulate(action)
        assert not dd_
        assert ooo_.time_before_cooldown_sub[ID_MAINT] == 0
        assert ii_["is_illegal"]
        # we check that's illegal (cooldown is 1 now so 0 later)
        ooo_, rr_, dd_, ii_ = env.step(action)
        assert not dd_
        assert ooo_.time_before_cooldown_sub[ID_MAINT] == 0
        assert ii_["is_illegal"]

        # we check that's legal next step
        oooo_, rr_, dd_, ii_ = env.step(action)
        assert not dd_
        assert oooo_.time_before_cooldown_sub[ID_MAINT] == 0
        assert not ii_["is_illegal"]
        
        # but now it's legal to simulate ("next step" - see just above- it's legal)
        oooo_, rr_, dd_, ii_ = ooo_.simulate(action)
        assert not dd_
        assert oooo_.time_before_cooldown_sub[ID_MAINT] == 0
        assert not ii_["is_illegal"]
        
   
if __name__ == "__main__":
    unittest.main()
