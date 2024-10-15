# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import numpy as np
import grid2op
import unittest
import warnings

import re

from grid2op.Parameters import Parameters


class Issue367Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Creation of the environment
            param = Parameters()
            param.NB_TIMESTEP_COOLDOWN_SUB = 3
            param.NB_TIMESTEP_COOLDOWN_LINE = 3
            param.NO_OVERFLOW_DISCONNECTION = True
            self.env = grid2op.make('l2rpn_wcci_2022', param=param, _add_to_name=type(self).__name__)
            
        self.env.set_id(0) 
        self.env.seed(0)
        self.obs = self.env.reset()

    def test_cooldown(self):
        thermal_limits = self.env.get_thermal_limit()
        
        # Get the simulator
        sim = self.obs.get_simulator()

        # Manually compute rho using thermal limits
        rho_old = sim.current_obs.a_or / thermal_limits
        assert np.max(np.abs(sim.current_obs.rho.max() - rho_old.max())) <= 1e-5

        # Simulate 1 do_nothing action, with loads/gens/etc. staying the same
        do_nothing = self.env.action_space()
        new_sim = sim.predict(do_nothing)

        # Manually compute new rho, since  the rho from the simulator is wrong.
        rho_new = new_sim.current_obs.a_or / thermal_limits
        assert np.max(np.abs(new_sim.current_obs.rho.max() - rho_new.max())) <= 1e-5


    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    

if __name__ == "__main__":
    unittest.main()
