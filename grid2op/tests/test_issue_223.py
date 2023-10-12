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
from grid2op.Chronics import ChangeNothing
from grid2op.tests.helper_path_test import *

try:
    from grid2op.PlotGrid import PlotMatplot

    CAN_PLOT = True
except ImportError as exc_:
    CAN_PLOT = False


class Issue223Tester(unittest.TestCase):
    def _skip_if_not_installed(self):
        if not CAN_PLOT:
            self.skipTest("matplotlib is not installed")
            
    def reset_without_pp_futurewarnings(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            obs = self.env.reset()
        return obs
            
    def setUp(self) -> None:
        if CAN_PLOT:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env_nm = os.path.join(PATH_DATA_TEST, "5bus_modif_grid")
                self.env = grid2op.make(env_nm, test=True, chronics_class=ChangeNothing)
                self.env.seed(0)
                self.reset_without_pp_futurewarnings()

    def test_env_working(self):
        self._skip_if_not_installed()
        with warnings.catch_warnings():
            warnings.filterwarnings("error")  # there should be no warning there
            # in the issue, it crashes there
            plot_helper = PlotMatplot(self.env.observation_space)
            assert "sub_5" in plot_helper._grid_layout
            assert "sub_6" in plot_helper._grid_layout
            # now test i can plot an observation
            fig = plot_helper.plot_obs(self.reset_without_pp_futurewarnings())
