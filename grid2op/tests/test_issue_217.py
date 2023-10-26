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


class Issue217Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_nm = os.path.join(PATH_DATA_TEST, "5bus_modif_grid")
            self.env = grid2op.make(env_nm, test=True, chronics_class=ChangeNothing, _add_to_name=type(self).__name__)
            self.env.seed(0)
            self.env.reset()

    def test_env_working(self):
        assert self.env.n_sub == 7
        assert np.all(self.env.sub_info[[5, 6]] == 0)
