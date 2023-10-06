# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

# making sure test can be ran from:
# root package directory
# Grid2Op subdirectory
# Grid2Op/tests subdirectory
import sys
import os
import numpy as np
from pathlib import Path

from grid2op.dtypes import dt_float

test_dir = Path(__file__).parent.absolute()
grid2op_dir = os.fspath(test_dir.parent.absolute())
data_test_dir = os.path.abspath(os.path.join(grid2op_dir, "data_test"))
data_dir = os.path.abspath(os.path.join(grid2op_dir, "data"))

sys.path.insert(0, grid2op_dir)

PATH_DATA = data_dir
PATH_DATA_TEST = data_test_dir
PATH_CHRONICS = data_test_dir
PATH_CHRONICS_Make2 = os.path.abspath(os.path.join(grid2op_dir, "data"))
PATH_DATA_TEST_PP = os.path.abspath(os.path.join(PATH_DATA_TEST, "test_PandaPower"))
EXAMPLE_CHRONICSPATH = os.path.abspath(
    os.path.join(data_test_dir, "5bus_example", "chronics")
)
EXAMPLE_CASEFILE = os.path.abspath(
    os.path.join(data_test_dir, "5bus_example", "5bus_example.json")
)
PATH_DATA_MULTIMIX = os.path.abspath(os.path.join(data_test_dir, "multimix"))


class HelperTests:
    def setUp(self):
        self.tolvect = dt_float(1e-2)
        self.tol_one = dt_float(1e-5)
        super().setUp()

    def compare_vect(self, pred, true):
        res = dt_float(np.max(np.abs(pred - true))) <= self.tolvect
        res = res and dt_float(np.mean(np.abs(pred - true))) <= self.tolvect
        return res
