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
from abc import ABC, abstractmethod
import inspect
from grid2op.dtypes import dt_float
from grid2op.Backend import Backend

test_dir = Path(__file__).parent.absolute()
grid2op_dir = os.fspath(test_dir.parent.absolute())
data_test_dir = os.path.abspath(os.path.join(grid2op_dir, "data_test"))
data_dir = os.path.abspath(os.path.join(grid2op_dir, "data"))

# sys.path.insert(0, grid2op_dir)  # cause https://github.com/rte-france/Grid2Op/issues/577
# because the addition of `from grid2op._create_test_suite import create_test_suite`
# in grid2op "__init__.py"


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
        if hasattr(type(super()), "setUp"):
            # needed for backward compatibility
            super().setUp()
            
    def tearDown(self):
        # needed for backward compatibility
        pass
    
    def compare_vect(self, pred, true):
        res = dt_float(np.max(np.abs(pred - true))) <= self.tolvect
        res = res and dt_float(np.mean(np.abs(pred - true))) <= self.tolvect
        return res


class MakeBackend(ABC, HelperTests):
    @abstractmethod
    def make_backend(self, detailed_infos_for_cascading_failures=False) -> Backend:
        pass

    def make_backend_with_glue_code(self, detailed_infos_for_cascading_failures=False, extra_name="", n_busbar=2) -> Backend:
        Backend._clear_class_attribute()
        bk = self.make_backend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        type(bk)._clear_grid_dependant_class_attributes()
        type(bk).set_env_name(type(self).__name__ + extra_name)
        type(bk).set_n_busbar_per_sub(n_busbar)
        return bk
    
    def get_path(self) -> str:
        raise NotImplementedError(
            "This function should be implemented for the test suit you are developping"
        )

    def get_casefile(self) -> str:
        raise NotImplementedError(
            "This function should be implemented for the test suit you are developping"
        )

    def skip_if_needed(self) -> None:
        if hasattr(self, "tests_skipped"):
            nm_ = inspect.currentframe().f_back.f_code.co_name
            if nm_ in self.tests_skipped:
                self.skipTest('the test "{}" is skipped: it has been added to self.tests_skipped'.format(nm_))
                