# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This script provides a way to run the tests performed by grid2Op for the backend.

These tests are not 100% complete (some things might not be tested and are tested somewhere else)
but they cover a big part of what the backend is expected to do.

YOU NEED TO INSTALL GRID2OP FROM THE GITHUB REPO FOR THIS TO WORK !
To do that, simply:

1) clone grid2op repo
2) cd there 
3) run `pip install -e .`

(do this in a venv preferably)
"""

import unittest
import warnings

# first the backend class (for the example here)
from Step5_modify_topology import CustomBackend_Minimal

# then some required things
from grid2op.tests.helper_path_test import PATH_DATA_TEST_PP, PATH_DATA_TEST
from grid2op.tests.helper_path_test import HelperTests
PATH_DATA_TEST_INIT = PATH_DATA_TEST
PATH_DATA_TEST = PATH_DATA_TEST_PP

# then all the tests that can be automatically performed
from grid2op.tests.BaseBackendTest import BaseTestNames, BaseTestLoadingCase, BaseTestLoadingBackendFunc
from grid2op.tests.BaseBackendTest import BaseTestTopoAction, BaseTestEnvPerformsCorrectCascadingFailures
from grid2op.tests.BaseBackendTest import BaseTestChangeBusAffectRightBus, BaseTestShuntAction
from grid2op.tests.BaseBackendTest import BaseTestResetEqualsLoadGrid, BaseTestVoltageOWhenDisco, BaseTestChangeBusSlack
from grid2op.tests.BaseBackendTest import BaseIssuesTest, BaseStatusActions
from grid2op.tests.test_Environment import (TestLoadingBackendPandaPower as BaseTestLoadingBackendPandaPower, 
                                            TestResetOk as BaseTestResetOk)
from grid2op.tests.test_Environment import (TestResetAfterCascadingFailure as TestResetAfterCascadingFailure,
                                            TestCascadingFailure as BaseTestCascadingFailure)
from grid2op.tests.BaseRedispTest import BaseTestRedispatch, BaseTestRedispatchChangeNothingEnvironment
from grid2op.tests.BaseRedispTest import BaseTestRedispTooLowHigh, BaseTestDispatchRampingIllegalETC
from grid2op.tests.BaseRedispTest import BaseTestLoadingAcceptAlmostZeroSumRedisp

# then still some glue code, mainly for the names of the time series
from grid2op.Converter import BackendConverter
from grid2op.Backend import PandaPowerBackend

# our backend does not read the names from the grid, so this test is not relevant
# class TestNames(HelperTests, BaseTestNames):
#     def make_backend(self, detailed_infos_for_cascading_failures=False):
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore")
#             bk = BackendConverter(source_backend_class=CustomBackend_Minimal,
#                                   target_backend_class=PandaPowerBackend,
#                                   use_target_backend_name=True,
#                                   detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
#         return bk

#     def get_path(self):
#         return PATH_DATA_TEST_INIT

class TestLoadingCase(HelperTests, BaseTestLoadingCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = BackendConverter(source_backend_class=CustomBackend_Minimal,
                                  target_backend_class=PandaPowerBackend,
                                  use_target_backend_name=True,
                                  detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"
        
        
if __name__ == "__main__":
    unittest.main()
