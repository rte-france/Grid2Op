# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import unittest
from grid2op.Converter import BackendConverter

import grid2op
from grid2op.tests.helper_path_test import *
from grid2op import make
from grid2op.tests.helper_path_test import PATH_DATA_TEST_PP, PATH_DATA_TEST
from grid2op.Backend import PandaPowerBackend

from grid2op.tests.helper_path_test import HelperTests
from grid2op.tests.BaseBackendTest import (BaseTestNames,
                                           BaseTestLoadingCase,
                                           BaseTestLoadingBackendFunc,
                                           BaseTestTopoAction,
                                           BaseTestEnvPerformsCorrectCascadingFailures,
                                           BaseTestChangeBusAffectRightBus,
                                           BaseTestShuntAction,
                                           BaseTestResetEqualsLoadGrid,
                                           BaseTestVoltageOWhenDisco,
                                           BaseTestChangeBusSlack,
                                           BaseIssuesTest,
                                           BaseStatusActions)

PATH_DATA_TEST_INIT = PATH_DATA_TEST
PATH_DATA_TEST = PATH_DATA_TEST_PP

BKclass1 = PandaPowerBackend
BKclass2 = PandaPowerBackend


class TestLoading(HelperTests, unittest.TestCase):
    def test_init(self):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("rte_case14_realistic", test=True, backend=backend)


class TestNames(BaseTestNames, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend

    def get_path(self):
        return PATH_DATA_TEST_INIT


class TestLoadingCase(BaseTestLoadingCase, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"


class TestLoadingBackendFunc(BaseTestLoadingBackendFunc, unittest.TestCase):
    def setUp(self):
        # TODO find something more elegant
        BaseTestLoadingBackendFunc.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestLoadingBackendFunc.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"


class TestTopoAction(BaseTestTopoAction, unittest.TestCase):
    def setUp(self):
        BaseTestTopoAction.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestTopoAction.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"


class TestEnvPerformsCorrectCascadingFailures(
    BaseTestEnvPerformsCorrectCascadingFailures, unittest.TestCase
):
    def setUp(self):
        BaseTestEnvPerformsCorrectCascadingFailures.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestEnvPerformsCorrectCascadingFailures.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend

    def get_casefile(self):
        return "test_case14.json"

    def get_path(self):
        return PATH_DATA_TEST


class TestChangeBusAffectRightBus(BaseTestChangeBusAffectRightBus, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend


class TestShuntAction(BaseTestShuntAction, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend


class TestResetEqualsLoadGrid(BaseTestResetEqualsLoadGrid, unittest.TestCase):
    def setUp(self):
        BaseTestResetEqualsLoadGrid.setUp(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend


class TestVoltageOWhenDisco(BaseTestVoltageOWhenDisco, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend


class TestChangeBusSlack(BaseTestChangeBusSlack, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend


class TestIssuesTest(BaseIssuesTest, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend


class TestStatusAction(BaseStatusActions, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend


if __name__ == "__main__":
    unittest.main()
