# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import unittest
import warnings

import numpy as np

from grid2op import make

from grid2op.tests.helper_path_test import PATH_DATA_TEST_PP, PATH_DATA_TEST
from grid2op.Backend import PandaPowerBackend

from grid2op.tests.helper_path_test import HelperTests
from grid2op.tests.BaseBackendTest import BaseTestNames
from grid2op.tests.BaseBackendTest import BaseTestLoadingCase
from grid2op.tests.BaseBackendTest import BaseTestLoadingBackendFunc
from grid2op.tests.BaseBackendTest import BaseTestTopoAction
from grid2op.tests.BaseBackendTest import BaseTestEnvPerformsCorrectCascadingFailures
from grid2op.tests.BaseBackendTest import BaseTestChangeBusAffectRightBus
from grid2op.tests.BaseBackendTest import BaseTestShuntAction
from grid2op.tests.BaseBackendTest import BaseTestResetEqualsLoadGrid
from grid2op.tests.BaseBackendTest import BaseTestVoltageOWhenDisco
from grid2op.tests.BaseBackendTest import BaseTestChangeBusSlack
from grid2op.tests.BaseBackendTest import BaseIssuesTest
from grid2op.tests.BaseBackendTest import BaseStatusActions
PATH_DATA_TEST_INIT = PATH_DATA_TEST
PATH_DATA_TEST = PATH_DATA_TEST_PP

import warnings
warnings.simplefilter("error")


class TestNames(HelperTests, BaseTestNames):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)

    def get_path(self):
        return PATH_DATA_TEST_INIT


class TestLoadingCase(HelperTests, BaseTestLoadingCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"


class TestLoadingBackendFunc(HelperTests, BaseTestLoadingBackendFunc):
    def setUp(self):
        # TODO find something more elegant
        BaseTestLoadingBackendFunc.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestLoadingBackendFunc.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"


class TestTopoAction(HelperTests, BaseTestTopoAction):
    def setUp(self):
        BaseTestTopoAction.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestTopoAction.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"


class TestEnvPerformsCorrectCascadingFailures(HelperTests, BaseTestEnvPerformsCorrectCascadingFailures):
    def setUp(self):
        BaseTestEnvPerformsCorrectCascadingFailures.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestEnvPerformsCorrectCascadingFailures.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)

    def get_casefile(self):
        return "test_case14.json"

    def get_path(self):
        return PATH_DATA_TEST


class TestChangeBusAffectRightBus(HelperTests, BaseTestChangeBusAffectRightBus):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)


class TestShuntAction(HelperTests, BaseTestShuntAction):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)


class TestResetEqualsLoadGrid(HelperTests, BaseTestResetEqualsLoadGrid):
    def setUp(self):
        BaseTestResetEqualsLoadGrid.setUp(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)


class TestVoltageOWhenDisco(HelperTests, BaseTestVoltageOWhenDisco):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)


class TestChangeBusSlack(HelperTests, BaseTestChangeBusSlack):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)


class TestIssuesTest(HelperTests, BaseIssuesTest):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)


class TestStatusAction(HelperTests, BaseStatusActions):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)


# Specific to pandapower power
class TestChangeBusAffectRightBus2(unittest.TestCase):
    def skip_if_needed(self):
        pass

    def make_backend(self):
        return PandaPowerBackend()

    def test_set_bus(self):
        self.skip_if_needed()
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make(test=True, backend=backend)
        env.reset()
        # action = env.action_space({"change_bus": {"lines_or_id": [17]}})
        action = env.action_space({"set_bus": {"lines_or_id": [(17, 2)]}})
        obs, reward, done, info = env.step(action)
        assert np.all(np.isfinite(obs.v_or))
        assert np.sum(env.backend._grid["bus"]["in_service"]) == 15

    def test_change_bus(self):
        self.skip_if_needed()
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make(test=True, backend=backend)
        env.reset()
        action = env.action_space({"change_bus": {"lines_or_id": [17]}})
        obs, reward, done, info = env.step(action)
        assert np.all(np.isfinite(obs.v_or))
        assert np.sum(env.backend._grid["bus"]["in_service"]) == 15

    def test_change_bustwice(self):
        self.skip_if_needed()
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make(test=True, backend=backend)
        env.reset()
        action = env.action_space({"change_bus": {"lines_or_id": [17]}})
        obs, reward, done, info = env.step(action)
        assert not done
        assert np.all(np.isfinite(obs.v_or))
        assert np.sum(env.backend._grid["bus"]["in_service"]) == 15
        assert env.backend._grid["trafo"]["hv_bus"][2] == 18

        action = env.action_space({"change_bus": {"lines_or_id": [17]}})
        obs, reward, done, info = env.step(action)
        assert not done
        assert np.all(np.isfinite(obs.v_or))
        assert np.sum(env.backend._grid["bus"]["in_service"]) == 14
        assert env.backend._grid["trafo"]["hv_bus"][2] == 4


if __name__ == "__main__":
    unittest.main()
