# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import unittest
import warnings
import unittest

import numpy as np

from grid2op import make

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
                                           BaseStatusActions,
                                           BaseTestStorageAction)


class TestNames(BaseTestNames, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestLoadingCase(BaseTestLoadingCase, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestLoadingBackendFunc(BaseTestLoadingBackendFunc, unittest.TestCase):
    def setUp(self):
        # TODO find something more elegant
        BaseTestLoadingBackendFunc.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestLoadingBackendFunc.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestTopoAction(BaseTestTopoAction, unittest.TestCase):
    def setUp(self):
        BaseTestTopoAction.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestTopoAction.tearDown(self)


class TestEnvPerformsCorrectCascadingFailures(
    BaseTestEnvPerformsCorrectCascadingFailures, unittest.TestCase
):
    def setUp(self):
        BaseTestEnvPerformsCorrectCascadingFailures.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestEnvPerformsCorrectCascadingFailures.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestChangeBusAffectRightBus(BaseTestChangeBusAffectRightBus, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestShuntAction(BaseTestShuntAction, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestResetEqualsLoadGrid(BaseTestResetEqualsLoadGrid, unittest.TestCase):
    def setUp(self):
        BaseTestResetEqualsLoadGrid.setUp(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestVoltageOWhenDisco(BaseTestVoltageOWhenDisco, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestChangeBusSlack(BaseTestChangeBusSlack, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestIssuesTest(BaseIssuesTest, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestStatusAction(BaseStatusActions, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestStorageAction(BaseTestStorageAction, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


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
            env = make('rte_case14_realistic', test=True, backend=backend)
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
            env = make('rte_case14_realistic', test=True, backend=backend)
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
            env = make('rte_case14_realistic', test=True, backend=backend)
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
