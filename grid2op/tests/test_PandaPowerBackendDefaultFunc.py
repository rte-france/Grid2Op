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

from grid2op.tests.helper_path_test import PATH_DATA_TEST_PP, PATH_DATA_TEST
from grid2op.Backend import PandaPowerBackend, Backend

from grid2op.dtypes import dt_int, dt_bool
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


class PandaPowerBackendDefault(PandaPowerBackend):
    """
    test the class for pandapower, if the default implementation of the backend are used, instead of the
    more optimized pandapower implementation.
    """

    def __init__(self, detailed_infos_for_cascading_failures=False):
        PandaPowerBackend.__init__(
            self,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )

    def copy(self):
        return Backend.copy(self)

    def get_line_status(self):
        return Backend.get_line_status(self)

    def get_line_flow(self):
        return Backend.get_line_flow(self)

    def _disconnect_line(self, l_id):
        return Backend._disconnect_line(self, l_id)

    def get_topo_vect(self):
        """
        otherwise there are some infinite recursions
        """
        res = np.full(self.dim_topo, fill_value=np.NaN, dtype=dt_int)

        line_status = np.concatenate(
            (
                self._grid.line["in_service"].values,
                self._grid.trafo["in_service"].values,
            )
        ).astype(dt_bool)

        i = 0
        for row in self._grid.line[["from_bus", "to_bus"]].values:
            bus_or_id = row[0]
            bus_ex_id = row[1]
            if line_status[i]:
                res[self.line_or_pos_topo_vect[i]] = (
                    1 if bus_or_id == self.line_or_to_subid[i] else 2
                )
                res[self.line_ex_pos_topo_vect[i]] = (
                    1 if bus_ex_id == self.line_ex_to_subid[i] else 2
                )
            else:
                res[self.line_or_pos_topo_vect[i]] = -1
                res[self.line_ex_pos_topo_vect[i]] = -1
            i += 1

        nb = self._grid.line.shape[0]
        i = 0
        for row in self._grid.trafo[["hv_bus", "lv_bus"]].values:
            bus_or_id = row[0]
            bus_ex_id = row[1]

            j = i + nb
            if line_status[j]:
                res[self.line_or_pos_topo_vect[j]] = (
                    1 if bus_or_id == self.line_or_to_subid[j] else 2
                )
                res[self.line_ex_pos_topo_vect[j]] = (
                    1 if bus_ex_id == self.line_ex_to_subid[j] else 2
                )
            else:
                res[self.line_or_pos_topo_vect[j]] = -1
                res[self.line_ex_pos_topo_vect[j]] = -1
            i += 1

        i = 0
        for bus_id in self._grid.gen["bus"].values:
            res[self.gen_pos_topo_vect[i]] = 1 if bus_id == self.gen_to_subid[i] else 2
            i += 1

        i = 0
        for bus_id in self._grid.load["bus"].values:
            res[self.load_pos_topo_vect[i]] = (
                1 if bus_id == self.load_to_subid[i] else 2
            )
            i += 1

        # do not forget storage units !
        i = 0
        for bus_id in self._grid.storage["bus"].values:
            res[self.storage_pos_topo_vect[i]] = (
                1 if bus_id == self.storage_to_subid[i] else 2
            )
            i += 1
        return res


class TestNames(HelperTests, BaseTestNames, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackendDefault(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_path(self):
        return PATH_DATA_TEST_INIT


class TestLoadingCase(HelperTests, BaseTestLoadingCase, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackendDefault(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"


class TestLoadingBackendFunc(HelperTests, BaseTestLoadingBackendFunc, unittest.TestCase):
    def setUp(self):
        # TODO find something more elegant
        BaseTestLoadingBackendFunc.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestLoadingBackendFunc.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackendDefault(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"


class TestTopoAction(HelperTests, BaseTestTopoAction, unittest.TestCase):
    def setUp(self):
        BaseTestTopoAction.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestTopoAction.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackendDefault(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"


class TestEnvPerformsCorrectCascadingFailures(
    HelperTests, BaseTestEnvPerformsCorrectCascadingFailures, unittest.TestCase
):
    def setUp(self):
        BaseTestEnvPerformsCorrectCascadingFailures.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestEnvPerformsCorrectCascadingFailures.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackendDefault(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_casefile(self):
        return "test_case14.json"

    def get_path(self):
        return PATH_DATA_TEST


class TestChangeBusAffectRightBus(HelperTests, BaseTestChangeBusAffectRightBus, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackendDefault(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestShuntAction(HelperTests, BaseTestShuntAction, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackendDefault(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestResetEqualsLoadGrid(HelperTests, BaseTestResetEqualsLoadGrid, unittest.TestCase):
    def setUp(self):
        BaseTestResetEqualsLoadGrid.setUp(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackendDefault(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestVoltageOWhenDisco(HelperTests, BaseTestVoltageOWhenDisco, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackendDefault(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestChangeBusSlack(HelperTests, BaseTestChangeBusSlack, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackendDefault(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestIssuesTest(HelperTests, BaseIssuesTest, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackendDefault(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestStatusAction(HelperTests, BaseStatusActions, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackendDefault(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


# Specific to pandapower power
class TestChangeBusAffectRightBus2(unittest.TestCase):
    def skip_if_needed(self):
        pass

    def make_backend(self):
        return PandaPowerBackendDefault()

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
