# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest

from grid2op.tests.helper_path_test import PATH_DATA_TEST_PP, PATH_DATA_TEST

import grid2op
from grid2op.Backend import PandaPowerBackend
from grid2op.tests.BaseRedispTest import (
    BaseTestRedispatch,
    BaseTestRedispatchChangeNothingEnvironment,
)
from grid2op.tests.BaseRedispTest import (
    BaseTestRedispTooLowHigh,
    BaseTestDispatchRampingIllegalETC,
)
from grid2op.tests.BaseRedispTest import BaseTestLoadingAcceptAlmostZeroSumRedisp

PATH_DATA_TEST_INIT = PATH_DATA_TEST
PATH_DATA_TEST = PATH_DATA_TEST_PP


class TestRedispatch(BaseTestRedispatch, unittest.TestCase):
    def setUp(self):
        # TODO find something more elegant
        BaseTestRedispatch.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestRedispatch.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestRedispatchChangeNothingEnvironment(
    BaseTestRedispatchChangeNothingEnvironment, unittest.TestCase
):
    def setUp(self):
        # TODO find something more elegant
        BaseTestRedispatchChangeNothingEnvironment.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestRedispatchChangeNothingEnvironment.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestRedispTooLowHigh(BaseTestRedispTooLowHigh, unittest.TestCase):
    def setUp(self):
        # TODO find something more elegant
        BaseTestRedispTooLowHigh.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestRedispTooLowHigh.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestDispatchRampingIllegalETC(BaseTestDispatchRampingIllegalETC, unittest.TestCase):
    def setUp(self):
        # TODO find something more elegant
        BaseTestDispatchRampingIllegalETC.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestDispatchRampingIllegalETC.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


class TestLoadingAcceptAlmostZeroSumRedisp(
    BaseTestLoadingAcceptAlmostZeroSumRedisp, unittest.TestCase
):
    def setUp(self):
        # TODO find something more elegant
        BaseTestLoadingAcceptAlmostZeroSumRedisp.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestLoadingAcceptAlmostZeroSumRedisp.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )


if __name__ == "__main__":
    unittest.main()
