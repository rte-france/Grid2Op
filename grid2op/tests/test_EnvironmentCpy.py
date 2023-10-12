# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


# This module will test that the environment, when copied, works as expected (ie with making some basic tests
# for the results of "env.copy()"

import unittest
import warnings

from grid2op.tests.helper_path_test import *

from grid2op.Reward import L2RPNReward
from grid2op.tests.test_Environment import (
    TestLoadingBackendPandaPower as Aux_TestLoadingBackendPandaPower,
    TestIllegalAmbiguous as Aux_TestIllegalAmbiguous,
    TestOtherReward as Aux_TestOtherReward,
    TestResetOk as Aux_TestResetOk,
    TestLineChangeLastBus as Aux_TestLineChangeLastBus,
    TestResetAfterCascadingFailure as Aux_TestResetAfterCascadingFailure,
    TestCascadingFailure as Aux_TestCascadingFailure,
    TestLoading2envDontCrash as Aux_TestLoading2envDontCrash,
    TestDeactivateForecast as Aux_TestDeactivateForecast,
    TestMaxIter as Aux_TestMaxIter,
)
from grid2op.tests.test_Agent import TestAgent as Aux_TestAgent

DEBUG = False
PROFILE_CODE = False
if PROFILE_CODE:
    import cProfile


class TestLoadingBackendPandaPowerCopy(Aux_TestLoadingBackendPandaPower):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestIllegalAmbiguousCopy(Aux_TestIllegalAmbiguous):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestOtherRewardCopy(Aux_TestOtherReward):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestResetOkCopy(Aux_TestResetOk):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()

    def test_reset_after_blackout_withdetailed_info(self, env=None):
        backend = self.make_backend(detailed_infos_for_cascading_failures=True)
        if env is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env = grid2op.make(
                    "rte_case5_example",
                    test=True,
                    reward_class=L2RPNReward,
                    other_rewards={"test": L2RPNReward},
                    backend=backend,
                    _add_to_name=type(self).__name__,
                )
        super().test_reset_after_blackout_withdetailed_info(env=env.copy())


class TestLineChangeLastBusCopy(Aux_TestLineChangeLastBus):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestResetAfterCascadingFailureCopy(Aux_TestResetAfterCascadingFailure):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestCascadingFailureCopy(Aux_TestCascadingFailure):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestLoading2envDontCrashCopy(Aux_TestLoading2envDontCrash):
    def setUp(self):
        super().setUp()
        self.env1_orig = self.env1
        self.env1 = self.env1.copy()
        self.env2_orig = self.env2
        self.env2 = self.env2.copy()


class TestDeactivateForecastCopy(Aux_TestDeactivateForecast):
    def setUp(self):
        super().setUp()
        self.env1_orig = self.env1
        self.env1 = self.env1.copy()


class TestMaxIterCopy(Aux_TestMaxIter):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestAgentCopy(Aux_TestAgent):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


if __name__ == "__main__":
    unittest.main()
