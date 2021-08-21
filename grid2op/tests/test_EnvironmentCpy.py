# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


# This module will test that the environment, when copied, works as expected (ie with making some basic tests
# for the results of "env.copy()"

import copy
import pdb
import time
import warnings

from grid2op.tests.helper_path_test import *

from grid2op.Reward import L2RPNReward
from grid2op.MakeEnv import make
from grid2op.tests.test_Environment import (TestLoadingBackendPandaPower, TestIllegalAmbiguous, TestOtherReward,
                                            TestResetOk, TestLineChangeLastBus, TestResetAfterCascadingFailure,
                                            TestCascadingFailure, TestLoading2envDontCrash,
                                            TestDeactivateForecast, TestMaxIter
                                            )
from grid2op.tests.test_Agent import TestAgent

DEBUG = False
PROFILE_CODE = False
if PROFILE_CODE:
    import cProfile


class TestLoadingBackendPandaPowerCopy(TestLoadingBackendPandaPower):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestIllegalAmbiguousCopy(TestIllegalAmbiguous):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestOtherRewardCopy(TestOtherReward):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestResetOkCopy(TestResetOk):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()

    def test_reset_after_blackout_withdetailed_info(self, env=None):
        backend = self.make_backend(detailed_infos_for_cascading_failures=True)
        if env is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env = make("rte_case5_example", test=True, reward_class=L2RPNReward,
                           other_rewards={"test": L2RPNReward},
                           backend=backend)
        super().test_reset_after_blackout_withdetailed_info(env=env.copy())


class TestLineChangeLastBusCopy(TestLineChangeLastBus):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestResetAfterCascadingFailureCopy(TestResetAfterCascadingFailure):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestCascadingFailureCopy(TestCascadingFailure):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestLoading2envDontCrashCopy(TestLoading2envDontCrash):
    def setUp(self):
        super().setUp()
        self.env1_orig = self.env1
        self.env1 = self.env1.copy()
        self.env2_orig = self.env2
        self.env2 = self.env2.copy()


class TestDeactivateForecastCopy(TestDeactivateForecast):
    def setUp(self):
        super().setUp()
        self.env1_orig = self.env1
        self.env1 = self.env1.copy()


class TestMaxIterCopy(TestMaxIter):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


class TestAgentCopy(TestAgent):
    def setUp(self):
        super().setUp()
        self.env_orig = self.env
        self.env = self.env.copy()


if __name__ == "__main__":
    unittest.main()
