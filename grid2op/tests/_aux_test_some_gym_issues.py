# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


"""put at the same place some test for gym, mainly to run them in a single command"""

import unittest

from test_issue_185 import Issue185Tester
from test_issue_196 import Issue196Tester
from test_issue_281 import Issue281Tester
from test_issue_282 import Issue282Tester
from test_issue_283 import Issue283Tester
from test_issue_379 import Issue379Tester
from test_issue_407 import Issue407Tester
from test_issue_418 import Issue418Tester
from test_defaultgym_compat import (TestGymCompatModule,
                             TestBoxGymObsSpace,
                             TestBoxGymActSpace,
                             TestMultiDiscreteGymActSpace,
                             TestDiscreteGymActSpace,
                             TestAllGymActSpaceWithAlarm,
                             TestGOObsInRange
                             )
from test_gym_env_renderer import TestGymEnvRenderer
from test_GymConverter import (TestWithoutConverterWCCI,
                               TestIdToAct,
                               TestToVect,
                               TestDropAttr,
                               TestContinuousToDiscrete,
                               TestWithoutConverterStorage,
                               TestDiscreteActSpace,
                               )
from test_timeOutEnvironment import TestTOEnvGym
from test_pickling import TestMultiProc
from test_alert_gym_compat import *
from test_basic_env_ls import TestBasicEnvironmentGym
from test_gym_asynch_env import *
from test_l2rpn_idf_2023 import TestL2RPNIDF2023Tester
from test_MaskedEnvironment import TestMaskedEnvironmentGym
from test_multidiscrete_act_space import *
from test_n_busbar_per_sub import TestGym_3busbars, TestGym_1busbar
from test_timeOutEnvironment import TestTOEnvGym


if __name__ == "__main__":
    unittest.main()
