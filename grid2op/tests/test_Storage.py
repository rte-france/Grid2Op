# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import pdb
import time
import warnings

from grid2op.tests.helper_path_test import *

from grid2op.Exceptions import *
from grid2op.Environment import Environment
from grid2op.Backend import PandaPowerBackend
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChronicsHandler, GridStateFromFile, ChangeNothing
from grid2op.Reward import L2RPNReward
from grid2op.MakeEnv import make
from grid2op.Rules import RulesChecker, DefaultRules
from grid2op.Action import *
from grid2op.dtypes import dt_float

DEBUG = False
PROFILE_CODE = False
if PROFILE_CODE:
    import cProfile

import warnings
warnings.simplefilter("error")


class TestStorageEnv(unittest.TestCase):
    """test the env part of the storage functionality"""
    pass