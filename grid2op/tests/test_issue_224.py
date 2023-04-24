# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings

import grid2op
from grid2op.Chronics import ChangeNothing
from grid2op.tests.helper_path_test import *
from grid2op.Runner import Runner
from grid2op.Parameters import Parameters


class Issue224Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_nm = os.path.join(
                PATH_DATA_TEST, "l2rpn_neurips_2020_track1_with_alarm"
            )
            self.env = grid2op.make(env_nm, test=True, chronics_class=ChangeNothing)
            self.env.seed(0)
            self.env.reset()

    def test_env_alarmtime_default(self):
        """test default values are correct"""
        assert self.env.parameters.ALARM_WINDOW_SIZE == 12
        assert self.env.parameters.ALARM_BEST_TIME == 12
        runner = Runner(**self.env.get_params_for_runner())
        env_runner = runner.init_env()
        assert env_runner.parameters.ALARM_WINDOW_SIZE == 12
        assert env_runner.parameters.ALARM_BEST_TIME == 12

    def test_env_alarmtime_changed(self):
        """test everything is correct when something is modified"""
        param = Parameters()
        param.ALARM_WINDOW_SIZE = 99
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_nm = os.path.join(
                PATH_DATA_TEST, "l2rpn_neurips_2020_track1_with_alarm"
            )
            env = grid2op.make(
                env_nm, test=True, chronics_class=ChangeNothing, param=param
            )
        assert env.parameters.ALARM_WINDOW_SIZE == 99
        assert env.parameters.ALARM_BEST_TIME == 12
        runner = Runner(**env.get_params_for_runner())
        env_runner = runner.init_env()
        assert env_runner.parameters.ALARM_WINDOW_SIZE == 99
        assert env_runner.parameters.ALARM_BEST_TIME == 12

        param = Parameters()
        param.ALARM_BEST_TIME = 42
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_nm = os.path.join(
                PATH_DATA_TEST, "l2rpn_neurips_2020_track1_with_alarm"
            )
            env = grid2op.make(
                env_nm, test=True, chronics_class=ChangeNothing, param=param
            )
        assert env.parameters.ALARM_WINDOW_SIZE == 12
        assert env.parameters.ALARM_BEST_TIME == 42
        runner = Runner(**env.get_params_for_runner())
        env_runner = runner.init_env()
        assert env_runner.parameters.ALARM_WINDOW_SIZE == 12
        assert env_runner.parameters.ALARM_BEST_TIME == 42
