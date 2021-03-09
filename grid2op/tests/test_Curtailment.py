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

import grid2op
from grid2op.Parameters import Parameters
from grid2op.dtypes import dt_float
from grid2op.Action import PlayableAction, CompleteAction

import warnings

# TODO check when there is also redispatching


class TestCurtailmentEnv(HelperTests):
    """test the env part of the storage functionality"""
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            param = Parameters()
            param.NO_OVERFLOW_DISCONNECTION = True
            self.env1 = grid2op.make("educ_case14_storage", test=True, action_class=PlayableAction, param=param)
            self.env2 = self.env1.copy()

    def tearDown(self) -> None:
        self.env1.close()
        self.env2.close()

    def test_action_ok(self):
        """test a storage action is supported (basic test)"""
        act = self.env1.action_space({"curtail": [(2, 0.5)]})
        str_ = act.__str__()
        real_str = 'This action will:\n' \
                   '\t - NOT change anything to the injections\n' \
                   '\t - NOT perform any redispatching action\n' \
                   '\t - NOT modify any storage capacity\n' \
                   '\t - Perform the following curtailment:\n' \
                   '\t \t - Limit unit "gen_5_2" to 50.0% of its Pmax (setpoint: 0.500)\n' \
                   '\t - NOT force any line status\n' \
                   '\t - NOT switch any line status\n\t - NOT switch anything in the topology\n' \
                   '\t - NOT force any particular bus configuration'
        assert str_ == real_str

    def test_curtailment_no_effect(self):
        act = self.env1.action_space({"curtail": [(2, 5)]})
        obs1, *_ = self.env1.step(act)
        obs2, *_ = self.env2.step(self.env2.action_space())
        assert obs1 == obs2

    def test_curtailment_effect(self):
        """
        test different kind of equations when curtailment has been applied
        in this example the 23% of curtailment is just the right amont so that, during the
        """
        gen_id = 2
        ratio_max = 0.23
        act = self.env1.action_space({"curtail": [(gen_id, ratio_max)]})
        self.env1.step(act)
        self.env2.step(self.env2.action_space())
        for _ in range(100):
            obs1_3, reward1_3, done1_3, info1_3 = self.env1.step(self.env1.action_space())
            obs2_3, reward2_3, done2_3, info2_3 = self.env2.step(self.env2.action_space())
            np.array_equal(obs1_3.gen_p, obs2_3.gen_p)
            if done1_3 or done2_3:
                break
            assert abs(np.sum(obs1_3.actual_dispatch) + (obs1_3.gen_p[gen_id] - obs2_3.gen_p[gen_id])) <= 1e-4
            this_prod_th = obs1_3.gen_p_before_curtail[self.env1.gen_renewable]
            th_prod = obs2_3.gen_p[self.env1.gen_renewable]
            assert np.max(np.abs(this_prod_th - th_prod)) <= 1e-4
            assert obs1_3.gen_p[gen_id] <= ratio_max * obs1_3.gen_pmax[gen_id] + 1e-5
            amount_curtailed = (np.sum(this_prod_th - obs1_3.gen_p[self.env1.gen_renewable]))
            assert abs(np.sum(obs1_3.actual_dispatch) - amount_curtailed) <= 1e-4


if __name__ == "__main__":
    unittest.main()
