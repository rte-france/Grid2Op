# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
from grid2op.Reward import BaseReward
import warnings
from grid2op.dtypes import dt_float

import pdb


class TestReward(BaseReward):
    def __init__(self):
        super().__init__()
        self.reward_min = dt_float(100.0)  # Note difference from below
        self.reward_max = dt_float(0.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error:
            return dt_float(-10.0)
        else:
            return dt_float(1.0)


class Issue146Tester(unittest.TestCase):
    def test_issue_146(self):
        """
        the reward helper skipped the call to the reward when "has_error" was True
        This was not really an issue... but rather a enhancement, but still
        """

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "rte_case14_realistic", test=True, reward_class=TestReward,
                _add_to_name=type(self).__name__
            )

        action = env.action_space(
            {"set_bus": {"substations_id": [(1, [2, 2, 1, 1, 2, -1])]}}
        )
        obs, reward, done, info = env.step(action)
        assert done
        assert reward == dt_float(
            -10.0
        ), 'reward should be -10.0 and not "reward_min" (ie 100.)'


if __name__ == "__main__":
    unittest.main()
