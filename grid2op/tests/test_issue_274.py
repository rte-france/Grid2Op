# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import grid2op
import unittest
import numpy as np


class Issue274Tester(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_icaps_2021", test=True, _add_to_name=type(self).__name__)

    def test_same_opponent_state(self):
        """test that the opponent state is correctly copied"""
        self.env.seed(3)
        self.env.reset()
        init_attack_times = 1 * self.env._opponent._attack_times
        env_cpy = self.env.copy()
        after_attack_times = 1 * self.env._opponent._attack_times
        copy_attack_times = 1 * env_cpy._opponent._attack_times
        assert np.all(init_attack_times == after_attack_times)
        assert np.all(init_attack_times == copy_attack_times)

    def test_same_opponent_space(self):
        """test that the opponent space state (in particular the current attack) is properly copied"""
        self.env.seed(3)
        self.env.reset()
        import pdb

        init_attack_times = 1 * self.env._opponent._attack_times
        assert np.all(init_attack_times == [5, 105, 180])
        for i in range(5):
            obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is None
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        init_line_attacked = np.where(info["opponent_attack_line"])[0]

        for i in range(2):
            env_cpy = self.env.copy()
            *_, info = self.env.step(self.env.action_space())
            *_, info_cpy = env_cpy.step(self.env.action_space())
            assert (
                info["opponent_attack_line"] is not None
            ), f"no line attacked at iteration {i}"
            assert (
                info_cpy["opponent_attack_line"] is not None
            ), f"no line attacked at iteration {i} for the copy env"
            line_attacked = np.where(info["opponent_attack_line"])[0]
            cpy_line_attacked = np.where(info_cpy["opponent_attack_line"])[0]
            assert (
                init_line_attacked == line_attacked
            ), f"wrong line attack at iteration {i}"
            assert (
                init_line_attacked == cpy_line_attacked
            ), f"wrong line attack at iteration {i} for the copy env"
