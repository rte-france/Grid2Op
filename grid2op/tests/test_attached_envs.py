# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import unittest
import grid2op

from grid2op.Action import (PowerlineSetAction, PlayableAction, DontAct)
from grid2op.Observation import CompleteObservation
from grid2op.Opponent import GeometricOpponent

import pdb

# TODO refactor to have 1 base class, maybe


class TestL2RPNNEURIPS2020_Track1(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_neurips_2020_track1", test=True, _add_to_name=type(self).__name__)
            self.env.seed(0)
            _ = self.env.reset()

    def test_elements(self):
        assert self.env.n_sub == 36
        assert self.env.n_line == 59
        assert self.env.n_load == 37
        assert self.env.n_gen == 22
        assert self.env.n_storage == 0

    def test_opponent(self):
        assert issubclass(self.env._opponent_action_class, PowerlineSetAction)
        assert self.env._opponent_action_space.n == self.env.n_line

    def test_action_space(self):
        assert issubclass(self.env.action_space.subtype, PlayableAction)
        assert self.env.action_space.n == 494, f"{self.env.action_space.n} instead of 494"

    def test_observation_space(self):
        assert issubclass(self.env.observation_space.subtype, CompleteObservation)
        size_th = 1266
        assert self.env.observation_space.n == size_th, (
            f"obs space size is {self.env.observation_space.n}, should be {size_th}"
        )

    def test_random_action(self):
        """test i can perform some step (random)"""
        i = 0
        for i in range(10):
            act = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(act)
            if done:
                break
        assert i >= 1, (
            "could not perform the random action test because it games over first time step. "
            "Please fix the test and try again"
        )


class TestL2RPNICAPS2021(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_icaps_2021", test=True, _add_to_name=type(self).__name__)
            self.env.seed(0)
            _ = self.env.reset()

    def test_elements(self):
        assert self.env.n_sub == 36
        assert self.env.n_line == 59
        assert self.env.n_load == 37
        assert self.env.n_gen == 22
        assert self.env.n_storage == 0

    def test_opponent(self):
        assert issubclass(self.env._opponent_action_class, PowerlineSetAction)
        assert isinstance(self.env._opponent, GeometricOpponent)
        assert self.env._opponent_action_space.n == self.env.n_line

    def test_action_space(self):
        assert issubclass(self.env.action_space.subtype, PlayableAction)
        assert self.env.action_space.n == 519, (
            f"act space size is {self.env.action_space.n}, should be {519}"
        )

    def test_observation_space(self):
        assert issubclass(self.env.observation_space.subtype, CompleteObservation)
        size_th = 1363
        assert self.env.observation_space.n == size_th, (
            f"obs space size is "
            f"{self.env.observation_space.n}, "
            f"should be {size_th}"
        )

    def test_random_action(self):
        """test i can perform some step (random)"""
        i = 0
        for i in range(10):
            act = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(act)
            if done:
                break
        assert i >= 1, (
            "could not perform the random action test because it games over first time step. "
            "Please fix the test and try again"
        )


class TestL2RPNNEURIPS2020_Track2(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_neurips_2020_track2", test=True, _add_to_name=type(self).__name__)
            self.env.seed(2)  # 0 or 1 breaks the test `test_random_action`
            _ = self.env.reset()

    def test_elements(self):
        assert self.env.n_sub == 118
        assert self.env.n_line == 186
        assert self.env.n_load == 99
        assert self.env.n_gen == 62
        assert self.env.n_storage == 0

    def test_opponent(self):
        assert issubclass(self.env._opponent_action_class, DontAct)
        assert self.env._opponent_action_space.n == 0

    def test_action_space(self):
        assert issubclass(self.env.action_space.subtype, PlayableAction)
        assert self.env.action_space.n == 1500, f"{self.env.action_space.n} instead of 1500"

    def test_observation_space(self):
        assert issubclass(self.env.observation_space.subtype, CompleteObservation)
        size_th = 3868
        assert self.env.observation_space.n == size_th, (
            f"obs space size is {self.env.observation_space.n}, should be {size_th}"
        )

    def test_random_action(self):
        """test i can perform some step (random)"""
        i = 0
        for i in range(10):
            act = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(act)
            if done:
                break
        assert i >= 1, (
            "could not perform the random action test because it games over first 10 steps. "
            "Please fix the test and try again"
        )


class TestL2RPN_CASE14_SANDBOX(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
            self.env.seed(42)
            _ = self.env.reset()

    def test_elements(self):
        assert self.env.n_sub == 14
        assert self.env.n_line == 20
        assert self.env.n_load == 11
        assert self.env.n_gen == 6
        assert self.env.n_storage == 0

    def test_opponent(self):
        assert issubclass(self.env._opponent_action_class, DontAct)
        assert self.env._opponent_action_space.n == 0

    def test_action_space(self):
        assert issubclass(self.env.action_space.subtype, PlayableAction)
        assert self.env.action_space.n == 166, f"{self.env.action_space.n} instead of 166"

    def test_observation_space(self):
        assert issubclass(self.env.observation_space.subtype, CompleteObservation)
        size_th = 467
        assert self.env.observation_space.n == size_th, (
            f"obs space size is {self.env.observation_space.n}," f"should be {size_th}"
        )

    def test_random_action(self):
        """test i can perform some step (random)"""
        i = 0
        for i in range(10):
            act = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(act)
            if done:
                break
        assert i >= 1, (
            "could not perform the random action test because it games over first time step. "
            "Please fix the test and try again"
        )


class TestEDUC_CASE14_REDISP(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_redisp", test=True, _add_to_name=type(self).__name__)
            self.env.seed(0)
            _ = self.env.reset()

    def test_elements(self):
        assert self.env.n_sub == 14
        assert self.env.n_line == 20
        assert self.env.n_load == 11
        assert self.env.n_gen == 6
        assert self.env.n_storage == 0

    def test_opponent(self):
        assert issubclass(self.env._opponent_action_class, DontAct)
        assert self.env._opponent_action_space.n == 0

    def test_action_space(self):
        assert issubclass(self.env.action_space.subtype, PlayableAction)
        assert self.env.action_space.n == 26, f"{self.env.action_space.n} instead of 26"

    def test_observation_space(self):
        assert issubclass(self.env.observation_space.subtype, CompleteObservation)
        size_th = 467
        assert self.env.observation_space.n == size_th, (
            f"obs space size is {self.env.observation_space.n}," f"should be {size_th}"
        )

    def test_random_action(self):
        """test i can perform some step (random)"""
        i = 0
        for i in range(10):
            act = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(act)
            if done:
                break
        assert i >= 1, (
            "could not perform the random action test because it games over first time step. "
            "Please fix the test and try again"
        )


class TestEDUC_STORAGE(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage", test=True, _add_to_name=type(self).__name__)
            self.env.seed(0)
            _ = self.env.reset()

    def test_elements(self):
        assert self.env.n_sub == 14
        assert self.env.n_line == 20
        assert self.env.n_load == 11
        assert self.env.n_gen == 6
        assert self.env.n_storage == 2

    def test_opponent(self):
        assert issubclass(self.env._opponent_action_class, DontAct)
        assert self.env._opponent_action_space.n == 0

    def test_action_space(self):
        assert issubclass(self.env.action_space.subtype, PlayableAction)
        assert self.env.action_space.n == 28, f"{self.env.action_space.n} instead of 28"

    def test_observation_space(self):
        assert issubclass(self.env.observation_space.subtype, CompleteObservation)
        size_th = 475
        assert self.env.observation_space.n == size_th, (
            f"obs space size is {self.env.observation_space.n}," f"should be {size_th}"
        )

    def test_random_action(self):
        """test i can perform some step (random)"""
        i = 0
        for i in range(10):
            act = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(act)
            if done:
                break
        assert i >= 1, (
            "could not perform the random action test because it games over first time step. "
            "Please fix the test and try again"
        )


if __name__ == "__main__":
    unittest.main()
