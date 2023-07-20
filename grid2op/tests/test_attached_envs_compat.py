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
import numpy as np

from grid2op.Space import GridObjects
from grid2op.Action.powerlineSetAction import PowerlineSetAction
from grid2op.Action.PlayableAction import PlayableAction
from grid2op.Observation.completeObservation import CompleteObservation
from grid2op.Action.dontAct import DontAct

import pdb

# TODO refactor to have 1 base class, maybe


class TestL2RPNNEURIPS2020_Track1Compat(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "l2rpn_neurips_2020_track1",
                test=True,
                _compat_glop_version=GridObjects.BEFORE_COMPAT_VERSION,
                _add_to_name="test_attached_compat_0",
            )
            self.env.seed(0)

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
        assert self.env.action_space.n == 494

    def test_observation_space(self):
        assert issubclass(self.env.observation_space.subtype, CompleteObservation)
        assert self.env.observation_space.n == 1266

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


class TestL2RPNNEURIPS2020_Track2Compat(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "l2rpn_neurips_2020_track2",
                test=True,
                _compat_glop_version=GridObjects.BEFORE_COMPAT_VERSION,
                _add_to_name="test_attached_compat_1",
            )
            self.env.seed(0)

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
        assert self.env.action_space.n == 1500

    def test_observation_space(self):
        assert issubclass(self.env.observation_space.subtype, CompleteObservation)
        assert (
            "curtailment" not in self.env.observation_space.subtype.attr_list_vect
        ), "curtailment should not be there"
        assert self.env.observation_space.n == 3868

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


class TestL2RPN_CASE14_SANDBOXCompat(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "l2rpn_case14_sandbox",
                test=True,
                _compat_glop_version=GridObjects.BEFORE_COMPAT_VERSION,
                _add_to_name="test_attached_compat_2",
            )
            self.env.seed(42)

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
        assert self.env.action_space.n == 160

    def test_observation_space(self):
        assert issubclass(self.env.observation_space.subtype, CompleteObservation)
        assert self.env.observation_space.n == 420

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


class TestEDUC_CASE14_REDISPCompat(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_redisp",
                test=True,
                _compat_glop_version=GridObjects.BEFORE_COMPAT_VERSION,
                _add_to_name="test_attached_compat_3",
            )
            self.env.seed(0)

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
        assert self.env.action_space.n == 26

    def test_observation_space(self):
        assert issubclass(self.env.observation_space.subtype, CompleteObservation)
        assert self.env.observation_space.n == 420

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


class TestCompatMode_WhenStorage(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage",
                test=True,
                _compat_glop_version=GridObjects.BEFORE_COMPAT_VERSION,
                _add_to_name="test_attached_compat_4",
            )
            self.env.seed(0)

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
        assert self.env.action_space.n == 26

    def test_observation_space(self):
        assert issubclass(self.env.observation_space.subtype, CompleteObservation)
        assert self.env.observation_space.n == 420

    def test_same_env_as_no_storage(self):
        res = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_redisp", test=True)
        for attr in self.env.observation_space.attr_list_vect:
            tmp = getattr(self.env.observation_space._template_obj, attr).shape
            tmp2 = getattr(env.observation_space._template_obj, attr).shape
            if tmp:
                res += tmp[0]
            else:
                res += 1
            assert tmp == tmp2, f"error for {attr}"

        # TODO for all the other attributes too (maybe they are, i'm not sure)...
        for type_ in ["load", "gen", "line_or", "line_ex"]:
            for el in [
                f"{type_}_pos_topo_vect",
                f"{type_}_to_sub_pos",
                f"{type_}_to_subid",
            ]:
                assert np.array_equal(
                    getattr(type(env), el), getattr(type(self.env), el)
                ), f"error for {el}"

        for el in ["sub_info", "grid_objects_types", "_topo_vect_to_sub"]:
            assert np.array_equal(
                getattr(type(env), el), getattr(type(self.env), el)
            ), f"error for {el}"

    def test_random_action(self):
        """test i can perform some step (random)"""
        i = 0
        for i in range(10):
            act = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(act)
            if done:
                pdb.set_trace()
                break
        assert i >= 1, (
            "could not perform the random action test because it games over first time step. "
            "Please fix the test and try again"
        )

if __name__ == "__main__":
    unittest.main()
