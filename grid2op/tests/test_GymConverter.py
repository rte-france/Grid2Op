# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

# TODO test the json part but... https://github.com/openai/gym-http-api/issues/62 or https://github.com/openai/gym/issues/1841
import tempfile
import json
from grid2op.dtypes import dt_float, dt_bool, dt_int
from grid2op.tests.helper_path_test import *
from grid2op.MakeEnv import make
from grid2op.Converter import IdToAct, ToVect
from grid2op.gym_compat import GymActionSpace, GymObservationSpace
from grid2op.gym_compat import GymEnv
from grid2op.gym_compat import ContinuousToDiscreteConverter

import pdb

import warnings
warnings.simplefilter("error")


class BaseTestGymConverter:
    def __init__(self):
        self.tol = 1e-6

    def _aux_test_json(self, space, obj=None):
        if obj is None:
            obj = space.sample()
        obj_json = space.to_jsonable([obj])
        # test save to json
        with tempfile.TemporaryFile(mode="w") as f:
            json.dump(obj_json, fp=f)

        # test read from json
        obj2 = space.from_jsonable(obj_json)[0]

        # test they are equal
        for k, v in obj2.items():
            assert k in obj
            tmp = obj[k]
            if isinstance(tmp, (int, float, dt_float, dt_int, dt_bool)):
                assert np.all(np.abs(float(obj[k]) - float(obj2[k])) <= self.tol)
            elif len(tmp) == 1:
                assert np.all(np.abs(float(obj[k]) - float(obj2[k])) <= self.tol)
            else:
                assert np.all(np.abs(obj[k].astype(dt_float) - obj2[k].astype(dt_float)) <= self.tol)
        for k, v in obj.items():
            assert k in obj2  # make sure every keys of obj are in obj2


class TestWithoutConverter(unittest.TestCase, BaseTestGymConverter):
    def setUp(self) -> None:
        BaseTestGymConverter.__init__(self)

    def test_creation(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("l2rpn_wcci_2020", test=True) as env:
                # test i can create
                obs_space = GymObservationSpace(env)
                act_space = GymActionSpace(env)

    def test_json(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("l2rpn_wcci_2020", test=True) as env:
                # test i can create
                obs_space = GymObservationSpace(env)
                act_space = GymActionSpace(env)

                obs_space.seed(0)
                act_space.seed(0)

                self._aux_test_json(obs_space)
                self._aux_test_json(act_space)

    def test_to_from_gym_obs(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("l2rpn_wcci_2020", test=True) as env:
                obs_space = GymObservationSpace(env)

                obs = env.reset()
                gym_obs = obs_space.to_gym(obs)
                self._aux_test_json(obs_space, gym_obs)
                assert obs_space.contains(gym_obs)
                obs2 = obs_space.from_gym(gym_obs)
                assert obs == obs2

                for i in range(10):
                    obs, *_ = env.step(env.action_space())
                    gym_obs = obs_space.to_gym(obs)
                    self._aux_test_json(obs_space, gym_obs)
                    assert obs_space.contains(gym_obs), "gym space does not contain the observation for ts {}".format(i)
                    obs2 = obs_space.from_gym(gym_obs)
                    assert obs == obs2, "obs and converted obs are not equal for ts {}".format(i)

    def test_to_from_gym_act(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("l2rpn_wcci_2020", test=True) as env:
                act_space = GymActionSpace(env)

                act = env.action_space()
                gym_act = act_space.to_gym(act)
                self._aux_test_json(act_space, gym_act)
                assert act_space.contains(gym_act)
                act2 = act_space.from_gym(gym_act)
                assert act == act2

                act_space.seed(0)
                for i in range(10):
                    gym_act = act_space.sample()
                    act = act_space.from_gym(gym_act)
                    self._aux_test_json(act_space, gym_act)
                    gym_act2 = act_space.to_gym(act)
                    act2 = act_space.from_gym(gym_act2)
                    assert act == act2


class BaseTestConverter(BaseTestGymConverter):
    def init_converter(self, env):
        raise NotImplementedError()

    def test_creation(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("l2rpn_wcci_2020", test=True) as env:
                # test i can create
                converter = self.init_converter(env)
                act_space = GymActionSpace(env=env, converter=converter)
                act_space.sample()

    def test_json(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("l2rpn_wcci_2020", test=True) as env:
                # test i can create
                converter = self.init_converter(env)
                act_space = GymActionSpace(env=env, converter=converter)
                act_space.seed(0)
                self._aux_test_json(act_space)

    def test_to_from_gym_act(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("l2rpn_wcci_2020", test=True) as env:
                converter = self.init_converter(env)
                act_space = GymActionSpace(env=env, converter=converter)
                act_space.seed(0)
                converter.seed(0)

                gym_act = act_space.sample()
                act = act_space.from_gym(gym_act)
                self._aux_test_json(act_space, gym_act)
                gym_act2 = act_space.to_gym(act)
                act2 = act_space.from_gym(gym_act2)
                g2op_act = converter.convert_act(act)
                g2op_act2 = converter.convert_act(act2)
                assert g2op_act == g2op_act2

                act_space.seed(0)
                for i in range(10):
                    gym_act = act_space.sample()
                    act = act_space.from_gym(gym_act)
                    self._aux_test_json(act_space, gym_act)
                    gym_act2 = act_space.to_gym(act)
                    act2 = act_space.from_gym(gym_act2)
                    g2op_act = converter.convert_act(act)
                    g2op_act2 = converter.convert_act(act2)
                    assert g2op_act == g2op_act2


class TestIdToAct(unittest.TestCase, BaseTestConverter):
    def init_converter(self, env):
        return IdToAct(env.action_space)

    def setUp(self) -> None:
        BaseTestGymConverter.__init__(self)


class TestToVect(unittest.TestCase, BaseTestConverter):
    def init_converter(self, env):
        return ToVect(env.action_space)

    def setUp(self) -> None:
        BaseTestGymConverter.__init__(self)


class TestDropAttr(unittest.TestCase):
    """test the method to remove part of the attribute of the action / observation space"""
    def test_keep_only_attr(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("educ_case14_redisp", test=True)
            gym_env = GymEnv(env)
            attr_kept = sorted(("rho", "line_status", "actual_dispatch", "target_dispatch"))
            ob_space = gym_env.observation_space
            ob_space = ob_space.keep_only_attr(attr_kept)
            assert np.all(sorted(ob_space.spaces.keys()) == attr_kept)

    def test_ignore_attr(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("educ_case14_redisp", test=True)
            gym_env = GymEnv(env)
            attr_deleted = sorted(("rho", "line_status", "actual_dispatch", "target_dispatch"))
            ob_space = gym_env.observation_space
            ob_space = ob_space.ignore_attr(attr_deleted)
            for el in attr_deleted:
                assert not el in ob_space.spaces


class TestContinuousToDiscrete(unittest.TestCase):
    """test the ContinuousToDiscreteConverter converter"""
    def setUp(self) -> None:
        self.tol = 1e-4

    def test_split_in_3(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("educ_case14_redisp", test=True)
            gym_env = GymEnv(env)
            act_space = gym_env.action_space
            act_space = act_space.reencode_space("redispatch",
                                                 ContinuousToDiscreteConverter(nb_bins=3,
                                                                               init_space=act_space["redispatch"])
                                                 )

            # with 3 interval like [-10, 10] (the second generator)
            # should be split => 0 -> [-10, -3.33), 1 => [-3.33, 3.33), [3.33, 10.]
            # test the "all 0" action (all 0 => encoded to 1, because i have 3 bins)
            g2op_object = np.array([0., 0., 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [1, 1, 0, 0, 0, 1])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(res2 == 0.)

            # test the all 0 action, but one is not 0 (negative)
            g2op_object = np.array([0., -3.2, 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [1, 1, 0, 0, 0, 1])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(res2 == 0.)

            # test the all 0 action, but one is not 0 (positive)
            g2op_object = np.array([0., 3.2, 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [1, 1, 0, 0, 0, 1])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(res2 == 0.)

            # test one is 2
            g2op_object = np.array([0., 3.4, 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [1, 2, 0, 0, 0, 1])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(np.abs(res2 - [0., 5.0, 0., 0., 0., 0.]) <= self.tol)

            # test one is 0
            g2op_object = np.array([0., -3.4, 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [1, 0, 0, 0, 0, 1])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(np.abs(res2 - [0., -5.0, 0., 0., 0., 0.]) <= self.tol)

    def test_split_in_5(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("educ_case14_redisp", test=True)
            gym_env = GymEnv(env)
            act_space = gym_env.action_space
            act_space = act_space.reencode_space("redispatch",
                                                 ContinuousToDiscreteConverter(nb_bins=5,
                                                                               init_space=act_space["redispatch"])
                                                 )

            # with 5
            g2op_object = np.array([0., 0., 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [2, 2, 0, 0, 0, 2])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(res2 == 0.)

            # test the all 0 action, but one is not 0 (negative)
            g2op_object = np.array([0., -1.9, 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [2, 2, 0, 0, 0, 2])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(res2 == 0.)

            # positive side
            g2op_object = np.array([0., 2.1, 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [2, 3, 0, 0, 0, 2])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(np.abs(res2 - [0., 3.33333, 0., 0., 0., 0.]) <= self.tol)

            g2op_object = np.array([0., 5.9, 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [2, 3, 0, 0, 0, 2])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(np.abs(res2 - [0., 3.33333, 0., 0., 0., 0.]) <= self.tol)

            g2op_object = np.array([0., 6.1, 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [2, 4, 0, 0, 0, 2])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(np.abs(res2 - [0., 6.666666, 0., 0., 0., 0.]) <= self.tol)

            # negative side
            g2op_object = np.array([0., -2.1, 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [2, 1, 0, 0, 0, 2])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(np.abs(res2 - [0., -3.3333, 0., 0., 0., 0.]) <= self.tol)

            g2op_object = np.array([0., -5.9, 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [2, 1, 0, 0, 0, 2])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(np.abs(res2 - [0., -3.33333, 0., 0., 0., 0.]) <= self.tol)

            g2op_object = np.array([0., -6.1, 0., 0., 0., 0.])
            res = act_space._keys_encoding["_redispatch"].g2op_to_gym(g2op_object)
            assert np.all(res == [2, 0, 0, 0, 0, 2])
            res2 = act_space._keys_encoding["_redispatch"].gym_to_g2op(res)
            assert np.all(np.abs(res2 - [0., -6.666666, 0., 0., 0., 0.]) <= self.tol)
