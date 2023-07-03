# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
# TODO test the json part but... https://github.com/openai/gym-http-api/issues/62 or https://github.com/openai/gym/issues/1841
# TODO when functions are called in the converter (especially with graph)
from grid2op.tests.helper_path_test import *


import grid2op
from grid2op.dtypes import dt_float, dt_int
from grid2op.tests.helper_path_test import *
from grid2op.Action import PlayableAction

from grid2op.gym_compat import GymActionSpace, GymObservationSpace
from grid2op.gym_compat import GymEnv
from grid2op.gym_compat import ContinuousToDiscreteConverter
from grid2op.gym_compat import ScalerAttrConverter
from grid2op.gym_compat import MultiToTupleConverter
from grid2op.gym_compat import (
    GYM_AVAILABLE, 
    GYMNASIUM_AVAILABLE,
    BoxGymObsSpace,
    BoxGymActSpace,
    MultiDiscreteActSpace,
    DiscreteActSpace,
)
from grid2op.gym_compat.utils import _compute_extra_power_for_losses, _MAX_GYM_VERSION_RANDINT, GYM_VERSION

import pdb

class AuxilliaryForTest:
    def _aux_GymEnv_cls(self):
        return GymEnv
    
    def _aux_ContinuousToDiscreteConverter_cls(self):
        return ContinuousToDiscreteConverter
    
    def _aux_ScalerAttrConverter_cls(self):
        return ScalerAttrConverter
    
    def _aux_MultiToTupleConverter_cls(self):
        return MultiToTupleConverter
    
    def _aux_BoxGymObsSpace_cls(self):
        return BoxGymObsSpace
    
    def _aux_BoxGymActSpace_cls(self):
        return BoxGymActSpace
    
    def _aux_MultiDiscreteActSpace_cls(self):
        return MultiDiscreteActSpace
    
    def _aux_DiscreteActSpace_cls(self):
        return DiscreteActSpace
    
    def _aux_Box_cls(self):
        if GYMNASIUM_AVAILABLE:
            from gymnasium.spaces import Box
            return Box
        if GYM_AVAILABLE:
            from gym.spaces import Box
            return Box
    
    def _aux_MultiDiscrete_cls(self):
        if GYMNASIUM_AVAILABLE:
            from gymnasium.spaces import MultiDiscrete
            return MultiDiscrete
        if GYM_AVAILABLE:
            from gym.spaces import MultiDiscrete
            return MultiDiscrete
    
    def _aux_Discrete_cls(self):
        if GYMNASIUM_AVAILABLE:
            from gymnasium.spaces import Discrete
            return Discrete
        if GYM_AVAILABLE:
            from gym.spaces import Discrete
            return Discrete
        
    def _aux_Tuple_cls(self):
        if GYMNASIUM_AVAILABLE:
            from gymnasium.spaces import Tuple
            return Tuple
        if GYM_AVAILABLE:
            from gym.spaces import Tuple
            return Tuple
        
    def _aux_Dict_cls(self):
        if GYMNASIUM_AVAILABLE:
            from gymnasium.spaces import Dict
            return Dict
        if GYM_AVAILABLE:
            from gym.spaces import Dict
            return Dict
            
    def _skip_if_no_gym(self):
        if not GYM_AVAILABLE and not GYMNASIUM_AVAILABLE:
            self.skipTest("Gym is not available")
    
    
class _AuxTestGymCompatModule:
    def setUp(self) -> None:
        self._skip_if_no_gym()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "l2rpn_case14_sandbox", test=True, _add_to_name="TestGymCompatModule"
            )
        self.env.seed(0)
        self.env.reset()  # seed part !

    def tearDown(self) -> None:
        self.env.close()

    def test_print_with_no_storage(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "l2rpn_icaps_2021", test=True, _add_to_name="TestGymCompatModule"
            )
        self.env.seed(0)
        self.env.reset()  # seed part !
        env_gym = self._aux_GymEnv_cls()(self.env)
        str_ = env_gym.action_space.__str__()  # this crashed
        str_ = env_gym.observation_space.__str__()

    def test_can_create(self):
        env_gym = self._aux_GymEnv_cls()(self.env)
        assert isinstance(env_gym, self._aux_GymEnv_cls())
        assert isinstance(env_gym.action_space, self._aux_Dict_cls())
        assert isinstance(env_gym.observation_space, self._aux_Dict_cls())
        
    def test_convert_togym(self):
        """test i can create the env"""
        env_gym = self._aux_GymEnv_cls()(self.env)
        dim_act_space = np.sum(
            [
                np.sum(env_gym.action_space[el].shape)
                for el in env_gym.action_space.spaces
            ]
        )
        assert dim_act_space == 166, f"{dim_act_space} != 166"
        dim_obs_space = np.sum(
            [
                np.sum(env_gym.observation_space[el].shape).astype(int)
                for el in env_gym.observation_space.spaces
            ]
        )
        size_th = 536  # as of grid2Op 1.7.1 (where all obs attributes are there)
        assert (
            dim_obs_space == size_th
        ), f"Size should be {size_th} but is {dim_obs_space}"

        # test that i can do basic stuff there
        obs, info = env_gym.reset()
        for k in env_gym.observation_space.spaces.keys():
            assert obs[k] in env_gym.observation_space[k], f"error for key: {k}"
        act = env_gym.action_space.sample()
        obs2, reward2, done2, truncated, info2 = env_gym.step(act)
        assert obs2 in env_gym.observation_space

        # test for the __str__ method
        str_ = self.env.action_space.__str__()
        str_ = self.env.observation_space.__str__()

    def test_ignore(self):
        """test the ignore_attr method"""
        env_gym = self._aux_GymEnv_cls()(self.env)
        env_gym.action_space = env_gym.action_space.ignore_attr("set_bus").ignore_attr(
            "set_line_status"
        )
        dim_act_space = np.sum(
            [
                np.sum(env_gym.action_space[el].shape)
                for el in env_gym.action_space.spaces
            ]
        )
        assert dim_act_space == 89, f"{dim_act_space=} != 89"

    def test_keep_only(self):
        """test the keep_only_attr method"""
        env_gym = self._aux_GymEnv_cls()(self.env)
        env_gym.observation_space = env_gym.observation_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "topo_vect", "actual_dispatch"]
        )
        new_dim_obs_space = np.sum(
            [
                np.sum(env_gym.observation_space[el].shape).astype(int)
                for el in env_gym.observation_space.spaces
            ]
        )
        assert new_dim_obs_space == 100

    def test_scale_attr_converter(self):
        """test a scale_attr converter"""
        env_gym = self._aux_GymEnv_cls()(self.env)
        ob_space = env_gym.observation_space

        key = "actual_dispatch"
        low = -self.env.gen_pmax
        high = 1.0 * self.env.gen_pmax
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"
        ob_space = ob_space.reencode_space(
            "actual_dispatch",
            self._aux_ScalerAttrConverter_cls()(substract=0.0, divide=self.env.gen_pmax),
        )
        env_gym.observation_space = ob_space
        obs, info = env_gym.reset()
        assert key in env_gym.observation_space.spaces
        low = np.zeros(self.env.n_gen) - 1
        high = np.zeros(self.env.n_gen) + 1
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"
        assert obs in env_gym.observation_space

    def test_add_key(self):
        """test the add_key feature"""
        env_gym = self._aux_GymEnv_cls()(self.env)
        shape_ = (self.env.dim_topo, self.env.dim_topo)
        key = "connectivity_matrix"
        env_gym.observation_space.add_key(
            key,
            lambda obs: obs.connectivity_matrix(),
            self._aux_Box_cls()(
                shape=shape_,
                low=np.zeros(shape_, dtype=dt_float),
                high=np.ones(shape_, dtype=dt_float),
                dtype=dt_float,
            ),
        )

        # we highly recommend to "reset" the environment after setting up the observation space
        obs_gym, info = env_gym.reset()
        assert key in env_gym.observation_space.spaces
        assert obs_gym in env_gym.observation_space

    def test_chain_converter(self):
        """test i can do two converters on the same key
        
        this method depends on the version of gym you have installed, tests are made for gym-0.23.1
        """

        from grid2op._glop_platform_info import _IS_LINUX, _IS_WINDOWS, _IS_MACOS

        if _IS_MACOS:
            self.skipTest("Test not suited on macos")
        env_gym = self._aux_GymEnv_cls()(self.env)
        env_gym.action_space = env_gym.action_space.reencode_space(
            "redispatch", self._aux_ContinuousToDiscreteConverter_cls()(nb_bins=11)
        )
        env_gym.action_space.seed(0)
        act_gym = env_gym.action_space.sample()
        if _IS_WINDOWS:
            res = (7, 9, 0, 0, 0, 9)
        else:
            # it's linux
            if GYM_VERSION <= _MAX_GYM_VERSION_RANDINT:
                res = (1, 2, 0, 0, 0, 0)
                res = (5, 3, 0, 0, 0, 1)
                res = (2, 2, 0, 0, 0, 9)
            else:
                res = (0, 6, 0, 0, 0, 5)
                res = (10, 3, 0, 0, 0, 7)
        
        assert np.all(
            act_gym["redispatch"] == res
        ), f'wrong action: {act_gym["redispatch"]}'
        act_gym = env_gym.action_space.sample()
        if _IS_WINDOWS:
            res = (2, 9, 0, 0, 0, 1)
        else:
            # it's linux
            if GYM_VERSION <= _MAX_GYM_VERSION_RANDINT:
                res = (0, 1, 0, 0, 0, 4)
                res = (5, 5, 0, 0, 0, 9)
                res = (0, 9, 0, 0, 0, 7)
            else:
                res = (2, 9, 0, 0, 0, 1)
                res = (7, 5, 0, 0, 0, 8)
        assert np.all(
            act_gym["redispatch"] == res
        ), f'wrong action: {act_gym["redispatch"]}'
        assert isinstance(env_gym.action_space["redispatch"], self._aux_MultiDiscrete_cls())
        env_gym.action_space = env_gym.action_space.reencode_space(
            "redispatch", self._aux_MultiToTupleConverter_cls()()
        )
        assert isinstance(env_gym.action_space["redispatch"], self._aux_Tuple_cls())

        # and now test that the redispatching is properly computed
        env_gym.action_space.seed(0)
        # TODO this doesn't work... because when you seed it appears to use the same seed on all
        # on all the "sub part" of the Tuple.. Thanks gym !
        # see https://github.com/openai/gym/issues/2166
        act_gym = env_gym.action_space.sample()
        if _IS_WINDOWS:
            res_tup = (6, 5, 0, 0, 0, 9)
            res_disp = np.array([0.833333, 0.0, 0.0, 0.0, 0.0, 10.0], dtype=dt_float)
        else:
            # it's linux
            if GYM_VERSION <= _MAX_GYM_VERSION_RANDINT:
                res_tup = (1, 4, 0, 0, 0, 8)
                res_disp = np.array(
                    [-3.3333333, -1.666667, 0.0, 0.0, 0.0, 7.5], dtype=dt_float
                )
                res_tup = (7, 4, 0, 0, 0, 0)
                res_disp = np.array(
                    [1.666667, -1.666667, 0.0, 0.0, 0.0, -12.5], dtype=dt_float
                )
                res_tup = (8, 5, 0, 0, 0, 8)
                res_disp = np.array(
                    [2.5, 0.0, 0.0, 0.0, 0.0, 7.5], dtype=dt_float
                )
            else:
                res_tup = (8, 9, 0, 0, 0, 2)
                res_tup = (8, 2, 0, 0, 0, 9)
                res_disp = np.array(
                    [2.5, -5., 0., 0., 0., 10.], dtype=dt_float
                )
            
        assert (
            act_gym["redispatch"] == res_tup
        ), f'error. redispatch is {act_gym["redispatch"]}'
        act_glop = env_gym.action_space.from_gym(act_gym)
        assert np.array_equal(
            act_glop._redispatch, res_disp
        ), f"error. redispatch is {act_glop._redispatch}"
        act_gym = env_gym.action_space.sample()

        if _IS_WINDOWS:
            res_tup = (5, 8, 0, 0, 0, 10)
            res_disp = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 12.5], dtype=dt_float)
        else:
            # it's linux
            if GYM_VERSION <= _MAX_GYM_VERSION_RANDINT:
                res_tup = (3, 9, 0, 0, 0, 0)
                res_disp = np.array(
                    [-1.6666665, 6.666666, 0.0, 0.0, 0.0, -12.5], dtype=dt_float
                )
                res_tup = (8, 6, 0, 0, 0, 0)
                res_disp = np.array(
                    [2.5, 1.666666, 0.0, 0.0, 0.0, -12.5], dtype=dt_float
                )
                res_tup = (7, 6, 0, 0, 0, 4)
                res_disp = np.array(
                    [1.666667, 1.666666, 0.0, 0.0, 0.0, -2.5], dtype=dt_float
                )
            else:
                res_tup = (4, 2, 0, 0, 0, 5)
                res_tup = (3, 8, 0, 0, 0, 8)
                res_disp = np.array(
                    [-1.6666665, 5., 0.0, 0.0, 0.0, 7.5], dtype=dt_float
                )
        assert (
            act_gym["redispatch"] == res_tup
        ), f'error. redispatch is {act_gym["redispatch"]}'
        act_glop = env_gym.action_space.from_gym(act_gym)
        assert np.allclose(
            act_glop._redispatch, res_disp, atol=1e-5
        ), f"error. redispatch is {act_glop._redispatch}"

    def test_all_together(self):
        """combine all test above (for the action space)"""
        env_gym = self._aux_GymEnv_cls()(self.env)
        env_gym.action_space = env_gym.action_space.ignore_attr("set_bus").ignore_attr(
            "set_line_status"
        )
        env_gym.action_space = env_gym.action_space.reencode_space(
            "redispatch", self._aux_ContinuousToDiscreteConverter_cls()(nb_bins=11)
        )
        env_gym.action_space = env_gym.action_space.reencode_space(
            "change_bus", self._aux_MultiToTupleConverter_cls()()
        )
        env_gym.action_space = env_gym.action_space.reencode_space(
            "change_line_status", self._aux_MultiToTupleConverter_cls()()
        )
        env_gym.action_space = env_gym.action_space.reencode_space(
            "redispatch", self._aux_MultiToTupleConverter_cls()()
        )

        assert isinstance(env_gym.action_space["redispatch"], self._aux_Tuple_cls())
        assert isinstance(env_gym.action_space["change_bus"], self._aux_Tuple_cls())
        assert isinstance(env_gym.action_space["change_line_status"], self._aux_Tuple_cls())

        act_gym = env_gym.action_space.sample()
        act_glop = env_gym.action_space.from_gym(act_gym)
        act_gym2 = env_gym.action_space.to_gym(act_glop)
        act_glop2 = env_gym.action_space.from_gym(act_gym2)

        assert act_gym in env_gym.action_space
        assert act_gym2 in env_gym.action_space

        assert isinstance(act_gym["redispatch"], tuple)
        assert isinstance(act_gym["change_bus"], tuple)
        assert isinstance(act_gym["change_line_status"], tuple)

        # check the gym actions are the same
        for k in act_gym.keys():
            assert np.array_equal(act_gym[k], act_gym2[k]), f"error for {k}"
        for k in act_gym2.keys():
            assert np.array_equal(act_gym[k], act_gym2[k]), f"error for {k}"
        # check grid2op action are the same
        assert act_glop == act_glop2

    def test_low_high_obs_space(self):
        """test the observation space, by default, is properly converted"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "educ_case14_storage", test=True, _add_to_name="TestGymCompatModule"
            )
        env.seed(0)
        env.reset()  # seed part !
        env_gym = self._aux_GymEnv_cls()(env)
        assert "a_ex" in env_gym.observation_space.spaces
        assert np.array_equal(
            env_gym.observation_space["a_ex"].low,
            np.zeros(
                shape=(env.n_line,),
            ),
        )
        assert "a_or" in env_gym.observation_space.spaces
        assert np.array_equal(
            env_gym.observation_space["a_or"].low,
            np.zeros(
                shape=(env.n_line,),
            ),
        )

        key = "actual_dispatch"
        assert key in env_gym.observation_space.spaces
        low = np.minimum(env.gen_pmin, -env.gen_pmax)
        high = np.maximum(-env.gen_pmin, +env.gen_pmax)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "curtailment"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(shape=(env.n_gen,))
        high = np.ones(shape=(env.n_gen,))
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "curtailment_limit"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(shape=(env.n_gen,))
        high = np.ones(shape=(env.n_gen,))
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        # Discrete
        assert "day" in env_gym.observation_space.spaces
        assert "day_of_week" in env_gym.observation_space.spaces
        assert "hour_of_day" in env_gym.observation_space.spaces
        assert "minute_of_hour" in env_gym.observation_space.spaces
        assert "month" in env_gym.observation_space.spaces
        assert "year" in env_gym.observation_space.spaces

        # multi binary
        assert "line_status" in env_gym.observation_space.spaces

        key = "duration_next_maintenance"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(shape=(env.n_line,), dtype=dt_int) - 1
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_int) - 1
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "gen_p"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(shape=(env.n_gen,), dtype=dt_float)
        high = 1.0 * env.gen_pmax
        low -= env._tol_poly
        high += env._tol_poly
        # for "power losses" that are not properly computed in the original data
        extra_for_losses = _compute_extra_power_for_losses(env.observation_space)
        low -= extra_for_losses
        high += extra_for_losses
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "gen_p_before_curtail"
        low = np.zeros(shape=(env.n_gen,), dtype=dt_float)
        high = 1.0 * env.gen_pmax
        low -= env._tol_poly
        high += env._tol_poly
        # for "power losses" that are not properly computed in the original data
        extra_for_losses = _compute_extra_power_for_losses(env.observation_space)
        low -= extra_for_losses
        high += extra_for_losses
        assert key in env_gym.observation_space.spaces
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "gen_q"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_gen,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_gen,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "gen_v"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(shape=(env.n_gen,), dtype=dt_int)
        high = np.full(shape=(env.n_gen,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "load_p"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_load,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_load,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "load_q"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_load,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_load,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "load_v"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_load,), fill_value=0.0, dtype=dt_float)
        high = np.full(shape=(env.n_load,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "p_ex"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "p_or"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "q_ex"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "q_or"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "rho"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=0.0, dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "storage_charge"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_storage,), fill_value=0.0, dtype=dt_float)
        high = env.storage_Emax
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "storage_power"
        assert key in env_gym.observation_space.spaces
        low = -env.storage_max_p_absorb
        high = env.storage_max_p_prod
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "storage_power_target"
        assert key in env_gym.observation_space.spaces
        low = -env.storage_max_p_absorb
        high = env.storage_max_p_prod
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "target_dispatch"
        assert key in env_gym.observation_space.spaces
        low = -env.gen_pmax
        high = env.gen_pmax
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "time_before_cooldown_line"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(env.n_line, dtype=dt_int)
        high = np.zeros(env.n_line, dtype=dt_int) + max(
            env.parameters.NB_TIMESTEP_RECONNECTION,
            env.parameters.NB_TIMESTEP_COOLDOWN_LINE,
            env._oppSpace.attack_max_duration,
        )
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "time_before_cooldown_sub"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(env.n_sub, dtype=dt_int)
        high = (
            np.zeros(env.n_sub, dtype=dt_int) + env.parameters.NB_TIMESTEP_COOLDOWN_SUB
        )
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "time_next_maintenance"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(env.n_line, dtype=dt_int) - 1
        high = np.full(env.n_line, fill_value=np.inf, dtype=dt_int) - 1
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "timestep_overflow"
        assert key in env_gym.observation_space.spaces
        low = np.full(env.n_line, fill_value=np.inf, dtype=dt_int)
        high = np.full(env.n_line, fill_value=np.inf, dtype=dt_int) - 1
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "topo_vect"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(env.dim_topo, dtype=dt_int) - 1
        high = np.zeros(env.dim_topo, dtype=dt_int) + 2
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "v_or"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=0.0, dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        key = "v_ex"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=0.0, dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(
            env_gym.observation_space[key].low, low
        ), f"issue for {key}"
        assert np.array_equal(
            env_gym.observation_space[key].high, high
        ), f"issue for {key}"

        # TODO add tests for the alarm feature and curtailment and storage (if not present already)


class _AuxTestBoxGymObsSpace:
    def setUp(self) -> None:
        self._skip_if_no_gym()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage",
                test=True,
                action_class=PlayableAction,
                _add_to_name="TestBoxGymObsSpace",
            )
        self.env.seed(0)
        self.env.reset()  # seed part !
        self.obs_env = self.env.reset()
        self.env_gym = self._aux_GymEnv_cls()(self.env)

    def test_assert_raises_creation(self):
        with self.assertRaises(RuntimeError):
            self.env_gym.observation_space = self._aux_BoxGymObsSpace_cls()(
                self.env_gym.observation_space
            )

    def test_can_create(self):
        kept_attr = [
            "gen_p",
            "load_p",
            "topo_vect",
            "rho",
            "actual_dispatch",
            "connectivity_matrix",
        ]
        self.env_gym.observation_space = self._aux_BoxGymObsSpace_cls()(
            self.env.observation_space,
            attr_to_keep=kept_attr,
            divide={
                "gen_p": self.env.gen_pmax,
                "load_p": self.obs_env.load_p,
                "actual_dispatch": self.env.gen_pmax,
            },
            functs={
                "connectivity_matrix": (
                    lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                    0.0,
                    1.0,
                    None,
                    None,
                )
            },
        )
        assert isinstance(self.env_gym.observation_space, self._aux_Box_cls())
        obs_gym, info = self.env_gym.reset()
        assert obs_gym in self.env_gym.observation_space
        assert self.env_gym.observation_space._attr_to_keep == sorted(kept_attr)
        assert len(obs_gym) == 3583

    def test_can_create_int(self):
        kept_attr = ["topo_vect", "line_status"]
        self.env_gym.observation_space = self._aux_BoxGymObsSpace_cls()(
            self.env.observation_space, attr_to_keep=kept_attr
        )
        obs_gym, info = self.env_gym.reset()
        assert obs_gym in self.env_gym.observation_space
        assert self.env_gym.observation_space._attr_to_keep == sorted(kept_attr)
        assert len(obs_gym) == 79
        assert obs_gym.dtype == dt_int

    def test_scaling(self):
        kept_attr = ["gen_p", "load_p"]
        # first test, with nothing
        observation_space = self._aux_BoxGymObsSpace_cls()(
            self.env.observation_space, attr_to_keep=kept_attr
        )
        self.env_gym.observation_space = observation_space
        obs_gym, info = self.env_gym.reset()
        assert obs_gym in observation_space
        assert observation_space._attr_to_keep == kept_attr
        assert len(obs_gym) == 17
        assert np.abs(obs_gym).max() >= 80

        # second test: just scaling (divide)
        observation_space = self._aux_BoxGymObsSpace_cls()(
            self.env.observation_space,
            attr_to_keep=kept_attr,
            divide={"gen_p": self.env.gen_pmax, "load_p": self.obs_env.load_p},
        )
        self.env_gym.observation_space = observation_space
        obs_gym, info = self.env_gym.reset()
        assert obs_gym in observation_space
        assert observation_space._attr_to_keep == kept_attr
        assert len(obs_gym) == 17
        assert np.abs(obs_gym).max() <= 2
        assert np.abs(obs_gym).max() >= 1.0

        # third step: center and reduce too
        observation_space = self._aux_BoxGymObsSpace_cls()(
            self.env.observation_space,
            attr_to_keep=kept_attr,
            divide={"gen_p": self.env.gen_pmax, "load_p": self.obs_env.load_p},
            subtract={"gen_p": 100.0, "load_p": 100.0},
        )
        self.env_gym.observation_space = observation_space
        obs_gym, info = self.env_gym.reset()
        assert obs_gym in observation_space
        assert observation_space._attr_to_keep == kept_attr
        assert len(obs_gym) == 17
        # the substract are calibrated so that the maximum is really close to 0
        assert obs_gym.max() <= 0
        assert obs_gym.max() >= -0.5

    def test_functs(self):
        """test the functs keyword argument"""
        # test i can make something with a funct keyword
        kept_attr = [
            "gen_p",
            "load_p",
            "topo_vect",
            "rho",
            "actual_dispatch",
            "connectivity_matrix",
        ]
        self.env_gym.observation_space = self._aux_BoxGymObsSpace_cls()(
            self.env.observation_space,
            attr_to_keep=kept_attr,
            divide={
                "gen_p": self.env.gen_pmax,
                "load_p": self.obs_env.load_p,
                "actual_dispatch": self.env.gen_pmax,
            },
            functs={
                "connectivity_matrix": (
                    lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                    0.0,
                    1.0,
                    None,
                    None,
                )
            },
        )
        obs_gym, info = self.env_gym.reset()
        assert obs_gym in self.env_gym.observation_space
        assert self.env_gym.observation_space._attr_to_keep == sorted(kept_attr)
        assert len(obs_gym) == 3583

        # test the stuff crashes if not used properly
        # bad shape provided
        with self.assertRaises(RuntimeError):
            tmp = self._aux_BoxGymObsSpace_cls()(
                self.env.observation_space,
                attr_to_keep=kept_attr,
                divide={
                    "gen_p": self.env.gen_pmax,
                    "load_p": self.obs_env.load_p,
                    "actual_dispatch": self.env.gen_pmax,
                },
                functs={
                    "connectivity_matrix": (
                        lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                        None,
                        None,
                        22,
                        None,
                    )
                },
            )
        # wrong input (tuple too short)
        with self.assertRaises(RuntimeError):
            tmp = self._aux_BoxGymObsSpace_cls()(
                self.env.observation_space,
                attr_to_keep=kept_attr,
                divide={
                    "gen_p": self.env.gen_pmax,
                    "load_p": self.obs_env.load_p,
                    "actual_dispatch": self.env.gen_pmax,
                },
                functs={
                    "connectivity_matrix": (
                        lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                        None,
                        None,
                        22,
                    )
                },
            )

        # function cannot be called
        with self.assertRaises(RuntimeError):
            tmp = self._aux_BoxGymObsSpace_cls()(
                self.env.observation_space,
                attr_to_keep=kept_attr,
                divide={
                    "gen_p": self.env.gen_pmax,
                    "load_p": self.obs_env.load_p,
                    "actual_dispatch": self.env.gen_pmax,
                },
                functs={
                    "connectivity_matrix": (
                        self.obs_env.connectivity_matrix().flatten(),
                        None,
                        None,
                        None,
                        None,
                    )
                },
            )

        # low not correct
        with self.assertRaises(RuntimeError):
            tmp = self._aux_BoxGymObsSpace_cls()(
                self.env.observation_space,
                attr_to_keep=kept_attr,
                divide={
                    "gen_p": self.env.gen_pmax,
                    "load_p": self.obs_env.load_p,
                    "actual_dispatch": self.env.gen_pmax,
                },
                functs={
                    "connectivity_matrix": (
                        lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                        0.5,
                        1.0,
                        None,
                        None,
                    )
                },
            )

        # high not correct
        with self.assertRaises(RuntimeError):
            tmp = self._aux_BoxGymObsSpace_cls()(
                self.env.observation_space,
                attr_to_keep=kept_attr,
                divide={
                    "gen_p": self.env.gen_pmax,
                    "load_p": self.obs_env.load_p,
                    "actual_dispatch": self.env.gen_pmax,
                },
                functs={
                    "connectivity_matrix": (
                        lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                        0.0,
                        0.9,
                        None,
                        None,
                    )
                },
            )

        # not added in attr_to_keep
        with self.assertRaises(RuntimeError):
            tmp = self._aux_BoxGymObsSpace_cls()(
                self.env.observation_space,
                attr_to_keep=["gen_p", "load_p", "topo_vect", "rho", "actual_dispatch"],
                divide={
                    "gen_p": self.env.gen_pmax,
                    "load_p": self.obs_env.load_p,
                    "actual_dispatch": self.env.gen_pmax,
                },
                functs={
                    "connectivity_matrix": (
                        lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                        0.0,
                        1.0,
                        None,
                        None,
                    )
                },
            )

        # another normal function
        self.env_gym.observation_space = self._aux_BoxGymObsSpace_cls()(
            self.env.observation_space,
            attr_to_keep=["connectivity_matrix", "log_load"],
            functs={
                "connectivity_matrix": (
                    lambda grid2opobs: grid2opobs.connectivity_matrix().flatten(),
                    0.0,
                    1.0,
                    None,
                    None,
                ),
                "log_load": (
                    lambda grid2opobs: np.log(grid2opobs.load_p + 1.0),
                    None,
                    10.0,
                    None,
                    None,
                ),
            },
        )


class _AuxTestBoxGymActSpace:
    def setUp(self) -> None:
        self._skip_if_no_gym()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage",
                test=True,
                action_class=PlayableAction,
                _add_to_name="TestBoxGymActSpace",
            )
        self.env.seed(0)
        self.env.reset()  # seed part !
        self.obs_env = self.env.reset()
        self.env_gym = self._aux_GymEnv_cls()(self.env)

    def test_assert_raises_creation(self):
        with self.assertRaises(RuntimeError):
            self.env_gym.action_space = self._aux_BoxGymActSpace_cls()(self.env_gym.action_space)

    def test_can_create(self):
        """test a simple creation"""
        kept_attr = ["set_bus", "change_bus", "redispatch"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_gym.action_space = self._aux_BoxGymActSpace_cls()(
                self.env.action_space, attr_to_keep=kept_attr
            )
        assert isinstance(self.env_gym.action_space, self._aux_Box_cls())
        self.env_gym.action_space.seed(0)
        grid2op_act = self.env_gym.action_space.from_gym(
            self.env_gym.action_space.sample()
        )
        assert isinstance(grid2op_act, PlayableAction)
        assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
        assert len(self.env_gym.action_space.sample()) == 121
        # check that all types
        ok_setbus = False
        ok_change_bus = False
        ok_redisp = False
        for _ in range(10):
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            ok_setbus = ok_setbus or np.any(grid2op_act.set_bus != 0)
            ok_change_bus = ok_change_bus or np.any(grid2op_act.change_bus)
            ok_redisp = ok_redisp or np.any(grid2op_act.redispatch != 0.0)
        if (not ok_setbus) or (not ok_change_bus) or (not ok_redisp):
            raise RuntimeError("Some property of the actions are not modified !")

    def test_all_attr_modified(self):
        """test all the attribute of the action can be modified"""
        all_attr = {
            "set_line_status": 20,
            "change_line_status": 20,
            "set_bus": 59,
            "change_bus": 59,
            "redispatch": 3,
            "set_storage": 2,
            "curtail": 3,
            "curtail_mw": 3,
        }
        func_check = {
            "set_line_status": lambda act: np.any(act.line_set_status != 0),
            "change_line_status": lambda act: np.any(act.line_change_status),
            "set_bus": lambda act: np.any(act.set_bus != 0.0),
            "change_bus": lambda act: np.any(act.change_bus),
            "redispatch": lambda act: np.any(act.redispatch != 0.0),
            "set_storage": lambda act: np.any(act.set_storage != 0.0),
            "curtail": lambda act: np.any(act.curtail != 1.0),
            "curtail_mw": lambda act: np.any(act.curtail != 1.0),
        }

        for attr_nm in sorted(all_attr.keys()):
            kept_attr = [attr_nm]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.env_gym.action_space = self._aux_BoxGymActSpace_cls()(
                    self.env.action_space, attr_to_keep=kept_attr
                )
            self.env_gym.action_space.seed(0)
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            assert isinstance(grid2op_act, PlayableAction)
            assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
            assert (
                len(self.env_gym.action_space.sample()) == all_attr[attr_nm]
            ), f"wrong size for {attr_nm}"
            self.env_gym.action_space.seed(0)
            # check that all types
            ok_ = func_check[attr_nm](grid2op_act)
            if not ok_:
                raise RuntimeError(
                    f"Some property of the actions are not modified for attr {attr_nm}"
                )

    def test_all_attr_modified_when_float(self):
        """test all the attribute of the action can be modified when the action is converted to a float"""
        redisp_size = 3
        all_attr = {
            "set_line_status": 20 + redisp_size,
            "change_line_status": 20 + redisp_size,
            "set_bus": 59 + redisp_size,
            "change_bus": 59 + redisp_size,
            "redispatch": redisp_size + redisp_size,
            "set_storage": 2 + redisp_size,
            "curtail": 3 + redisp_size,
            "curtail_mw": 3 + redisp_size,
        }
        func_check = {
            "set_line_status": lambda act: np.any(act.line_set_status != 0)
            and ~np.all(act.line_set_status != 0),
            "change_line_status": lambda act: np.any(act.line_change_status)
            and ~np.all(act.line_change_status),
            "set_bus": lambda act: np.any(act.set_bus != 0.0)
            and ~np.all(act.set_bus != 0.0),
            "change_bus": lambda act: np.any(act.change_bus)
            and ~np.all(act.change_bus),
            "redispatch": lambda act: np.any(act.redispatch != 0.0),
            "set_storage": lambda act: np.any(act.set_storage != 0.0),
            "curtail": lambda act: np.any(act.curtail != 1.0),
            "curtail_mw": lambda act: np.any(act.curtail != 1.0),
        }

        for attr_nm in sorted(all_attr.keys()):
            kept_attr = [attr_nm, "redispatch"]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.env_gym.action_space = self._aux_BoxGymActSpace_cls()(
                    self.env.action_space, attr_to_keep=kept_attr
                )
            self.env_gym.action_space.seed(0)
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            assert isinstance(grid2op_act, PlayableAction)
            assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
            assert (
                len(self.env_gym.action_space.sample()) == all_attr[attr_nm]
            ), f"wrong size for {attr_nm}"
            self.env_gym.action_space.seed(0)
            # check that all types
            ok_ = func_check[attr_nm](grid2op_act)
            if not ok_:
                raise RuntimeError(
                    f"Some property of the actions are not modified for attr {attr_nm}"
                )

    def test_curtailment_dispatch(self):
        """test curtail action will have no effect on non renewable, and dispatch action no effect
        on non dispatchable
        """
        kept_attr = ["curtail", "redispatch"]
        self.env_gym.action_space.close()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_gym.action_space = self._aux_BoxGymActSpace_cls()(
                self.env.action_space, attr_to_keep=kept_attr
            )
        self.env_gym.action_space.seed(0)
        grid2op_act = self.env_gym.action_space.from_gym(
            self.env_gym.action_space.sample()
        )
        assert isinstance(grid2op_act, PlayableAction)
        assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
        assert len(self.env_gym.action_space.sample()) == 6, "wrong size"
        self.env_gym.action_space.seed(0)
        for _ in range(10):
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            assert np.all(
                grid2op_act.redispatch[~grid2op_act.gen_redispatchable] == 0.0
            )
            assert np.all(grid2op_act.curtail[~grid2op_act.gen_renewable] == -1.0)

    def test_can_create_int(self):
        """test that if I use only discrete value, it gives me an array with discrete values"""
        kept_attr = ["change_line_status", "set_bus"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_gym.action_space = self._aux_BoxGymActSpace_cls()(
                self.env.action_space, attr_to_keep=kept_attr
            )
        self.env_gym.action_space.seed(0)
        act_gym = self.env_gym.action_space.sample()
        assert self.env_gym.action_space._attr_to_keep == kept_attr
        assert act_gym.dtype == dt_int
        assert len(act_gym) == 79

        kept_attr = ["change_line_status", "set_bus", "redispatch"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_gym.action_space = self._aux_BoxGymActSpace_cls()(
                self.env.action_space, attr_to_keep=kept_attr
            )
        self.env_gym.action_space.seed(0)
        act_gym = self.env_gym.action_space.sample()
        assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
        assert act_gym.dtype == dt_float
        assert len(act_gym) == 79 + 3

    def test_scaling(self):
        """test the add and multiply stuff"""
        kept_attr = ["redispatch"]
        # first test, with nothing
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_gym.action_space = self._aux_BoxGymActSpace_cls()(
                self.env.action_space, attr_to_keep=kept_attr
            )
        self.env_gym.action_space.seed(0)
        gen_redisp = self.env.gen_redispatchable
        act_gym = self.env_gym.action_space.sample()
        assert np.array_equal(
            self.env_gym.action_space.low, -self.env.gen_max_ramp_down[gen_redisp]
        )
        assert np.array_equal(
            self.env_gym.action_space.high, self.env.gen_max_ramp_up[gen_redisp]
        )
        assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
        assert len(act_gym) == 3
        assert np.any(act_gym >= 1.0)
        assert np.any(act_gym <= -1.0)
        grid2op_act = self.env_gym.action_space.from_gym(act_gym)
        assert not grid2op_act.is_ambiguous()[0]
        # second test: just scaling (divide)
        self.env_gym.action_space.close()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_gym.action_space = self._aux_BoxGymActSpace_cls()(
                self.env.action_space,
                attr_to_keep=kept_attr,
                multiply={"redispatch": self.env.gen_max_ramp_up[gen_redisp]},
            )
        self.env_gym.action_space.seed(0)
        assert np.array_equal(self.env_gym.action_space.low, -np.ones(3))
        assert np.array_equal(self.env_gym.action_space.high, np.ones(3))
        act_gym = self.env_gym.action_space.sample()
        assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
        assert len(act_gym) == 3
        assert np.all(act_gym <= 1.0)
        assert np.all(act_gym >= -1.0)
        grid2op_act2 = self.env_gym.action_space.from_gym(act_gym)
        assert not grid2op_act2.is_ambiguous()[0]
        assert np.all(np.isclose(grid2op_act.redispatch, grid2op_act2.redispatch))

        # third step: center and reduce too
        self.env_gym.action_space.close()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_gym.action_space = self._aux_BoxGymActSpace_cls()(
                self.env.action_space,
                attr_to_keep=kept_attr,
                multiply={"redispatch": self.env.gen_max_ramp_up[gen_redisp]},
                add={"redispatch": self.env.gen_max_ramp_up[gen_redisp]},
            )
        assert np.array_equal(self.env_gym.action_space.low, -np.ones(3) - 1.0)
        assert np.array_equal(self.env_gym.action_space.high, np.ones(3) - 1.0)
        self.env_gym.action_space.seed(0)
        act_gym = self.env_gym.action_space.sample()
        assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
        assert len(act_gym) == 3
        assert np.all(act_gym <= 0.0)
        assert np.all(act_gym >= -2.0)
        grid2op_act3 = self.env_gym.action_space.from_gym(act_gym)
        assert np.all(grid2op_act3.redispatch[~grid2op_act3.gen_redispatchable] == 0.0)
        assert not grid2op_act3.is_ambiguous()[0]
        assert np.all(np.isclose(grid2op_act.redispatch, grid2op_act3.redispatch))


class _AuxTestMultiDiscreteGymActSpace:
    def setUp(self) -> None:
        self._skip_if_no_gym()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage",
                test=True,
                action_class=PlayableAction,
                _add_to_name="TestMultiDiscreteGymActSpace",
            )
        self.env.seed(0)
        self.env.reset()  # seed part !
        self.obs_env = self.env.reset()
        self.env_gym = self._aux_GymEnv_cls()(self.env)

    def test_assert_raises_creation(self):
        with self.assertRaises(RuntimeError):
            self.env_gym.action_space = self._aux_MultiDiscreteActSpace_cls()(self.env_gym.action_space)

    def test_can_create(self):
        """test a simple creation"""
        kept_attr = ["set_bus", "change_bus", "redispatch"]
        del self.env_gym.action_space
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_gym.action_space = self._aux_MultiDiscreteActSpace_cls()(
                self.env.action_space, attr_to_keep=kept_attr
            )
        assert isinstance(self.env_gym.action_space, self._aux_MultiDiscrete_cls())
        
        self.env_gym.action_space.seed(0)
        gym_act = self.env_gym.action_space.sample()
        grid2op_act = self.env_gym.action_space.from_gym(gym_act)
        assert isinstance(grid2op_act, PlayableAction)
        assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
        assert len(self.env_gym.action_space.sample()) == 121
        # check that all types
        ok_setbus = False
        ok_change_bus = False
        ok_redisp = False
        for _ in range(10):
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            ok_setbus = ok_setbus or np.any(grid2op_act.set_bus != 0)
            ok_change_bus = ok_change_bus or np.any(grid2op_act.change_bus)
            ok_redisp = ok_redisp or np.any(grid2op_act.redispatch != 0.0)
        if (not ok_setbus) or (not ok_change_bus) or (not ok_redisp):
            raise RuntimeError("Some property of the actions are not modified !")

    def test_use_bins(self):
        """test the binarized version work"""
        kept_attr = ["set_bus", "change_bus", "redispatch"]
        for nb_bin in [3, 6, 9, 12]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.env_gym.action_space = self._aux_MultiDiscreteActSpace_cls()(
                    self.env.action_space,
                    attr_to_keep=kept_attr,
                    nb_bins={"redispatch": nb_bin},
                )
            self.env_gym.action_space.seed(0)
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            assert isinstance(grid2op_act, PlayableAction)
            assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
            assert len(self.env_gym.action_space.sample()) == 121
            assert np.all(
                self.env_gym.action_space.nvec[59:62] == [nb_bin, nb_bin, nb_bin]
            )
            ok_setbus = False
            ok_change_bus = False
            ok_redisp = False
            for _ in range(10):
                grid2op_act = self.env_gym.action_space.from_gym(
                    self.env_gym.action_space.sample()
                )
                ok_setbus = ok_setbus or np.any(grid2op_act.set_bus != 0)
                ok_change_bus = ok_change_bus or np.any(grid2op_act.change_bus)
                ok_redisp = ok_redisp or np.any(grid2op_act.redispatch != 0.0)
            if (not ok_setbus) or (not ok_change_bus) or (not ok_redisp):
                raise RuntimeError("Some property of the actions are not modified !")

    def test_use_substation(self):
        """test the keyword sub_set_bus, sub_change_bus"""
        kept_attr = ["sub_set_bus"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_gym.action_space = self._aux_MultiDiscreteActSpace_cls()(
                self.env.action_space, attr_to_keep=kept_attr
            )
        self.env_gym.action_space.seed(0)
        grid2op_act = self.env_gym.action_space.from_gym(
            self.env_gym.action_space.sample()
        )
        assert isinstance(grid2op_act, PlayableAction)
        assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
        assert len(self.env_gym.action_space.sample()) == 14
        assert np.all(
            self.env_gym.action_space.nvec
            == [4, 30, 6, 32, 16, 114, 5, 1, 16, 4, 4, 4, 8, 4]
        )
        # assert that i can "do nothing" in all substation
        for sub_id, li_act in enumerate(
            self.env_gym.action_space._sub_modifiers[kept_attr[0]]
        ):
            assert li_act[0] == self.env.action_space()
        ok_setbus = False
        for _ in range(10):
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            ok_setbus = ok_setbus or np.any(grid2op_act.set_bus != 0)
        if not ok_setbus:
            raise RuntimeError("Some property of the actions are not modified !")

        kept_attr = ["sub_change_bus"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_gym.action_space = self._aux_MultiDiscreteActSpace_cls()(
                self.env.action_space, attr_to_keep=kept_attr
            )
        self.env_gym.action_space.seed(0)
        grid2op_act = self.env_gym.action_space.from_gym(
            self.env_gym.action_space.sample()
        )
        assert isinstance(grid2op_act, PlayableAction)
        assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
        assert len(self.env_gym.action_space.sample()) == 14
        assert np.all(
            self.env_gym.action_space.nvec
            == [4, 32, 8, 32, 16, 128, 4, 4, 16, 4, 4, 4, 8, 4]
        )
        # assert that i can "do nothing" in all substation
        for sub_id, li_act in enumerate(
            self.env_gym.action_space._sub_modifiers[kept_attr[0]]
        ):
            assert li_act[0] == self.env.action_space()
        ok_changebus = False
        for _ in range(10):
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            ok_changebus = ok_changebus or np.any(grid2op_act.change_bus)
        if not ok_changebus:
            raise RuntimeError("Some property of the actions are not modified !")

    def test_supported_keys(self):
        """test that i can modify every action with the keys"""
        dims = {
            "set_line_status": 20,
            "change_line_status": 20,
            "set_bus": 59,
            "change_bus": 59,
            "sub_set_bus": 14,
            "sub_change_bus": 14,
            "one_sub_set": 1,
            "one_sub_change": 1,
            "redispatch": 3,
            "curtail": 3,
            "curtail_mw": 3,
            "set_storage": 2,
        }

        func_check = {
            "set_line_status": lambda act: np.any(act.line_set_status != 0)
            and ~np.all(act.line_set_status != 0),
            "change_line_status": lambda act: np.any(act.line_change_status)
            and ~np.all(act.line_change_status),
            "set_bus": lambda act: np.any(act.set_bus != 0.0)
            and ~np.all(act.set_bus != 0.0),
            "change_bus": lambda act: np.any(act.change_bus)
            and ~np.all(act.change_bus),
            "redispatch": lambda act: np.any(act.redispatch != 0.0),
            "set_storage": lambda act: np.any(act.set_storage != 0.0),
            "curtail": lambda act: np.any(act.curtail != 1.0),
            "curtail_mw": lambda act: np.any(act.curtail != 1.0),
            "sub_change_bus": lambda act: np.any(act.change_bus)
            and ~np.all(act.change_bus),
            "sub_set_bus": lambda act: np.any(act.set_bus != 0.0)
            and ~np.all(act.set_bus != 0.0),
            "one_sub_set": lambda act: np.any(act.set_bus != 0.0)
            and ~np.all(act.set_bus != 0.0),
            "one_sub_change": lambda act: np.any(act.change_bus)
            and ~np.all(act.change_bus),
        }

        for attr_nm in sorted(dims.keys()):
            kept_attr = [attr_nm]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.env_gym.action_space = self._aux_MultiDiscreteActSpace_cls()(
                    self.env.action_space, attr_to_keep=kept_attr
                )
            assert self.env_gym.action_space._attr_to_keep == kept_attr
            assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
            assert (
                len(self.env_gym.action_space.sample()) == dims[attr_nm]
            ), f"wrong size for {attr_nm}"
            self.env_gym.action_space.seed(0)
            assert (
                len(self.env_gym.action_space.sample()) == dims[attr_nm]
            ), f"wrong size for {attr_nm}"
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            assert isinstance(grid2op_act, PlayableAction)
            # check that all types
            ok_ = func_check[attr_nm](grid2op_act)
            if not ok_:
                raise RuntimeError(
                    f"Some property of the actions are not modified for attr {attr_nm}"
                )


class _AuxTestDiscreteGymActSpace:
    def setUp(self) -> None:
        self._skip_if_no_gym()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage",
                test=True,
                action_class=PlayableAction,
                _add_to_name="TestMultiDiscreteGymActSpace",
            )
        self.env.seed(0)
        self.env.reset()  # seed part !
        self.obs_env = self.env.reset()
        self.env_gym = self._aux_GymEnv_cls()(self.env)

    def test_assert_raises_creation(self):
        with self.assertRaises(RuntimeError):
            self.env_gym.action_space = self._aux_DiscreteActSpace_cls()(self.env_gym.action_space)

    def test_can_create(self):
        """test a simple creation"""
        kept_attr = ["set_bus", "change_bus", "redispatch"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_gym.action_space = self._aux_DiscreteActSpace_cls()(
                self.env.action_space, attr_to_keep=kept_attr
            )
        assert isinstance(self.env_gym.action_space, self._aux_Discrete_cls())
        
        self.env_gym.action_space.seed(0)
        grid2op_act = self.env_gym.action_space.from_gym(
            self.env_gym.action_space.sample()
        )
        assert isinstance(grid2op_act, PlayableAction)
        assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
        act = self.env_gym.action_space.sample()
        assert isinstance(act, (int, np.int32, np.int64, dt_int)), f"{act} not an int but {type(act)}"
        assert self.env_gym.action_space.n == 525

        # check that all types
        ok_setbus = False
        ok_change_bus = False
        ok_redisp = False
        for _ in range(30):
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            ok_setbus = ok_setbus or np.any(grid2op_act.set_bus != 0)
            ok_change_bus = ok_change_bus or np.any(grid2op_act.change_bus)
            ok_redisp = ok_redisp or np.any(grid2op_act.redispatch != 0.0)
        if (not ok_setbus) or (not ok_change_bus) or (not ok_redisp):
            raise RuntimeError("Some property of the actions are not modified !")

    def test_use_bins(self):
        """test the binarized version work"""
        kept_attr = ["set_bus", "change_bus", "redispatch"]
        for nb_bin in [3, 6, 9, 12]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.env_gym.action_space = self._aux_DiscreteActSpace_cls()(
                    self.env.action_space,
                    attr_to_keep=kept_attr,
                    nb_bins={"redispatch": nb_bin},
                )
            self.env_gym.action_space.seed(0)
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            assert isinstance(grid2op_act, PlayableAction)
            assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
            assert self.env_gym.action_space.n == 525 - (7 - nb_bin) * 3 * 2

    def test_supported_keys(self):
        """test that i can modify every action with the keys"""
        dims = {
            "set_line_status": 101,
            "change_line_status": 21,
            "set_bus": 235,
            "change_bus": 255,
            "redispatch": 37,
            "curtail": 22,
            "curtail_mw": 31,
            "set_storage": 25,
        }

        func_check = {
            "set_line_status": lambda act: np.any(act.line_set_status != 0)
            and ~np.all(act.line_set_status != 0),
            "change_line_status": lambda act: np.any(act.line_change_status)
            and ~np.all(act.line_change_status),
            "set_bus": lambda act: np.any(act.set_bus != 0.0)
            and ~np.all(act.set_bus != 0.0),
            "change_bus": lambda act: np.any(act.change_bus)
            and ~np.all(act.change_bus),
            "redispatch": lambda act: np.any(act.redispatch != 0.0),
            "set_storage": lambda act: np.any(act.set_storage != 0.0),
            "curtail": lambda act: np.any(act.curtail != 1.0),
            "curtail_mw": lambda act: np.any(act.curtail != 1.0),
        }
        for attr_nm in sorted(dims.keys()):
            kept_attr = [attr_nm]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.env_gym.action_space = self._aux_DiscreteActSpace_cls()(
                    self.env.action_space, attr_to_keep=kept_attr
                )
            assert self.env_gym.action_space._attr_to_keep == sorted(kept_attr)
            assert (
                self.env_gym.action_space.n == dims[attr_nm]
            ), f"wrong size for {attr_nm}"
            self.env_gym.action_space.seed(0)
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            assert isinstance(grid2op_act, PlayableAction)
            # check that all types
            ok_ = func_check[attr_nm](grid2op_act)
            if not ok_:
                pdb.set_trace()
                raise RuntimeError(
                    f"Some property of the actions are not modified for attr {attr_nm}"
                )


class _AuxTestAllGymActSpaceWithAlarm:
    def setUp(self) -> None:
        self._skip_if_no_gym()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                os.path.join(PATH_DATA_TEST, "l2rpn_neurips_2020_track1_with_alarm"),
                test=True,
                action_class=PlayableAction,
                _add_to_name="TestAllGymActSpaceWithAlarm",
            )
        self.env.seed(0)
        self.env.reset()  # seed part !
        self.obs_env = self.env.reset()
        self.env_gym = self._aux_GymEnv_cls()(self.env)

    def test_supported_keys_box(self):
        """test all the attribute of the action can be modified when the action is converted to a float"""
        all_attr = {
            "set_line_status": 59,
            "change_line_status": 59,
            "set_bus": 177,
            "change_bus": 177,
            "redispatch": np.sum(self.env.gen_redispatchable),
            "set_storage": 0,
            "curtail": np.sum(self.env.gen_renewable),
            "curtail_mw": np.sum(self.env.gen_renewable),
            "raise_alarm": 3,
        }
        func_check = {
            "set_line_status": lambda act: np.any(act.line_set_status != 0)
            and ~np.all(act.line_set_status != 0),
            "change_line_status": lambda act: np.any(act.line_change_status)
            and ~np.all(act.line_change_status),
            "set_bus": lambda act: np.any(act.set_bus != 0.0)
            and ~np.all(act.set_bus != 0.0),
            "change_bus": lambda act: np.any(act.change_bus)
            and ~np.all(act.change_bus),
            "redispatch": lambda act: np.any(act.redispatch != 0.0),
            "set_storage": lambda act: np.any(act.set_storage != 0.0),
            "curtail": lambda act: np.any(act.curtail != 1.0),
            "curtail_mw": lambda act: np.any(act.curtail != 1.0),
            "raise_alarm": lambda act: np.any(act.raise_alarm)
            and ~np.all(act.raise_alarm),
        }

        for attr_nm in sorted(all_attr.keys()):
            kept_attr = [attr_nm]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.env_gym.action_space = self._aux_BoxGymActSpace_cls()(
                    self.env.action_space, attr_to_keep=kept_attr
                )
            self.env_gym.action_space.seed(0)
            gym_act = self.env_gym.action_space.sample()
            grid2op_act = self.env_gym.action_space.from_gym(gym_act)
            assert isinstance(grid2op_act, PlayableAction)
            assert self.env_gym.action_space._attr_to_keep == kept_attr
            assert (
                len(self.env_gym.action_space.sample()) == all_attr[attr_nm]
            ), f"wrong size for {attr_nm}"
            # check that all types
            ok_ = func_check[attr_nm](grid2op_act)
            if not ok_ and attr_nm != "set_storage":
                # NB for "set_storage" as there are no storage unit on this grid, then this test is doomed to fail
                # this is why i don't perform it in this case
                raise RuntimeError(
                    f"Some property of the actions are not modified for attr {attr_nm}"
                )

    def test_supported_keys_multidiscrete(self):
        """test that i can modify every action with the keys"""
        dims = {
            "set_line_status": 59,
            "change_line_status": 59,
            "set_bus": 177,
            "change_bus": 177,
            "redispatch": np.sum(self.env.gen_redispatchable),
            "curtail": np.sum(self.env.gen_renewable),
            "curtail_mw": np.sum(self.env.gen_renewable),
            "set_storage": 0,
            "raise_alarm": 3,
        }

        func_check = {
            "set_line_status": lambda act: np.any(act.line_set_status != 0)
            and ~np.all(act.line_set_status != 0),
            "change_line_status": lambda act: np.any(act.line_change_status)
            and ~np.all(act.line_change_status),
            "set_bus": lambda act: np.any(act.set_bus != 0.0)
            and ~np.all(act.set_bus != 0.0),
            "change_bus": lambda act: np.any(act.change_bus)
            and ~np.all(act.change_bus),
            "redispatch": lambda act: np.any(act.redispatch != 0.0),
            "set_storage": lambda act: np.any(act.set_storage != 0.0),
            "curtail": lambda act: np.any(act.curtail != -1.0),
            "curtail_mw": lambda act: np.any(act.curtail != -1.0),
            "raise_alarm": lambda act: np.any(act.raise_alarm)
            and ~np.all(act.raise_alarm),
        }

        for attr_nm in sorted(dims.keys()):
            kept_attr = [attr_nm]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.env_gym.action_space = self._aux_MultiDiscreteActSpace_cls()(
                    self.env.action_space, attr_to_keep=kept_attr
                )
            assert self.env_gym.action_space._attr_to_keep == kept_attr
            self.env_gym.action_space.seed(0)
            assert (
                len(self.env_gym.action_space.sample()) == dims[attr_nm]
            ), f"wrong size for {attr_nm}"
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            assert isinstance(grid2op_act, PlayableAction)
            # check that all types
            ok_ = func_check[attr_nm](grid2op_act)
            if not ok_ and attr_nm != "set_storage":
                # NB for "set_storage" as there are no storage unit on this grid, then this test is doomed to fail
                # this is why i don't perform it in this case
                raise RuntimeError(
                    f"Some property of the actions are not modified for attr {attr_nm}"
                )

    def test_supported_keys_discrete(self):
        """test that i can modify every action with the keys"""
        dims = {
            "set_line_status": 5 * 59 + 1,
            "change_line_status": 59 + 1,
            # "set_bus": 5*177,  # already tested on the case 14 and takes a lot to compute !
            # "change_bus": 255,  # already tested on the case 14 and takes a lot to compute !
            "redispatch": 121,
            "curtail": 85,
            "curtail_mw": 121,
            "set_storage": 1,
            "raise_alarm": 4,
        }

        func_check = {
            "set_line_status": lambda act: np.any(act.line_set_status != 0)
            and ~np.all(act.line_set_status != 0),
            "change_line_status": lambda act: np.any(act.line_change_status)
            and ~np.all(act.line_change_status),
            "set_bus": lambda act: np.any(act.set_bus != 0.0)
            and ~np.all(act.set_bus != 0.0),
            "change_bus": lambda act: np.any(act.change_bus)
            and ~np.all(act.change_bus),
            "redispatch": lambda act: np.any(act.redispatch != 0.0),
            "set_storage": lambda act: np.any(act.set_storage != 0.0),
            "curtail": lambda act: np.any(act.curtail != 1.0),
            "curtail_mw": lambda act: np.any(act.curtail != 1.0),
            "raise_alarm": lambda act: np.any(act.raise_alarm)
            and ~np.all(act.raise_alarm),
        }
        for attr_nm in sorted(dims.keys()):
            kept_attr = [attr_nm]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.env_gym.action_space = self._aux_DiscreteActSpace_cls()(
                    self.env.action_space, attr_to_keep=kept_attr
                )
            assert self.env_gym.action_space._attr_to_keep == kept_attr
            assert self.env_gym.action_space.n == dims[attr_nm], (
                f"wrong size for {attr_nm}, should be {dims[attr_nm]} "
                f"but is {self.env_gym.action_space.n}"
            )
            self.env_gym.action_space.seed(1)  # with seed 0 it does not work
            act_gym = self.env_gym.action_space.sample()
            grid2op_act = self.env_gym.action_space.from_gym(act_gym)
            assert isinstance(grid2op_act, PlayableAction)
            # check that all types
            ok_ = func_check[attr_nm](grid2op_act)
            if not ok_ and attr_nm != "set_storage":
                # NB for "set_storage" as there are no storage unit on this grid, then this test is doomed to fail
                # this is why i don't perform it in this case
                raise RuntimeError(
                    f"Some property of the actions are not modified for attr {attr_nm}"
                )


class _AuxTestGOObsInRange:
    def setUp(self) -> None:
        self._skip_if_no_gym()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage",
                test=True,
                action_class=PlayableAction,
                _add_to_name="TestMultiDiscreteGymActSpace",
            )
        self.env.seed(0)
        self.env.reset()  # seed part !
        self.obs_env = self.env.reset()
        self.env_gym = self._aux_GymEnv_cls()(self.env)

    def test_obs_in_go_state_dont_exceed_max(self):
        obs, reward, done, info = self.env.step(
            self.env.action_space({"set_bus": {"generators_id": [(0, -1)]}})
        )
        assert done
        gym_obs = self.env_gym.observation_space.to_gym(obs)
        for key in self.env_gym.observation_space.spaces.keys():
            assert key in gym_obs, f"key: {key} no in the observation"
        for key in gym_obs.keys():
            assert gym_obs[key] in self.env_gym.observation_space.spaces[key], f"error for {key}"
            

class _AuxObsAllAttr:
    def test_all_attr_in_obs(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True,
                               action_class=PlayableAction)
        gym_env = self._aux_GymEnv_cls()(env)
        obs, info = gym_env.reset()
        all_attrs = ["year",
                     "month",
                     "day",
                     "hour_of_day",
                     "minute_of_hour",
                     "day_of_week",
                     "timestep_overflow",
                     "line_status",
                     "topo_vect",
                     "gen_p",
                     "gen_q",
                     "gen_v",
                     "gen_margin_up",
                     "gen_margin_down",
                     "load_p",
                     "load_q",
                     "load_v",
                     "p_or",
                     "q_or",
                     "v_or",
                     "a_or",
                     "p_ex",
                     "q_ex",
                     "v_ex",
                     "a_ex",
                     "rho",
                     "time_before_cooldown_line",
                     "time_before_cooldown_sub",
                     "time_next_maintenance",
                     "duration_next_maintenance",
                     "target_dispatch",
                     "actual_dispatch",
                     "storage_charge",
                     "storage_power_target",
                     "storage_power",
                     "is_alarm_illegal",
                     "time_since_last_alarm",
                    #  "last_alarm",
                    #  "attention_budget",
                    #  "was_alarm_used_after_game_over",
                     "_shunt_p",
                     "_shunt_q",
                     "_shunt_v",
                     "_shunt_bus",
                     "thermal_limit",
                     "gen_p_before_curtail",
                     "curtailment",
                     "curtailment_limit",
                     "curtailment_limit_effective",
                     "theta_or",
                     "theta_ex",
                     "load_theta",
                     "gen_theta",
                     "storage_theta",
                     "current_step",
                     "max_step",
                     "delta_time"]
        for el in all_attrs:
            assert el in obs.keys(), f"\"{el}\" not in obs.keys()"

