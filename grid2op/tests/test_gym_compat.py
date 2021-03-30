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
import grid2op
from grid2op.dtypes import dt_float, dt_bool, dt_int
from grid2op.tests.helper_path_test import *
from grid2op.MakeEnv import make
from grid2op.Converter import IdToAct, ToVect
from grid2op.Action import PlayableAction

try:
    import gym
    from gym.spaces import Box
    from grid2op.gym_compat import GymActionSpace, GymObservationSpace
    from grid2op.gym_compat import GymEnv
    from grid2op.gym_compat import ContinuousToDiscreteConverter
    from grid2op.gym_compat import ScalerAttrConverter
    from grid2op.gym_compat import MultiToTupleConverter
    from grid2op.gym_compat import BoxGymObsSpace
    GYM_AVAIL = True
except ImportError:
    GYM_AVAIL = False

import pdb

import warnings
warnings.simplefilter("error")


class TestGymCompatModule(unittest.TestCase):
    def _skip_if_no_gym(self):
        if not GYM_AVAIL:
            self.skipTest("Gym is not available")

    def setUp(self) -> None:
        self._skip_if_no_gym()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox",
                                    test=True,
                                    _add_to_name="TestGymCompatModule")

    def tearDown(self) -> None:
        self.env.close()

    def test_convert_togym(self):
        """test i can create the env"""
        env_gym = GymEnv(self.env)
        dim_act_space = np.sum([np.sum(env_gym.action_space[el].shape) for el in env_gym.action_space.spaces])
        assert dim_act_space == 160
        dim_obs_space = np.sum([np.sum(env_gym.observation_space[el].shape).astype(int)
                                for el in env_gym.observation_space.spaces])
        assert dim_obs_space == 432

        # test that i can do basic stuff there
        obs = env_gym.reset()
        for k in env_gym.observation_space.spaces.keys():
            assert obs[k] in env_gym.observation_space[k], f"error for {k}"
        act = env_gym.action_space.sample()
        obs2, reward2, done2, info2 = env_gym.step(act)
        assert obs2 in env_gym.observation_space

        # test for the __str__ method
        str_ = self.env.action_space.__str__()
        str_ = self.env.observation_space.__str__()

    def test_ignore(self):
        """test the ignore_attr method"""
        env_gym = GymEnv(self.env)
        env_gym.action_space = env_gym.action_space.ignore_attr("set_bus").ignore_attr("set_line_status")
        dim_act_space = np.sum([np.sum(env_gym.action_space[el].shape) for el in env_gym.action_space.spaces])
        assert dim_act_space == 83

    def test_keep_only(self):
        """test the keep_only_attr method"""
        env_gym = GymEnv(self.env)
        env_gym.observation_space = env_gym.observation_space.keep_only_attr(["rho", "gen_p", "load_p",
                                                                              "topo_vect",
                                                                              "actual_dispatch"])
        new_dim_obs_space = np.sum([np.sum(env_gym.observation_space[el].shape).astype(int)
                                    for el in env_gym.observation_space.spaces])
        assert new_dim_obs_space == 100

    def test_scale_attr_converter(self):
        """test a scale_attr converter"""
        env_gym = GymEnv(self.env)
        ob_space = env_gym.observation_space
        ob_space = ob_space.reencode_space("actual_dispatch",
                                           ScalerAttrConverter(substract=0.,
                                                               divide=self.env.gen_pmax
                                                               )
                                           )
        env_gym.observation_space = ob_space
        obs = env_gym.reset()
        key = "actual_dispatch"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(self.env.n_gen) - 1
        high = np.zeros(self.env.n_gen) + 1
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"
        assert obs in env_gym.observation_space

    def test_add_key(self):
        """test the add_key feature"""
        env_gym = GymEnv(self.env)
        shape_ = (self.env.dim_topo, self.env.dim_topo)
        key = "connectivity_matrix"
        env_gym.observation_space.add_key(key,
                                          lambda obs: obs.connectivity_matrix(),
                                          Box(shape=shape_,
                                              low=np.zeros(shape_, dtype=dt_float),
                                              high=np.ones(shape_, dtype=dt_float),
                                              dtype=dt_float
                                              )
                                          )

        # we highly recommend to "reset" the environment after setting up the observation space
        obs_gym = env_gym.reset()
        assert key in env_gym.observation_space.spaces
        assert obs_gym in env_gym.observation_space

    def test_chain_converter(self):
        """test i can do two converters on the same key"""
        env_gym = GymEnv(self.env)
        env_gym.action_space = env_gym.action_space.reencode_space("redispatch",
                                                                   ContinuousToDiscreteConverter(nb_bins=11)
                                                                   )
        env_gym.action_space.seed(0)
        act_gym = env_gym.action_space.sample()
        assert np.all(act_gym["redispatch"] == (0, 10, 0, 0, 0, 7))
        act_gym = env_gym.action_space.sample()
        assert np.all(act_gym["redispatch"] == (4, 7, 0, 0, 0, 10))
        assert isinstance(env_gym.action_space["redispatch"], gym.spaces.MultiDiscrete)
        env_gym.action_space = env_gym.action_space.reencode_space("redispatch", MultiToTupleConverter())
        assert isinstance(env_gym.action_space["redispatch"], gym.spaces.Tuple)

        # and now test that the redispatching is properly computed
        # env_gym.action_space.seed(0)
        # TODO this doesn't work... because when you seed it appears to use the same seed on all
        # on all the "sub part" of the Tuple.. THanks gym !
        # see https://github.com/openai/gym/issues/2166
        # act_gym = env_gym.action_space.sample()
        # assert act_gym["redispatch"] == (0, 10, 0, 0, 0, 7)
        # act_glop = env_gym.action_space.from_gym(act_gym)
        # assert np.array_equal(act_glop._redispatch,
        #                       np.array([-4.1666665, -8.333333, 0., 0., 0., -12.5], dtype=dt_float)
        #                       )
        # act_gym = env_gym.action_space.sample()
        # assert act_gym["redispatch"] == (4, 7, 0, 0, 0, 10)
        # act_glop = env_gym.action_space.from_gym(act_gym)
        # assert np.array_equal(act_glop._redispatch,
        #                       np.array([0.833333, 1.666666, 0., 0., 0., 2.5], dtype=dt_float)
        #                       )

    def test_all_together(self):
        """combine all test above (for the action space)"""
        env_gym = GymEnv(self.env)
        env_gym.action_space = env_gym.action_space.ignore_attr("set_bus").ignore_attr("set_line_status")
        env_gym.action_space = env_gym.action_space.reencode_space("redispatch",
                                                                   ContinuousToDiscreteConverter(nb_bins=11)
                                                                   )
        env_gym.action_space = env_gym.action_space.reencode_space("change_bus", MultiToTupleConverter())
        env_gym.action_space = env_gym.action_space.reencode_space("change_line_status",
                                                                   MultiToTupleConverter())
        env_gym.action_space = env_gym.action_space.reencode_space("redispatch", MultiToTupleConverter())

        assert isinstance(env_gym.action_space["redispatch"], gym.spaces.Tuple)
        assert isinstance(env_gym.action_space["change_bus"], gym.spaces.Tuple)
        assert isinstance(env_gym.action_space["change_line_status"], gym.spaces.Tuple)

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
            env = grid2op.make("educ_case14_storage",
                               test=True,
                               _add_to_name="TestGymCompatModule")
        env_gym = GymEnv(env)
        assert "a_ex" in env_gym.observation_space.spaces
        assert np.array_equal(env_gym.observation_space["a_ex"].low, np.zeros(shape=(env.n_line, ), ))
        assert "a_or" in env_gym.observation_space.spaces
        assert np.array_equal(env_gym.observation_space["a_or"].low, np.zeros(shape=(env.n_line, ), ))

        key = "actual_dispatch"
        assert key in env_gym.observation_space.spaces
        low = np.minimum(env.gen_pmin,
                         -env.gen_pmax)
        high = np.maximum(-env.gen_pmin,
                          +env.gen_pmax)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "curtailment"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(shape=(env.n_gen,))
        high = np.ones(shape=(env.n_gen,))
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "curtailment_limit"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(shape=(env.n_gen,))
        high = np.ones(shape=(env.n_gen,))
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

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
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "gen_p"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(shape=(env.n_gen,), dtype=dt_int)
        high = env.gen_pmax * 1.2  # weird hey ? But expected because of slack bus
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "gen_p_before_curtail"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(shape=(env.n_gen,), dtype=dt_int)
        high = env.gen_pmax * 1.2  # weird hey ? But expected because of slack bus
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "gen_q"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_gen,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_gen,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "gen_v"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(shape=(env.n_gen,), dtype=dt_int)
        high = np.full(shape=(env.n_gen,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "load_p"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_load,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_load,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "load_q"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_load,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_load,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "load_v"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_load,), fill_value=0., dtype=dt_float)
        high = np.full(shape=(env.n_load,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "p_ex"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "p_or"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "q_ex"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "q_or"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=-np.inf, dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "rho"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=0., dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "storage_charge"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_storage,), fill_value=0., dtype=dt_float)
        high = env.storage_Emax
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "storage_power"
        assert key in env_gym.observation_space.spaces
        low = -env.storage_max_p_absorb
        high = env.storage_max_p_prod
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "storage_power_target"
        assert key in env_gym.observation_space.spaces
        low = -env.storage_max_p_absorb
        high = env.storage_max_p_prod
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "target_dispatch"
        assert key in env_gym.observation_space.spaces
        low = -env.gen_pmax
        high = env.gen_pmax
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "time_before_cooldown_line"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(env.n_line, dtype=dt_int)
        high = np.zeros(env.n_line, dtype=dt_int) + max(env.parameters.NB_TIMESTEP_RECONNECTION,
                                                        env.parameters.NB_TIMESTEP_COOLDOWN_LINE,
                                                        env._oppSpace.attack_duration)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "time_before_cooldown_sub"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(env.n_sub, dtype=dt_int)
        high = np.zeros(env.n_sub, dtype=dt_int) + env.parameters.NB_TIMESTEP_COOLDOWN_SUB
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "time_next_maintenance"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(env.n_line, dtype=dt_int) - 1
        high = np.full(env.n_line, fill_value=np.inf, dtype=dt_int) - 1
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "timestep_overflow"
        assert key in env_gym.observation_space.spaces
        low = np.full(env.n_line, fill_value=np.inf, dtype=dt_int)
        high = np.full(env.n_line, fill_value=np.inf, dtype=dt_int) - 1
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "topo_vect"
        assert key in env_gym.observation_space.spaces
        low = np.zeros(env.dim_topo, dtype=dt_int) - 1
        high = np.zeros(env.dim_topo, dtype=dt_int) + 2
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "v_or"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=0., dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"

        key = "v_ex"
        assert key in env_gym.observation_space.spaces
        low = np.full(shape=(env.n_line,), fill_value=0., dtype=dt_float)
        high = np.full(shape=(env.n_line,), fill_value=np.inf, dtype=dt_float)
        assert np.array_equal(env_gym.observation_space[key].low, low), f"issue for {key}"
        assert np.array_equal(env_gym.observation_space[key].high, high), f"issue for {key}"


class TestBoxGymObsSpace(unittest.TestCase):
    def _skip_if_no_gym(self):
        if not GYM_AVAIL:
            self.skipTest("Gym is not available")

    def setUp(self) -> None:
        self._skip_if_no_gym()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    action_class=PlayableAction,
                                    _add_to_name="TestBoxGymObsSpace")
        self.obs_env = self.env.reset()
        self.env_gym = GymEnv(self.env)

    def test_assert_raises_creation(self):
        with self.assertRaises(RuntimeError):
             self.env_gym.observation_space = BoxGymObsSpace(self.env_gym.observation_space)
    
    def test_can_create(self):
        kept_attr = ["gen_p", "load_p", "topo_vect", "rho", "actual_dispatch", "connectivity_matrix"]
        self.env_gym.observation_space = BoxGymObsSpace(self.env.observation_space,
                                                        attr_to_keep=kept_attr,
                                                divide={"gen_p": self.env.gen_pmax,
                                                        "load_p": self.obs_env.load_p,
                                                        "actual_dispatch": self.env.gen_pmax},
                                                functs={"connectivity_matrix": (
                                                            lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                                                            0., 1., None, None,
                                                            )
                                                        }
                                                )
        obs_gym =  self.env_gym.reset()
        assert obs_gym in self.env_gym.observation_space
        assert self.env_gym.observation_space._attr_to_keep == kept_attr
        assert len(obs_gym) == 3583

    def test_can_create_int(self):
        kept_attr = [ "topo_vect", "line_status"]
        self.env_gym.observation_space = BoxGymObsSpace(self.env.observation_space,
                                                        attr_to_keep=kept_attr
                                                        )
        obs_gym =  self.env_gym.reset()
        assert obs_gym in self.env_gym.observation_space
        assert self.env_gym.observation_space._attr_to_keep == kept_attr
        assert len(obs_gym) == 79   
        assert obs_gym.dtype == dt_int

    def test_scaling(self):
        kept_attr = ["gen_p", "load_p"]
        # first test, with nothing
        observation_space = BoxGymObsSpace(self.env.observation_space,
                                           attr_to_keep=kept_attr)
        self.env_gym.observation_space = observation_space
        obs_gym =  self.env_gym.reset()
        assert obs_gym in observation_space
        assert observation_space._attr_to_keep == kept_attr
        assert len(obs_gym) == 17
        assert np.abs(obs_gym).max() >= 80

        # second test: just scaling (divide)
        observation_space = BoxGymObsSpace(self.env.observation_space,
                                           attr_to_keep=kept_attr,
                                           divide={"gen_p": self.env.gen_pmax,
                                                  "load_p": self.obs_env.load_p},
                                           )
        self.env_gym.observation_space = observation_space
        obs_gym = self.env_gym.reset()
        assert obs_gym in observation_space
        assert observation_space._attr_to_keep == kept_attr
        assert len(obs_gym) == 17
        assert np.abs(obs_gym).max() <= 2
        assert np.abs(obs_gym).max() >= 1.

        # third step: center and reduce too
        observation_space = BoxGymObsSpace(self.env.observation_space,
                                    attr_to_keep=kept_attr,
                                    divide={"gen_p": self.env.gen_pmax,
                                            "load_p": self.obs_env.load_p},
                                    substract={"gen_p": 90.,
                                                "load_p": 100.},
                                    )
        self.env_gym.observation_space = observation_space
        obs_gym =  self.env_gym.reset()
        assert obs_gym in observation_space
        assert observation_space._attr_to_keep == kept_attr
        assert len(obs_gym) == 17
        # the substract are calibrated so that the maximum is really close to 0
        assert obs_gym.max() <= 0  
        assert obs_gym.max() >= -0.5

    def test_functs(self):
        """test the functs keyword argument"""
        # test i can make something with a funct keyword
        kept_attr = ["gen_p", "load_p", "topo_vect", "rho", "actual_dispatch", "connectivity_matrix"]
        self.env_gym.observation_space = BoxGymObsSpace(self.env.observation_space,
                                                        attr_to_keep=kept_attr,
                                                divide={"gen_p": self.env.gen_pmax,
                                                        "load_p": self.obs_env.load_p,
                                                        "actual_dispatch": self.env.gen_pmax},
                                                functs={"connectivity_matrix": (
                                                            lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                                                            0., 1., None, None,
                                                            )
                                                        }
                                                )
        obs_gym = self.env_gym.reset()
        assert obs_gym in self.env_gym.observation_space
        assert self.env_gym.observation_space._attr_to_keep == kept_attr
        assert len(obs_gym) == 3583

        # test the stuff crashes if not used properly
        # bad shape provided
        with self.assertRaises(RuntimeError):
            tmp = BoxGymObsSpace(self.env.observation_space,
                                 attr_to_keep=kept_attr,
                                 divide={"gen_p": self.env.gen_pmax,
                                         "load_p": self.obs_env.load_p,
                                         "actual_dispatch": self.env.gen_pmax},
                                 functs={"connectivity_matrix": (
                                         lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                                          None, None, 22, None)
                                        }
                                )
        # wrong input (tuple too short)
        with self.assertRaises(RuntimeError):
            tmp = BoxGymObsSpace(self.env.observation_space,
                                 attr_to_keep=kept_attr,
                                 divide={"gen_p": self.env.gen_pmax,
                                         "load_p": self.obs_env.load_p,
                                         "actual_dispatch": self.env.gen_pmax},
                                 functs={"connectivity_matrix": (
                                         lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                                          None, None, 22)
                                        }
                                )
                                
        # function cannot be called
        with self.assertRaises(RuntimeError):
            tmp = BoxGymObsSpace(self.env.observation_space,
                                 attr_to_keep=kept_attr,
                                 divide={"gen_p": self.env.gen_pmax,
                                         "load_p": self.obs_env.load_p,
                                         "actual_dispatch": self.env.gen_pmax},
                                 functs={"connectivity_matrix": (
                                          self.obs_env.connectivity_matrix().flatten(),
                                          None, None, None, None)
                                        }
                                )

        # low not correct
        with self.assertRaises(RuntimeError):
            tmp = BoxGymObsSpace(self.env.observation_space,
                                 attr_to_keep=kept_attr,
                                 divide={"gen_p": self.env.gen_pmax,
                                         "load_p": self.obs_env.load_p,
                                         "actual_dispatch": self.env.gen_pmax},
                                 functs={"connectivity_matrix": (
                                          lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                                          0.5, 1.0, None, None)
                                        }
                                )

        # high not correct
        with self.assertRaises(RuntimeError):
            tmp = BoxGymObsSpace(self.env.observation_space,
                                 attr_to_keep=kept_attr,
                                 divide={"gen_p": self.env.gen_pmax,
                                         "load_p": self.obs_env.load_p,
                                         "actual_dispatch": self.env.gen_pmax},
                                 functs={"connectivity_matrix": (
                                          lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                                          0., 0.9, None, None)
                                        }
                                )

        # not added in attr_to_keep
        with self.assertRaises(RuntimeError):
            tmp = BoxGymObsSpace(self.env.observation_space,
                                 attr_to_keep=["gen_p", "load_p", "topo_vect", "rho", "actual_dispatch"],
                                 divide={"gen_p": self.env.gen_pmax,
                                         "load_p": self.obs_env.load_p,
                                         "actual_dispatch": self.env.gen_pmax},
                                 functs={"connectivity_matrix": (
                                          lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                                          0., 1.0, None, None)
                                        }
                                )

        # another normal function
        self.env_gym.observation_space = BoxGymObsSpace(self.env.observation_space,
                                                   attr_to_keep=["connectivity_matrix", "log_load"],
                                                   functs={"connectivity_matrix":
                                                              (lambda grid2opobs: grid2opobs.connectivity_matrix().flatten(),
                                                               0., 1.0, None, None),
                                                           "log_load":
                                                            (lambda grid2opobs: np.log(grid2opobs.load_p),
                                                            None, 10., None, None)
                                                        }
                                                   )


if __name__ == "__main__":
    unittest.main()
