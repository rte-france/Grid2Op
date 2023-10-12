# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import pdb
import warnings

from grid2op.tests.helper_path_test import *

from grid2op.Exceptions import *
from grid2op.Environment import Environment
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChronicsHandler, GridStateFromFile, ChangeNothing
from grid2op.Action import BaseAction

from grid2op.tests.BaseBackendTest import MakeBackend


class BaseTestRedispatch(MakeBackend):

    def get_path(self):
        return PATH_DATA_TEST_PP

    def get_casefile(self):
        return "test_case14.json"
    
    def setUp(self):
        super().setUp()
        # powergrid
        self.backend = self.make_backend()
        self.path_matpower = self.get_path()
        self.case_file = self.get_casefile()

        # chronics
        self.path_chron = os.path.join(PATH_CHRONICS, "chronics")
        self.chronics_handler = ChronicsHandler(
            chronicsClass=GridStateFromFile, path=self.path_chron
        )
        self.id_chron_to_back_load = np.array([0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9])

        # force the verbose backend
        self.backend.detailed_infos_for_cascading_failures = True
        self.names_chronics_to_backend = {
            "loads": {
                "2_C-10.61": "load_1_0",
                "3_C151.15": "load_2_1",
                "14_C63.6": "load_13_2",
                "4_C-9.47": "load_3_3",
                "5_C201.84": "load_4_4",
                "6_C-6.27": "load_5_5",
                "9_C130.49": "load_8_6",
                "10_C228.66": "load_9_7",
                "11_C-138.89": "load_10_8",
                "12_C-27.88": "load_11_9",
                "13_C-13.33": "load_12_10",
            },
            "lines": {
                "1_2_1": "0_1_0",
                "1_5_2": "0_4_1",
                "9_10_16": "8_9_2",
                "9_14_17": "8_13_3",
                "10_11_18": "9_10_4",
                "12_13_19": "11_12_5",
                "13_14_20": "12_13_6",
                "2_3_3": "1_2_7",
                "2_4_4": "1_3_8",
                "2_5_5": "1_4_9",
                "3_4_6": "2_3_10",
                "4_5_7": "3_4_11",
                "6_11_11": "5_10_12",
                "6_12_12": "5_11_13",
                "6_13_13": "5_12_14",
                "4_7_8": "3_6_15",
                "4_9_9": "3_8_16",
                "5_6_10": "4_5_17",
                "7_8_14": "6_7_18",
                "7_9_15": "6_8_19",
            },
            "prods": {
                "1_G137.1": "gen_0_4",
                "3_G36.31": "gen_2_1",
                "6_G63.29": "gen_5_2",
                "2_G-56.47": "gen_1_0",
                "8_G40.43": "gen_7_3",
            },
        }

        # _parameters for the environment
        self.env_params = Parameters()
        self.env_params.ALLOW_DISPATCH_GEN_SWITCH_OFF = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = Environment(
                init_grid_path=os.path.join(self.path_matpower, self.case_file),
                backend=self.backend,
                init_env_path=self.path_matpower,
                chronics_handler=self.chronics_handler,
                parameters=self.env_params,
                names_chronics_to_backend=self.names_chronics_to_backend,
                actionClass=BaseAction,
                name="test_redisp_env1",
            )
        self.array_double_dispatch = np.array([0.0, 10.0, 20.0, 0.0, -30.0])
        # self.array_double_dispatch = np.array([0.,  11.208119,  12.846733, 0., -24.054852])
        self.tol_one = self.env._tol_poly

    def tearDown(self):
        self.env.close()
        super().tearDown()

    def test_negative_dispatch(self):
        self.skip_if_needed()
        act = self.env.action_space({"redispatch": [(1, -10)]})
        obs, reward, done, info = self.env.step(act)
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)
        assert np.all(obs.prod_p <= self.env.gen_pmax + self.tol_one)
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one

    def test_no_impact_env(self):
        # perform a valid redispatching action
        self.skip_if_needed()
        obs_init = self.env.reset()  # reset the environment
        act = self.env.action_space()
        for i in range(
            1
        ):  # number cherry picked to introduce explain the behaviour in the cells bellow
            obsinit, rewardinit, doneinit, infoinit = self.env.step(
                self.env.action_space()
            )
        ref_data = copy.deepcopy(obsinit.prod_p)
        act = self.env.action_space({"redispatch": [(0, -10)]})
        # act = env.action_space({"redispatch": [(4,0)]})
        obs, reward, done, info = self.env.step(act)
        assert self.compare_vect(obsinit.prod_p, ref_data)

        target_val = obs.prod_p + self.env._actual_dispatch
        assert self.compare_vect(
            obs.prod_p[:-1], target_val[:-1]
        )  # I remove last component which is the slack bus
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)
        assert np.all(target_val <= self.env.gen_pmax + self.tol_one)
        assert np.all(obs.prod_p - obsinit.prod_p <= self.env.gen_max_ramp_up)
        assert np.all(obsinit.prod_p - obs.prod_p <= self.env.gen_max_ramp_down)

    def test_basic_redispatch_act(self):
        # test of the implementation of a simple case redispatching on one generator, bellow ramp min and ramp max
        self.skip_if_needed()
        act = self.env.action_space({"redispatch": (2, 5)})
        obs, reward, done, info = self.env.step(act)
        assert np.abs(np.sum(self.env._actual_dispatch)) <= self.tol_one
        th_dispatch = np.array([0.0, -2.5, 5.0, 0.0, -2.5])
        th_dispatch = np.array([0.0, -1.4814819, 5.0, 0.0, -3.518518])
        assert self.compare_vect(self.env._actual_dispatch, th_dispatch)
        target_val = (
            self.chronics_handler.real_data.prod_p[1, :] + self.env._actual_dispatch
        )
        assert self.compare_vect(
            obs.prod_p[:-1], target_val[:-1]
        )  # I remove last component which is the slack bus
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)
        assert np.all(target_val <= self.env.gen_pmax + self.tol_one)

        # check that the redispatching is apply in the right direction
        indx_ok = self.env._target_dispatch != 0.0
        assert np.all(
            np.sign(self.env._actual_dispatch[indx_ok])
            == np.sign(self.env._target_dispatch[indx_ok])
        )

    def test_redispatch_act_above_pmax(self):
        # in this test, the asked redispatching for generator 2 would make it above pmax, so the environment
        # need to "cut" it automatically, without invalidating the action
        self.skip_if_needed()
        act = self.env.action_space({"redispatch": (2, 60)})
        obs, reward, done, info = self.env.step(act)
        assert np.abs(np.sum(self.env._actual_dispatch)) <= self.tol_one
        th_dispatch = np.array([0.0, -23.2999, 50.899902, 0.0, -27.600002])
        th_dispatch = np.array([0.0, -20.0, 40.0, 0.0, -20.0])
        th_dispatch = np.array([0.0, -13.227808, 50.90005, 0.0, -37.67224])
        assert self.compare_vect(self.env._actual_dispatch, th_dispatch)
        target_val = (
            self.chronics_handler.real_data.prod_p[1, :] + self.env._actual_dispatch
        )
        assert self.compare_vect(
            obs.prod_p[:-1], target_val[:-1]
        )  # I remove last component which is the slack bus
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)
        assert np.all(target_val <= self.env.gen_pmax + self.tol_one)

    def test_two_redispatch_act(self):
        self.skip_if_needed()
        act = self.env.action_space({"redispatch": (2, 20)})
        obs_first, reward, done, info = self.env.step(act)
        act = self.env.action_space({"redispatch": (1, 10)})
        obs, reward, done, info = self.env.step(act)
        th_dispatch = np.array([0.0, 10, 20.0, 0.0, 0.0])
        th_dispatch[1] += obs_first.actual_dispatch[1]
        assert self.compare_vect(self.env._target_dispatch, th_dispatch)
        # check that the redispatching is apply in the right direction
        indx_ok = self.env._target_dispatch != 0.0
        assert np.all(
            np.sign(self.env._actual_dispatch[indx_ok])
            == np.sign(self.env._target_dispatch[indx_ok])
        )
        th_dispatch = np.array([0.0, 10.0, 20.0, 0.0, -30.0])
        th_dispatch = np.array([0.0, 4.0765514, 20.004545, 0.0, -24.081097])
        th_dispatch = np.array([0., 4.0710216, 20.015802, 0., -24.086824])
        assert self.compare_vect(self.env._actual_dispatch, th_dispatch)

        target_val = (
            self.chronics_handler.real_data.prod_p[2, :] + self.env._actual_dispatch
        )
        assert self.compare_vect(
            obs.prod_p[:-1], target_val[:-1]
        )  # I remove last component which is the slack bus
        assert np.abs(np.sum(self.env._actual_dispatch)) <= self.tol_one
        assert np.all(target_val <= self.env.gen_pmax + self.tol_one)
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)

    def test_redispacth_two_gen(self):
        self.skip_if_needed()
        act = self.env.action_space({"redispatch": [(2, 20), (1, 10)]})
        obs, reward, done, info = self.env.step(act)
        assert not done
        th_dispatch = np.array([0.0, 10, 20.0, 0.0, 0.0])
        assert self.compare_vect(self.env._target_dispatch, th_dispatch)
        assert self.compare_vect(self.env._actual_dispatch, self.array_double_dispatch)

        # check that the redispatching is apply in the right direction
        indx_ok = self.env._target_dispatch != 0.0
        assert np.all(
            np.sign(self.env._actual_dispatch[indx_ok])
            == np.sign(self.env._target_dispatch[indx_ok])
        )
        assert np.all(obs.prod_p <= self.env.gen_pmax + self.tol_one)
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)

    def test_redispacth_all_gen(self):
        # this should be exactly the same as the previous one
        self.skip_if_needed()
        act = self.env.action_space({"redispatch": [(2, 20.0), (1, 10.0), (4, -30.0)]})
        obs, reward, done, info = self.env.step(act)

        th_dispatch = np.array([0.0, 10, 20.0, 0.0, -30.0])
        assert self.compare_vect(self.env._target_dispatch, th_dispatch)
        assert self.compare_vect(self.env._actual_dispatch, self.array_double_dispatch)

        # check that the redispatching is apply in the right direction
        indx_ok = self.env._target_dispatch != 0.0
        assert np.all(
            np.sign(self.env._actual_dispatch[indx_ok])
            == np.sign(self.env._target_dispatch[indx_ok])
        )
        assert np.all(obs.prod_p <= self.env.gen_pmax + self.tol_one)
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)

    def test_count_turned_on(self):
        self.skip_if_needed()
        act = self.env.action_space()

        # recoded it: it's the normal behavior to call "env.reset()" to get the first time step
        obs = self.env.reset()
        assert np.all(self.env._gen_uptime == np.array([0, 1, 1, 0, 1]))
        assert np.all(self.env._gen_downtime == np.array([1, 0, 0, 1, 0]))
        assert np.all(obs.prod_p <= self.env.gen_pmax + self.tol_one)
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)

        obs, reward, done, info = self.env.step(act)
        assert np.all(self.env._gen_uptime == np.array([0, 2, 2, 0, 2]))
        assert np.all(self.env._gen_downtime == np.array([2, 0, 0, 2, 0]))
        assert np.all(obs.prod_p <= self.env.gen_pmax + self.tol_one)
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)

        for i in range(64):
            obs, reward, done, info = self.env.step(act)
            assert np.all(obs.prod_p <= self.env.gen_pmax + self.tol_one)
            assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)

        obs, reward, done, info = self.env.step(act)
        assert np.all(self.env._gen_uptime == np.array([0, 67, 67, 1, 67]))
        assert np.all(self.env._gen_downtime == np.array([67, 0, 0, 0, 0]))
        assert np.all(obs.prod_p <= self.env.gen_pmax + self.tol_one)
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)

        obs, reward, done, info = self.env.step(act)
        assert np.all(self.env._gen_uptime == np.array([1, 68, 68, 2, 68]))
        assert np.all(self.env._gen_downtime == np.array([0, 0, 0, 0, 0]))
        assert np.all(obs.prod_p <= self.env.gen_pmax + self.tol_one)
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)

    def test_redispacth_twice_same(self):
        self.skip_if_needed()
        # this should be exactly the same as the previous one
        act = self.env.action_space({"redispatch": [(2, 5.0)]})
        obs, reward, done, info = self.env.step(act)
        assert np.all(obs.target_dispatch == np.array([0.0, 0.0, 5.0, 0.0, 0.0]))
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        th_disp = np.array([0.0, -2.5, 5.0, 0.0, -2.5])
        th_disp = np.array([0.0, -1.4814819, 5.0, 0.0, -3.518518])
        assert self.compare_vect(obs.actual_dispatch, th_disp)
        assert np.all(obs.prod_p <= self.env.gen_pmax + self.tol_one)
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)

        act = self.env.action_space({"redispatch": [(2, 5.0)]})
        obs, reward, done, info = self.env.step(act)
        assert np.all(obs.target_dispatch == np.array([0.0, 0.0, 10.0, 0.0, 0.0]))
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        th_disp = np.array([0.0, -5.0, 10.0, 0.0, -5.0])
        th_disp = np.array([0., -2.9629638, 10.,  0., -7.037036 ])
        assert self.compare_vect(obs.actual_dispatch, th_disp)
        assert np.all(obs.prod_p <= self.env.gen_pmax + self.tol_one)
        assert np.all(obs.prod_p - self.env.gen_pmin >= -self.tol_one)

    def test_redispacth_secondabovepmax(self):
        self.skip_if_needed()
        act = self.env.action_space({"redispatch": [(2, 20.0)]})
        obs0, reward, done, info = self.env.step(act)
        assert np.all(obs0.target_dispatch == np.array([0.0, 0.0, 20.0, 0.0, 0.0]))
        assert np.abs(np.sum(obs0.actual_dispatch)) <= self.tol_one
        th_disp = np.array([0.0, -10.0, 20.0, 0.0, -10.0])
        th_disp = np.array([0.0, -5.9259276, 20.0, 0.0, -14.074072])
        assert self.compare_vect(obs0.actual_dispatch, th_disp)
        assert np.all(obs0.prod_p <= self.env.gen_pmax + self.tol_one)
        assert np.all(obs0.prod_p >= self.env.gen_pmin - self.tol_one)

        act = self.env.action_space({"redispatch": [(2, 40.0)]})
        obs, reward, done, info = self.env.step(act)
        assert not info["is_dispatching_illegal"]
        assert np.all(obs.target_dispatch == np.array([0.0, 0.0, 60.0, 0.0, 0.0]))
        th_disp = np.array([0.0, -23.5, 50.4, 0.0, -26.900002])
        th_disp = np.array([0., -12.977809, 50.40005, 0., -37.42224 ])
        assert self.compare_vect(obs.actual_dispatch, th_disp)
        assert np.all(obs.prod_p[:-1] <= self.env.gen_pmax[:-1] + self.tol_one)
        assert np.all(obs.prod_p[:-1] >= self.env.gen_pmin[:-1] - self.tol_one)
        assert np.all(
            obs.prod_p[:-1] - obs0.prod_p[:-1] >= -self.env.gen_max_ramp_down[:-1]
        )
        assert np.all(
            obs.prod_p[:-1] - obs0.prod_p[:-1] <= self.env.gen_max_ramp_up[:-1]
        )

    def test_redispacth_non_dispatchable_generator(self):
        """Dispatch a non redispatchable generator is ambiguous"""
        self.skip_if_needed()
        act = self.env.action_space()
        obs, reward, done, info = self.env.step(act)

        # Check that generator 0 isn't redispatchable
        assert self.env.gen_redispatchable[0] == False
        # Check that generator 0 is off
        assert self.env._gen_downtime[0] >= 1

        # Try to redispatch
        redispatch_act = self.env.action_space({"redispatch": [(0, 5.0)]})
        obs, reward, done, info = self.env.step(redispatch_act)

        assert info["is_ambiguous"]


class BaseTestRedispatchChangeNothingEnvironment(MakeBackend):

    def get_path(self):
        return PATH_DATA_TEST_PP

    def get_casefile(self):
        return "test_case14.json"
    
    def setUp(self):
        super().setUp()
        # powergrid
        self.backend = self.make_backend()
        self.path_matpower = self.get_path()
        self.case_file = self.get_casefile()

        # chronics
        self.path_chron = os.path.join(PATH_CHRONICS, "chronics")
        self.chronics_handler = ChronicsHandler(chronicsClass=ChangeNothing)
        self.id_chron_to_back_load = np.array([0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9])

        # force the verbose backend
        self.backend.detailed_infos_for_cascading_failures = True
        self.names_chronics_to_backend = {
            "loads": {
                "2_C-10.61": "load_1_0",
                "3_C151.15": "load_2_1",
                "14_C63.6": "load_13_2",
                "4_C-9.47": "load_3_3",
                "5_C201.84": "load_4_4",
                "6_C-6.27": "load_5_5",
                "9_C130.49": "load_8_6",
                "10_C228.66": "load_9_7",
                "11_C-138.89": "load_10_8",
                "12_C-27.88": "load_11_9",
                "13_C-13.33": "load_12_10",
            },
            "lines": {
                "1_2_1": "0_1_0",
                "1_5_2": "0_4_1",
                "9_10_16": "8_9_2",
                "9_14_17": "8_13_3",
                "10_11_18": "9_10_4",
                "12_13_19": "11_12_5",
                "13_14_20": "12_13_6",
                "2_3_3": "1_2_7",
                "2_4_4": "1_3_8",
                "2_5_5": "1_4_9",
                "3_4_6": "2_3_10",
                "4_5_7": "3_4_11",
                "6_11_11": "5_10_12",
                "6_12_12": "5_11_13",
                "6_13_13": "5_12_14",
                "4_7_8": "3_6_15",
                "4_9_9": "3_8_16",
                "5_6_10": "4_5_17",
                "7_8_14": "6_7_18",
                "7_9_15": "6_8_19",
            },
            "prods": {
                "1_G137.1": "gen_0_4",
                "3_G36.31": "gen_2_1",
                "6_G63.29": "gen_5_2",
                "2_G-56.47": "gen_1_0",
                "8_G40.43": "gen_7_3",
            },
        }

        # _parameters for the environment
        self.env_params = Parameters()
        self.env_params.ALLOW_DISPATCH_GEN_SWITCH_OFF = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = Environment(
                init_grid_path=os.path.join(self.path_matpower, self.case_file),
                backend=self.backend,
                init_env_path=self.path_matpower,
                chronics_handler=self.chronics_handler,
                parameters=self.env_params,
                names_chronics_to_backend=self.names_chronics_to_backend,
                actionClass=BaseAction,
                name="test_redisp_env2",
            )
        self.tol_one = self.env._tol_poly

    def tearDown(self):
        self.env.close()
        super().tearDown()

    def test_redispatch_generator_off(self):
        """Redispatch a turned off generator is illegal"""
        self.skip_if_needed()
        # Step into simulation once
        nothing_act = self.env.action_space()
        obs, reward, done, info = self.env.step(nothing_act)

        # Check that generator 1 is redispatchable
        assert self.env.gen_redispatchable[1] == True

        # Check that generator 1 is off
        assert obs.prod_p[1] == 0
        assert self.env._gen_downtime[1] >= 1

        # Try to redispatch generator 1
        redispatch_act = self.env.action_space({"redispatch": [(1, 5.0)]})
        obs, reward, done, info = self.env.step(redispatch_act)

        assert info["is_dispatching_illegal"] == True


class BaseTestRedispTooLowHigh(MakeBackend):
    # test bug reported in issues https://github.com/rte-france/Grid2Op/issues/44
    def setUp(self) -> None:
        super().setUp()
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("rte_case14_redisp",
                            test=True,
                            backend=backend,
                            _add_to_name=type(self).__name__)

        # i don't want to be bother by ramps in these test (note that is NOT recommended to change that)
        type(self.env).gen_max_ramp_down[:] = 5000
        type(self.env).gen_max_ramp_up[:] = 5000
        act_cls = type(self.env.action_space())
        act_cls.gen_max_ramp_down[:] = 5000
        act_cls.gen_max_ramp_up[:] = 5000

        self.msg_ = (
            'Grid2OpException AmbiguousAction InvalidRedispatching NotEnoughGenerators "Attempt to use a '
            "redispatch action that does not sum to 0., but a"
        )
        self.tol_one = self.env._tol_poly

    def tearDown(self):
        self.env.close()
        super().tearDown()

    def test_redisp_toohigh_toolow(self):
        """
        This test that: 1) if i do a valid redispatching, it's valid
        2) if i set up a redispatching too high (higher than pmax - pmin for a generator) it's not valid
        3) if i set up a redispatching too low (demanding to decrease more than pmax - pmin) it's not valid
        :return:
        """
        self.skip_if_needed()
        # this dispatch (though legal) broke everything
        act = self.env.action_space({"redispatch": (0, -1)})
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert not info["is_dispatching_illegal"]
        assert np.all(self.env._target_dispatch == [-1.0, 0.0, 0.0, 0.0, 0.0])
        act = self.env.action_space({"redispatch": (0, 0)})
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert not info["is_dispatching_illegal"]
        assert np.all(self.env._target_dispatch == [-1.0, 0.0, 0.0, 0.0, 0.0])

        # this one is not correct: too high decrease
        act = self.env.action_space(
            {"redispatch": (0, self.env.gen_pmin[0] - self.env.gen_pmax[0])}
        )
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert info["is_dispatching_illegal"]
        assert np.all(self.env._target_dispatch == [-1.0, 0.0, 0.0, 0.0, 0.0])

        # this one is not correct: too high increase
        act = self.env.action_space(
            {"redispatch": (0, self.env.gen_pmax[0] - self.env.gen_pmin[0] + 2)}
        )
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert info["is_dispatching_illegal"]
        assert np.all(self.env._target_dispatch == [-1.0, 0.0, 0.0, 0.0, 0.0])

    def test_error_message_notzerosum_oneshot(self):
        self.skipTest("Ok with new redispatching implementation")
        act = self.env.action_space(
            {
                "redispatch": [
                    (0, 4.9999784936326535),
                    (1, 4.78524395611872),
                    (4, -9.999591852954794),
                ]
            }
        )
        obs, reward, done, info = self.env.step(act)
        assert info["is_dispatching_illegal"]
        assert info["exception"][0].__str__()[:140] == self.msg_

    def test_error_message_notzerosum_threesteps(self):
        self.skipTest("Ok with new redispatching implementation")
        act = self.env.action_space({"redispatch": [(0, 4.9999784936326535)]})
        obs, reward, done, info = self.env.step(act)
        assert info["is_dispatching_illegal"] is False

        act = self.env.action_space({"redispatch": [(1, 4.78524395611872)]})
        obs, reward, done, info = self.env.step(act)
        assert info["is_dispatching_illegal"] is False

        act = self.env.action_space({"redispatch": [(4, -9.999591852954794)]})
        obs, reward, done, info = self.env.step(act)
        assert info["is_dispatching_illegal"]
        assert info["exception"][0].__str__()[:140] == self.msg_


class BaseTestDispatchRampingIllegalETC(MakeBackend):
    def setUp(self):
        super().setUp()
        # powergrid
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("rte_case14_test", test=True, backend=backend,
                            _add_to_name=type(self).__name__)
        self.tol_one = self.env._tol_poly

    def tearDown(self):
        self.env.close()
        super().tearDown()

    def test_invalid_dispatch(self):
        self.skip_if_needed()
        self.env.set_id(0)  # make sure to use the same environment input data.
        obs_init = self.env.reset()  # reset the environment
        act = self.env.action_space()
        for i in range(
            2
        ):  # number cherry picked to introduce explain the behaviour in the cells bellow
            obsinit, rewardinit, doneinit, infoinit = self.env.step(act)
        act = self.env.action_space({"redispatch": [(0, -10)]})
        obs, reward, done, info = self.env.step(act)
        assert len(info["exception"])

    def test_redispatch_rampminmax(self):
        self.skip_if_needed()
        # test that the redispatch value is always above the ramp min and below the ramp max
        self.env.set_id(0)  # make sure to use the same environment input data.
        obs_init = self.env.reset()  # reset the environment
        act = self.env.action_space()
        for i in range(
            2
        ):  # number cherry picked to introduce explain the behaviour in the cells bellow
            obsinit, rewardinit, doneinit, infoinit = self.env.step(act)
        act = self.env.action_space({"redispatch": [(0, -5)]})
        # act = env.action_space({"redispatch": [(4,0)]})
        obs, reward, done, info = self.env.step(act)
        target_p = self.env.chronics_handler.real_data.data.prod_p[3, :]
        target_p_t = self.env.chronics_handler.real_data.data.prod_p[2, :]
        assert self.compare_vect(obsinit.prod_p[:-1], target_p_t[:-1])
        # only look at dispatchable generator, remove slack bus (last generator)
        assert np.all(
            obs.prod_p[0:2] - obsinit.prod_p[0:2]
            <= obs.gen_max_ramp_up[0:2] + self.tol_one
        )
        assert np.all(
            obs.prod_p[0:2] - obsinit.prod_p[0:2]
            >= -obs.gen_max_ramp_down[0:2] - self.tol_one
        )
        assert np.all(obs.prod_p[0:2] >= obs.gen_pmin[0:2] - self.tol_one)
        assert np.all(obs.prod_p[0:2] <= obs.gen_pmax[0:2] + self.tol_one)

    def test_redispatch_noneedtocurtaildispact(self):
        self.skip_if_needed()
        # test that the redispatch value is always above the ramp min and below the ramp max
        self.env.set_id(0)  # make sure to use the same environment input data.
        obs_init = self.env.reset()  # reset the environment
        act = self.env.action_space()
        for i in range(
            2
        ):  # number cherry picked to introduce explain the behaviour in the cells bellow
            obsinit, rewardinit, doneinit, infoinit = self.env.step(act)
            assert len(infoinit["exception"]) == 0
        act = self.env.action_space({"redispatch": [(0, +5)]})
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert np.all(self.env._target_dispatch == [5.0, 0.0, 0.0, 0.0, 0.0])
        target_p = self.env.chronics_handler.real_data.data.prod_p[3, :]
        target_p_t = self.env.chronics_handler.real_data.data.prod_p[2, :]
        assert self.compare_vect(obsinit.prod_p[:-1], target_p_t[:-1])
        # only look at dispatchable generator, remove slack bus (last generator)
        assert np.all(
            obs.prod_p[0:2] - obsinit.prod_p[0:2]
            <= obs.gen_max_ramp_up[0:2] + self.tol_one
        )
        assert np.all(
            obs.prod_p[0:2] - obsinit.prod_p[0:2]
            >= -obs.gen_max_ramp_down[0:2] - self.tol_one
        )
        assert np.all(
            np.abs(self.env._actual_dispatch - np.array([5.0, -2.5, 0.0, 0.0, -2.5]))
            <= self.tol_one
        )

    def test_sum0_again(self):
        # perform a valid redispatching action
        self.skip_if_needed()
        self.env.set_id(0)  # make sure to use the same environment input data.
        obs_init = self.env.reset()  # reset the environment
        act = self.env.action_space({"redispatch": [(0, +10)]})
        obs, reward, done, info = self.env.step(act)
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        indx_ok = self.env._target_dispatch != 0.0
        assert np.all(
            np.sign(self.env._actual_dispatch[indx_ok])
            == np.sign(self.env._target_dispatch[indx_ok])
        )

    def test_sum0_again2(self):
        self.skip_if_needed()
        env = self.env
        # perform a valid redispatching action
        env.set_id(0)  # make sure to use the same environment input data.
        obs_init = env.reset()  # reset the environment
        act = env.action_space()
        act = env.action_space({"redispatch": [(0, +5)]})
        obs, reward, done, info = env.step(act)
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        indx_ok = self.env._target_dispatch != 0.0
        assert np.all(
            np.sign(self.env._actual_dispatch[indx_ok])
            == np.sign(self.env._target_dispatch[indx_ok])
        )
        donothing = env.action_space()
        obsinit, reward, done, info = env.step(donothing)
        act = env.action_space({"redispatch": [(0, -5)]})
        # act = env.action_space({"redispatch": [(0,0)]})
        obs, reward, done, info = env.step(act)
        assert np.all(
            obs.prod_p[0:2] - obsinit.prod_p[0:2]
            <= obs.gen_max_ramp_up[0:2] + self.tol_one
        )
        assert np.all(
            obs.prod_p[0:2] - obsinit.prod_p[0:2]
            >= -obs.gen_max_ramp_down[0:2] - self.tol_one
        )
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one

    def test_sum0_again3(self):
        self.skip_if_needed()
        env = self.env
        # perform a valid redispatching action
        env.set_id(0)  # make sure to use the same environment input data.
        obs_init = env.reset()  # reset the environment
        act = env.action_space()
        # ask +5
        act = env.action_space({"redispatch": [(0, +5)]})
        obs, reward, done, info = env.step(act)
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        indx_ok = self.env._target_dispatch != 0.0
        assert np.all(
            np.sign(self.env._actual_dispatch[indx_ok])
            == np.sign(self.env._target_dispatch[indx_ok])
        )
        assert np.all(
            obs.prod_p[0:2] - obs_init.prod_p[0:2]
            <= obs.gen_max_ramp_up[0:2] + self.tol_one
        )
        assert np.all(
            obs.prod_p[0:2] - obs_init.prod_p[0:2]
            >= -obs.gen_max_ramp_down[0:2] - self.tol_one
        )
        assert np.all(
            np.abs(obs.actual_dispatch - np.array([5.0, -2.5, 0.0, 0.0, -2.5]))
            <= self.tol_one
        )
        assert len(info["exception"]) == 0
        # wait for the setpoint to be reached
        donothing = env.action_space()
        obsinit, reward, done, info = env.step(donothing)
        assert np.all(
            np.abs(obs.actual_dispatch - np.array([5.0, -2.5, 0.0, 0.0, -2.5]))
            <= self.tol_one
        )
        assert len(info["exception"]) == 0
        # "cancel" action
        act = env.action_space({"redispatch": [(0, -5)]})
        obs, reward, done, info = env.step(act)
        assert not done
        assert np.all(
            obs.prod_p[0:2] - obsinit.prod_p[0:2]
            <= obs.gen_max_ramp_up[0:2] + self.tol_one
        )
        assert np.all(
            obs.prod_p[0:2] - obsinit.prod_p[0:2]
            >= -obs.gen_max_ramp_down[0:2] - self.tol_one
        )
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        assert len(info["exception"]) == 0
        # wait for setpoint to be reached
        obsfinal, reward, done, info = env.step(donothing)
        assert not done
        assert np.all(
            obsfinal.prod_p[0:2] - obs.prod_p[0:2]
            <= obs.gen_max_ramp_up[0:2] + self.tol_one
        )
        assert np.all(
            obsfinal.prod_p[0:2] - obs.prod_p[0:2]
            >= -obs.gen_max_ramp_down[0:2] - self.tol_one
        )
        assert (
            np.abs(np.sum(obsfinal.actual_dispatch)) <= self.tol_one
        )  # redispatching should sum at 0.
        assert (
            np.sum(np.abs(obsfinal.actual_dispatch)) <= self.tol_one
        )  # redispatching should be canceled by now
        assert len(info["exception"]) == 0

    def test_dispatch_still_not_zero(self):
        self.skip_if_needed()
        env = self.env

        max_iter = 40
        # agent = GreedyEconomic(env.action_space)
        done = False
        # reward = env.reward_range[0]

        env.set_id(0)  # reset the env to the same id
        obs_init = env.reset()
        i = 0
        act = env.action_space({"redispatch": [(0, obs_init.gen_max_ramp_up[0])]})
        while not done:
            obs, reward, done, info = env.step(act)
            # print("act._redisp {}".format(act._redispatch))
            assert not done, "game over at iteration {}".format(i)
            assert len(info["exception"]) == 0, "error at iteration {}".format(i)
            # NB: only gen 0 and 1 are included because gen 2,3 are renewables and gen 4 is slack bus
            assert np.all(
                obs.prod_p[0:2] - obs_init.prod_p[0:2]
                <= obs.gen_max_ramp_up[0:2] + self.tol_one
            ), "above max_ramp for ts {}".format(i)
            assert np.all(
                obs.prod_p[0:2] - obs_init.prod_p[0:2]
                >= -obs.gen_max_ramp_down[0:2] - self.tol_one
            ), "below min_ramp for ts {}".format(i)
            try:
                assert np.all(
                    obs.prod_p[0:2] <= obs.gen_pmax[0:2]
                ), "above pmax for ts {}".format(i)
            except:
                pass
            assert np.all(
                obs.prod_p[0:2] >= -obs.gen_pmin[0:2]
            ), "below pmin for ts {}".format(i)
            assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one

            i += 1
            obs_init = obs
            if i >= max_iter:
                break

        obs, reward, done, info = env.step(act)
        assert np.all(
            obs.prod_p[0:2] - obs_init.prod_p[0:2]
            <= obs.gen_max_ramp_up[0:2] + self.tol_one
        ), "above max_ramp at the end"
        assert np.all(
            obs.prod_p[0:2] - obs_init.prod_p[0:2]
            >= -obs.gen_max_ramp_down[0:2] - self.tol_one
        ), "above min_ramp at the end"
        assert np.all(
            obs.prod_p[0:2] <= obs.gen_pmax[0:2] + self.tol_one
        ), "above pmax at the end"
        assert np.all(
            obs.prod_p[0:2] >= -obs.gen_pmin[0:2] - self.tol_one
        ), "below pmin at the end"
        assert (
            np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        ), "redisp not 0 at the end"
        # this redispatching is impossible because we ask to increase the value of the generator of 210
        # which is higher than pmax
        assert len(info["exception"]), "this redispatching should not be possible"


class BaseTestLoadingAcceptAlmostZeroSumRedisp(MakeBackend):
    def setUp(self):
        super().setUp()
        # powergrid
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("rte_case14_test", test=True, backend=backend,
                            _add_to_name=type(self).__name__)
        self.tol_one = self.env._tol_poly

    def tearDown(self):
        self.env.close()
        super().tearDown()

    def test_accept_almost_zerozum_too_high(self):
        self.skip_if_needed()
        self.skipTest("it is possible now to accept pretty much everything")
        redisp_act = self.env.action_space(
            {"redispatch": [(0, 3), (1, -1), (-1, -2 + 1e-7)]}
        )
        obs, reward, done, info = self.env.step(redisp_act)
        assert np.all(obs.prod_p[0:2] <= obs.gen_pmax[0:2])
        assert np.all(obs.prod_p[0:2] >= -obs.gen_pmin[0:2])
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        assert len(info["exception"]) == 0

    def test_accept_almost_zerozum_too_low(self):
        self.skip_if_needed()
        self.skipTest("it is possible now to accept pretty much everything")
        redisp_act = self.env.action_space(
            {"redispatch": [(0, 3), (1, -1), (-1, -2 - 1e-7)]}
        )
        obs, reward, done, info = self.env.step(redisp_act)
        assert np.all(obs.prod_p[0:2] <= obs.gen_pmax[0:2])
        assert np.all(obs.prod_p[0:2] >= -obs.gen_pmin[0:2])
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        assert len(info["exception"]) == 0

    def test_accept_almost_zerozum_shouldnotbepossible_low(self):
        self.skip_if_needed()
        self.skipTest("it is possible now to accept pretty much everything")
        redisp_act = self.env.action_space(
            {"redispatch": [(0, 3), (1, -1), (-1, -2 - 1e-1)]}
        )
        obs, reward, done, info = self.env.step(redisp_act)
        assert np.all(obs.prod_p[0:2] <= obs.gen_pmax[0:2])
        assert np.all(obs.prod_p[0:2] >= -obs.gen_pmin[0:2])
        assert np.all(obs.actual_dispatch == 0.0)
        assert len(info["exception"])

    def test_accept_almost_zerozum_shouldnotbepossible_high(self):
        self.skip_if_needed()
        self.skipTest("it is possible now to accept pretty much everything")
        redisp_act = self.env.action_space(
            {"redispatch": [(0, 3), (1, -1), (-1, -2 + 1e-1)]}
        )
        obs, reward, done, info = self.env.step(redisp_act)
        assert np.all(obs.prod_p[0:2] <= obs.gen_pmax[0:2])
        assert np.all(obs.prod_p[0:2] >= -obs.gen_pmin[0:2])
        assert np.all(obs.actual_dispatch == 0.0)
        assert len(info["exception"])
