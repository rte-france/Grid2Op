# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import sys
import unittest
import copy
import numpy as np
import pdb
import warnings

from grid2op.tests.helper_path_test import *

from grid2op.Exceptions import *
from grid2op.Environment import Environment
from grid2op.Backend import PandaPowerBackend
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChronicsHandler, GridStateFromFile, ChangeNothing
from grid2op.Reward import L2RPNReward
from grid2op.MakeEnv import make
from grid2op.Rules import RulesChecker, DefaultRules
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
import time


DEBUG = False
PROFILE_CODE = False
if PROFILE_CODE:
    import cProfile


class TestRedispatch(HelperTests):
    def setUp(self):
        # powergrid
        self.backend = PandaPowerBackend()
        self.path_matpower = PATH_DATA_TEST_PP
        self.case_file = "test_case14.json"
        # chronics
        self.path_chron = os.path.join(PATH_CHRONICS, "chronics")
        self.chronics_handler = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path_chron)
        self.id_chron_to_back_load = np.array([0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9])

        # force the verbose backend
        self.backend.detailed_infos_for_cascading_failures = True
        self.names_chronics_to_backend = {"loads": {"2_C-10.61": 'load_1_0', "3_C151.15": 'load_2_1',
                                                    "14_C63.6": 'load_13_2', "4_C-9.47": 'load_3_3',
                                                    "5_C201.84": 'load_4_4',
                                                    "6_C-6.27": 'load_5_5', "9_C130.49": 'load_8_6',
                                                    "10_C228.66": 'load_9_7',
                                                    "11_C-138.89": 'load_10_8', "12_C-27.88": 'load_11_9',
                                                    "13_C-13.33": 'load_12_10'},
                                          "lines": {'1_2_1': '0_1_0', '1_5_2': '0_4_1', '9_10_16': '8_9_2',
                                                    '9_14_17': '8_13_3',
                                                    '10_11_18': '9_10_4', '12_13_19': '11_12_5', '13_14_20': '12_13_6',
                                                    '2_3_3': '1_2_7', '2_4_4': '1_3_8', '2_5_5': '1_4_9',
                                                    '3_4_6': '2_3_10',
                                                    '4_5_7': '3_4_11', '6_11_11': '5_10_12', '6_12_12': '5_11_13',
                                                    '6_13_13': '5_12_14', '4_7_8': '3_6_15', '4_9_9': '3_8_16',
                                                    '5_6_10': '4_5_17',
                                                    '7_8_14': '6_7_18', '7_9_15': '6_8_19'},
                                          "prods": {"1_G137.1": 'gen_0_4', "3_G36.31": "gen_2_1", "6_G63.29": "gen_5_2",
                                                    "2_G-56.47": "gen_1_0", "8_G40.43": "gen_7_3"},
                                          }

        # _parameters for the environment
        self.env_params = Parameters()
        self.env = Environment(init_grid_path=os.path.join(self.path_matpower, self.case_file),
                               backend=self.backend,
                               chronics_handler=self.chronics_handler,
                               parameters=self.env_params,
                               names_chronics_to_backend=self.names_chronics_to_backend,
                               actionClass=BaseAction)
        self.array_double_dispatch = np.array([0.,  10.,  20.,   0., -30.])

    def tearDown(self):
        self.env.close()

    def test_negative_dispatch(self):
        act = self.env.action_space({"redispatch": [(1, -10)]})
        obs, reward, done, info = self.env.step(act)
        assert np.all(obs.prod_p >= self.env.gen_pmin)
        assert np.all(obs.prod_p <= self.env.gen_pmax)
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one

    def test_no_impact_env(self):
        # perform a valid redispatching action
        obs_init = self.env.reset()  # reset the environment
        act = self.env.action_space()
        for i in range(1):  # number cherry picked to introduce explain the behaviour in the cells bellow
            obsinit, rewardinit, doneinit, infoinit = self.env.step(self.env.action_space())
        ref_data = copy.deepcopy(obsinit.prod_p)
        act = self.env.action_space({"redispatch": [(0, -10)]})
        # act = env.action_space({"redispatch": [(4,0)]})
        obs, reward, done, info = self.env.step(act)
        assert self.compare_vect(obsinit.prod_p, ref_data)

        target_val = obs.prod_p + self.env.actual_dispatch
        assert self.compare_vect(obs.prod_p[:-1], target_val[:-1])  # I remove last component which is the slack bus
        assert np.all(obs.prod_p >= self.env.gen_pmin)
        assert np.all(target_val <= self.env.gen_pmax)
        assert np.all(obs.prod_p - obsinit.prod_p <= self.env.gen_max_ramp_up)
        assert np.all(obsinit.prod_p - obs.prod_p <= self.env.gen_max_ramp_down)

    def test_basic_redispatch_act(self):
        # test of the implementation of a simple case redispatching on one generator, bellow ramp min and ramp max
        act = self.env.action_space({"redispatch": [2, 5]})
        obs, reward, done, info = self.env.step(act)
        assert np.abs(np.sum(self.env.actual_dispatch)) <= self.tol_one
        th_dispatch = np.array([0., -1.44301856,  5.,  0., -3.55698144])
        assert self.compare_vect(self.env.actual_dispatch, th_dispatch)
        target_val = self.chronics_handler.real_data.prod_p[1, :] + self.env.actual_dispatch
        assert self.compare_vect(obs.prod_p[:-1], target_val[:-1])  # I remove last component which is the slack bus
        assert np.all(obs.prod_p >= self.env.gen_pmin)
        assert np.all(target_val <= self.env.gen_pmax)

        # check that the redispatching is apply in the right direction
        indx_ok = self.env.target_dispatch != 0.
        assert np.all(np.sign(self.env.actual_dispatch[indx_ok]) == np.sign(self.env.target_dispatch[indx_ok]))

    def test_redispatch_act_above_pmax(self):
        # in this test, the asked redispatching for generator 2 would make it above pmax, so the environment
        # need to "cut" it automatically, without invalidating the action
        act = self.env.action_space({"redispatch": [2, 60]})
        obs, reward, done, info = self.env.step(act)
        assert np.abs(np.sum(self.env.actual_dispatch)) <= self.tol_one
        th_dispatch = np.array([0., -10.57042905,  50.89066718,   0., -40.32023813])
        assert self.compare_vect(self.env.actual_dispatch, th_dispatch)
        target_val = self.chronics_handler.real_data.prod_p[1, :] + self.env.actual_dispatch
        assert self.compare_vect(obs.prod_p[:-1], target_val[:-1])  # I remove last component which is the slack bus
        assert np.all(obs.prod_p >= self.env.gen_pmin)
        assert np.all(target_val <= self.env.gen_pmax)

    def test_two_redispatch_act(self):
        act = self.env.action_space({"redispatch": [2, 20]})
        obs, reward, done, info = self.env.step(act)
        act = self.env.action_space({"redispatch": [1, 10]})
        obs, reward, done, info = self.env.step(act)
        th_dispatch = np.array([0., 10, 20., 0., 0.])
        assert self.compare_vect(self.env.target_dispatch, th_dispatch)
        # check that the redispatching is apply in the right direction
        indx_ok = self.env.target_dispatch != 0.
        assert np.all(np.sign(self.env.actual_dispatch[indx_ok]) == np.sign(self.env.target_dispatch[indx_ok]))
        th_dispatch = np.array([0.,  10.,  20.,   0., -30.])
        assert self.compare_vect(self.env.actual_dispatch, th_dispatch)

        target_val = self.chronics_handler.real_data.prod_p[2, :] + self.env.actual_dispatch
        assert self.compare_vect(obs.prod_p[:-1], target_val[:-1])  # I remove last component which is the slack bus
        assert np.abs(np.sum(self.env.actual_dispatch)) <= self.tol_one
        assert np.all(target_val <= self.env.gen_pmax)
        assert np.all(obs.prod_p >= self.env.gen_pmin)

    def test_redispacth_two_gen(self):
        act = self.env.action_space({"redispatch": [(2, 20), (1, 10)]})
        obs, reward, done, info = self.env.step(act)
        th_dispatch = np.array([0., 10, 20., 0., 0.])
        assert self.compare_vect(self.env.target_dispatch, th_dispatch)
        assert self.compare_vect(self.env.actual_dispatch, self.array_double_dispatch)

        # check that the redispatching is apply in the right direction
        indx_ok = self.env.target_dispatch != 0.
        assert np.all(np.sign(self.env.actual_dispatch[indx_ok]) == np.sign(self.env.target_dispatch[indx_ok]))
        assert np.all(obs.prod_p <= self.env.gen_pmax)
        assert np.all(obs.prod_p >= self.env.gen_pmin)

    def test_redispacth_all_gen(self):
        # this should be exactly the same as the previous one
        act = self.env.action_space({"redispatch": [(2, 20.), (1, 10.), (4, -30.)]})
        obs, reward, done, info = self.env.step(act)

        th_dispatch = np.array([0., 10, 20., 0., -30.])
        assert self.compare_vect(self.env.target_dispatch, th_dispatch)
        assert self.compare_vect(self.env.actual_dispatch, self.array_double_dispatch)

        # check that the redispatching is apply in the right direction
        indx_ok = self.env.target_dispatch != 0.
        assert np.all(np.sign(self.env.actual_dispatch[indx_ok]) == np.sign(self.env.target_dispatch[indx_ok]))
        assert np.all(obs.prod_p <= self.env.gen_pmax)
        assert np.all(obs.prod_p >= self.env.gen_pmin)

    def test_count_turned_on(self):
        act = self.env.action_space()
        obs, reward, done, info = self.env.step(act)
        # pdb.set_trace()
        assert np.all(self.env.gen_uptime == np.array([0, 1, 1, 0, 1]))
        assert np.all(self.env.gen_downtime == np.array([1, 0, 0, 1, 0]))
        assert np.all(obs.prod_p <= self.env.gen_pmax)
        assert np.all(obs.prod_p >= self.env.gen_pmin)

        obs, reward, done, info = self.env.step(act)
        assert np.all(self.env.gen_uptime == np.array([0, 2, 2, 0, 2]))
        assert np.all(self.env.gen_downtime == np.array([2, 0, 0, 2, 0]))
        assert np.all(obs.prod_p <= self.env.gen_pmax)
        assert np.all(obs.prod_p >= self.env.gen_pmin)

        for i in range(63):
            obs, reward, done, info = self.env.step(act)
            assert np.all(obs.prod_p <= self.env.gen_pmax)
            assert np.all(obs.prod_p >= self.env.gen_pmin)

        obs, reward, done, info = self.env.step(act)
        assert np.all(self.env.gen_uptime == np.array([0, 66, 66,  1, 66]))
        assert np.all(self.env.gen_downtime == np.array([66, 0, 0, 0, 0]))
        assert np.all(obs.prod_p <= self.env.gen_pmax)
        assert np.all(obs.prod_p >= self.env.gen_pmin)

        obs, reward, done, info = self.env.step(act)
        assert np.all(self.env.gen_uptime == np.array([1, 67, 67,  2, 67]))
        assert np.all(self.env.gen_downtime == np.array([0, 0, 0, 0, 0]))
        assert np.all(obs.prod_p <= self.env.gen_pmax)
        assert np.all(obs.prod_p >= self.env.gen_pmin)

    def test_redispacth_twice_same(self):
        # this should be exactly the same as the previous one
        act = self.env.action_space({"redispatch": [(2, 5.)]})
        obs, reward, done, info = self.env.step(act)
        assert np.all(obs.target_dispatch == np.array([ 0.,  0., 5.,  0.,  0.]))
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        assert self.compare_vect(obs.actual_dispatch, np.array([ 0., -1.44301856,  5.,  0., -3.55698144]))
        assert np.all(obs.prod_p <= self.env.gen_pmax)
        assert np.all(obs.prod_p >= self.env.gen_pmin)

        act = self.env.action_space({"redispatch": [(2, 5.)]})
        obs, reward, done, info = self.env.step(act)
        assert np.all(obs.target_dispatch == np.array([ 0.,  0., 10.,  0.,  0.]))
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        assert self.compare_vect(obs.actual_dispatch, np.array([0., -2.81339987, 10.,  0., -7.18660013]))
        assert np.all(obs.prod_p <= self.env.gen_pmax)
        assert np.all(obs.prod_p >= self.env.gen_pmin)

    def test_redispacth_secondabovepmax(self):
        act = self.env.action_space({"redispatch": [(2, 20.)]})
        obs, reward, done, info = self.env.step(act)
        assert np.all(obs.target_dispatch == np.array([0.,  0., 20.,  0.,  0.]))
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        assert self.compare_vect(obs.actual_dispatch, np.array([0., -5.36765536,  20., 0., -14.63234464]))
        assert np.all(obs.prod_p <= self.env.gen_pmax)
        assert np.all(obs.prod_p >= self.env.gen_pmin)

        act = self.env.action_space({"redispatch": [(2, 40.)]})
        obs, reward, done, info = self.env.step(act)
        assert np.all(obs.target_dispatch == np.array([0.,  0., 60.,  0.,  0.]))
        assert self.compare_vect(obs.actual_dispatch, np.array([0., -10.3814061, 50.39070301, 0., -40.00929691]))
        assert np.all(obs.prod_p[:-1] <= self.env.gen_pmax[:-1])
        assert np.all(obs.prod_p[:-1] >= self.env.gen_pmin[:-1])

    def test_redispacth_non_dispatchable_generator(self):
        """ Dispatch a non redispatchable generator is ambiguous """
        act = self.env.action_space()
        obs, reward, done, info = self.env.step(act)

        # Check that generator 0 isn't redispatchable
        assert self.env.gen_redispatchable[0] == False
        # Check that generator 0 is off
        assert self.env.gen_downtime[0] >= 1

        # Try to redispatch
        redispatch_act = self.env.action_space({"redispatch": [(0, 5.)]})
        obs, reward, done, info = self.env.step(redispatch_act)

        assert info['is_ambiguous']


class TestRedispatchChangeNothingEnvironment(HelperTests):
    def setUp(self):
        # powergrid
        self.backend = PandaPowerBackend()
        self.path_matpower = PATH_DATA_TEST_PP
        self.case_file = "test_case14.json"
        # chronics
        self.path_chron = os.path.join(PATH_CHRONICS, "chronics")
        self.chronics_handler = ChronicsHandler(chronicsClass=ChangeNothing)
        self.id_chron_to_back_load = np.array([0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9])

        # force the verbose backend
        self.backend.detailed_infos_for_cascading_failures = True
        self.names_chronics_to_backend = {"loads": {"2_C-10.61": 'load_1_0', "3_C151.15": 'load_2_1',
                                                    "14_C63.6": 'load_13_2', "4_C-9.47": 'load_3_3',
                                                    "5_C201.84": 'load_4_4',
                                                    "6_C-6.27": 'load_5_5', "9_C130.49": 'load_8_6',
                                                    "10_C228.66": 'load_9_7',
                                                    "11_C-138.89": 'load_10_8', "12_C-27.88": 'load_11_9',
                                                    "13_C-13.33": 'load_12_10'},
                                          "lines": {'1_2_1': '0_1_0', '1_5_2': '0_4_1', '9_10_16': '8_9_2',
                                                    '9_14_17': '8_13_3',
                                                    '10_11_18': '9_10_4', '12_13_19': '11_12_5', '13_14_20': '12_13_6',
                                                    '2_3_3': '1_2_7', '2_4_4': '1_3_8', '2_5_5': '1_4_9',
                                                    '3_4_6': '2_3_10',
                                                    '4_5_7': '3_4_11', '6_11_11': '5_10_12', '6_12_12': '5_11_13',
                                                    '6_13_13': '5_12_14', '4_7_8': '3_6_15', '4_9_9': '3_8_16',
                                                    '5_6_10': '4_5_17',
                                                    '7_8_14': '6_7_18', '7_9_15': '6_8_19'},
                                          "prods": {"1_G137.1": 'gen_0_4', "3_G36.31": "gen_2_1", "6_G63.29": "gen_5_2",
                                                    "2_G-56.47": "gen_1_0", "8_G40.43": "gen_7_3"},
                                          }

        # _parameters for the environment
        self.env_params = Parameters()
        self.env = Environment(init_grid_path=os.path.join(self.path_matpower, self.case_file),
                               backend=self.backend,
                               chronics_handler=self.chronics_handler,
                               parameters=self.env_params,
                               names_chronics_to_backend=self.names_chronics_to_backend,
                               actionClass=BaseAction)

    def tearDown(self):
        self.env.close()

    def test_redispatch_generator_off(self):
        """ Redispatch a turned off generator is illegal """

        # Step into simulation once
        nothing_act = self.env.action_space()
        obs, reward, done, info = self.env.step(nothing_act)

        # Check that generator 1 is redispatchable
        assert self.env.gen_redispatchable[1] == True

        # Check that generator 1 is off
        assert obs.prod_p[1] == 0
        assert self.env.gen_downtime[1] >= 1

        # Try to redispatch generator 1
        redispatch_act = self.env.action_space({"redispatch": [(1, 5.)]})
        obs, reward, done, info = self.env.step(redispatch_act)

        assert info['is_dispatching_illegal'] == True


class TestRedispTooLowHigh(HelperTests):
    # test bug reported in issues https://github.com/rte-france/Grid2Op/issues/44
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("case14_redisp")

        # i don't want to be bother by ramps in these test (note that is NOT recommended to change that)
        self.env.gen_max_ramp_down[:] = 5000
        self.env.gen_max_ramp_up[:] = 5000
        self.msg_ = 'Grid2OpException AmbiguousAction InvalidRedispatching NotEnoughGenerators "Attempt to use a ' \
               'redispatch action that does not sum to 0., but a'

    def tearDown(self):
        self.env.close()

    def test_redisp_toohigh_toolow(self):
        """
        This test that: 1) if i do a valid redispatching, it's valid
        2) if i set up a redispatching too high (higher than pmax - pmin for a generator) it's not valid
        3) if i set up a redispatching too low (demanding to decrease more than pmax - pmin) it's not valid
        :return:
        """
        # this dispatch (though legal) broke everything
        act = self.env.action_space({"redispatch": [0, -1]})
        obs, reward, done, info = self.env.step(act)
        assert info["is_dispatching_illegal"] is False
        assert np.all(self.env.target_dispatch == [-1., 0., 0., 0., 0.])
        act = self.env.action_space({"redispatch": [0, 0]})
        obs, reward, done, info = self.env.step(act)
        assert info["is_dispatching_illegal"] is False
        assert np.all(self.env.target_dispatch == [-1., 0., 0., 0., 0.])

        # this one is not correct: too high decrease
        act = self.env.action_space({"redispatch": [0, self.env.gen_pmin[0] - self.env.gen_pmax[0]]})
        obs, reward, done, info = self.env.step(act)
        assert info["is_dispatching_illegal"]
        assert np.all(self.env.target_dispatch == [-1., 0., 0., 0., 0.])

        # this one is not correct: too high increase
        act = self.env.action_space({"redispatch": [0, self.env.gen_pmax[0] - self.env.gen_pmin[0] +2 ]})
        obs, reward, done, info = self.env.step(act)
        assert info["is_dispatching_illegal"]
        assert np.all(self.env.target_dispatch == [-1., 0., 0., 0. ,0.])

    def test_error_message_notzerosum_oneshot(self):
        act = self.env.action_space(
            {"redispatch": [(0, 4.9999784936326535), (1, 4.78524395611872), (4, -9.999591852954794)]})
        obs, reward, done, info = self.env.step(act)
        assert info["is_dispatching_illegal"]
        assert info["exception"][0].__str__()[:140] == self.msg_

    def test_error_message_notzerosum_threesteps(self):

        act = self.env.action_space({"redispatch": [(0, 4.9999784936326535)]}) #, (1, 4.78524395611872), (4, -9.999591852954794)]})
        obs, reward, done, info = self.env.step(act)
        assert info["is_dispatching_illegal"] is False

        act = self.env.action_space({"redispatch": [(1, 4.78524395611872)]})
        obs, reward, done, info = self.env.step(act)
        assert info["is_dispatching_illegal"] is False

        act = self.env.action_space({"redispatch": [(4, -9.999591852954794)]})
        obs, reward, done, info = self.env.step(act)
        assert info["is_dispatching_illegal"]
        assert info["exception"][0].__str__()[:140] == self.msg_


class TestLoadingBackendPandaPower(HelperTests):
    def setUp(self):
        # powergrid
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("case14_test")

    def tearDown(self):
        self.env.close()

    def test_invalid_dispatch(self):
        self.env.set_id(0)  # make sure to use the same environment input data.
        obs_init = self.env.reset()  # reset the environment
        act = self.env.action_space()
        for i in range(2):  # number cherry picked to introduce explain the behaviour in the cells bellow
            obsinit, rewardinit, doneinit, infoinit = self.env.step(act)
        act = self.env.action_space({"redispatch": [(0, -10)]})
        obs, reward, done, info = self.env.step(act)
        assert len(info["exception"])

    def test_redispatch_rampminmax(self):
        # test that the redispatch value is always above the ramp min and below the ramp max
        self.env.set_id(0)  # make sure to use the same environment input data.
        obs_init = self.env.reset()  # reset the environment
        act = self.env.action_space()
        for i in range(2):  # number cherry picked to introduce explain the behaviour in the cells bellow
            obsinit, rewardinit, doneinit, infoinit = self.env.step(act)
        act = self.env.action_space({"redispatch": [(0, -5)]})
        # act = env.action_space({"redispatch": [(4,0)]})
        obs, reward, done, info = self.env.step(act)
        target_p = self.env.chronics_handler.real_data.data.prod_p[3, :]
        target_p_t = self.env.chronics_handler.real_data.data.prod_p[2, :]
        assert self.compare_vect(obsinit.prod_p[:-1], target_p_t[:-1])
        # only look at dispatchable generator, remove slack bus (last generator)
        assert np.all(obs.prod_p[0:2] - obsinit.prod_p[0:2] <= obs.gen_max_ramp_up[0:2])
        assert np.all(obs.prod_p[0:2] - obsinit.prod_p[0:2] >= -obs.gen_max_ramp_down[0:2])

    def test_redispatch_noneedtocurtaildispact(self):
         # test that the redispatch value is always above the ramp min and below the ramp max
         self.env.set_id(0)  # make sure to use the same environment input data.
         obs_init = self.env.reset()  # reset the environment
         act = self.env.action_space()
         for i in range(2):  # number cherry picked to introduce explain the behaviour in the cells bellow
             obsinit, rewardinit, doneinit, infoinit = self.env.step(act)
             assert len(infoinit["exception"]) == 0
         act = self.env.action_space({"redispatch": [(0, +5)]})
         obs, reward, done, info = self.env.step(act)
         target_p = self.env.chronics_handler.real_data.data.prod_p[3, :]
         target_p_t = self.env.chronics_handler.real_data.data.prod_p[2, :]
         assert self.compare_vect(obsinit.prod_p[:-1], target_p_t[:-1])
         # only look at dispatchable generator, remove slack bus (last generator)
         assert np.all(obs.prod_p[0:2] - obsinit.prod_p[0:2] <= obs.gen_max_ramp_up[0:2])
         assert np.all(obs.prod_p[0:2] - obsinit.prod_p[0:2] >= -obs.gen_max_ramp_down[0:2])
         assert np.all(np.abs(self.env.actual_dispatch - np.array([5., -2.5,  0.,  0., -2.5])) <= self.tol_one)

    def test_sum0_again(self):
         # perform a valid redispatching action
         self.env.set_id(0)  # make sure to use the same environment input data.
         obs_init = self.env.reset()  # reset the environment
         act = self.env.action_space({"redispatch": [(0, +10)]})
         obs, reward, done, info = self.env.step(act)
         assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
         indx_ok = self.env.target_dispatch != 0.
         assert np.all(np.sign(self.env.actual_dispatch[indx_ok]) == np.sign(self.env.target_dispatch[indx_ok]))

    def test_sum0_again2(self):
         env = self.env
         # perform a valid redispatching action
         env.set_id(0)  # make sure to use the same environment input data.
         obs_init = env.reset()  # reset the environment
         act = env.action_space()
         act = env.action_space({"redispatch": [(0, +5)]})
         obs, reward, done, info = env.step(act)
         assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
         indx_ok = self.env.target_dispatch != 0.
         assert np.all(np.sign(self.env.actual_dispatch[indx_ok]) == np.sign(self.env.target_dispatch[indx_ok]))
         donothing = env.action_space()
         obsinit, reward, done, info = env.step(donothing)
         act = env.action_space({"redispatch": [(0, -5)]})
         # act = env.action_space({"redispatch": [(0,0)]})
         obs, reward, done, info = env.step(act)
         assert np.all(obs.prod_p[0:2] - obsinit.prod_p[0:2] <= obs.gen_max_ramp_up[0:2])
         assert np.all(obs.prod_p[0:2] - obsinit.prod_p[0:2] >= -obs.gen_max_ramp_down[0:2])
         assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one

    def test_sum0_again3(self):
        env = self.env
        # perform a valid redispatching action
        env.set_id(0)  # make sure to use the same environment input data.
        obs_init = env.reset()  # reset the environment
        act = env.action_space()
         # ask +5
        act = env.action_space({"redispatch": [(0, +5)]})
        obs, reward, done, info = env.step(act)
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        indx_ok = self.env.target_dispatch != 0.
        assert np.all(np.sign(self.env.actual_dispatch[indx_ok]) == np.sign(self.env.target_dispatch[indx_ok]))
        assert np.all(obs.prod_p[0:2] - obs_init.prod_p[0:2] <= obs.gen_max_ramp_up[0:2])
        assert np.all(obs.prod_p[0:2] - obs_init.prod_p[0:2] >= -obs.gen_max_ramp_down[0:2])
        assert np.all(obs.actual_dispatch == np.array([5.0, -2.5, 0., 0., -2.5]))
        assert len(info['exception']) == 0
         # wait for the setpoint to be reached
        donothing = env.action_space()
        obsinit, reward, done, info = env.step(donothing)
        assert np.all(obs.actual_dispatch == np.array([5.0, -2.5, 0., 0., -2.5]))
        assert len(info['exception']) == 0
         # "cancel" action
        act = env.action_space({"redispatch": [(0, -5)]})
        obs, reward, done, info = env.step(act)
        assert np.all(obs.prod_p[0:2] - obsinit.prod_p[0:2] <= obs.gen_max_ramp_up[0:2])
        assert np.all(obs.prod_p[0:2] - obsinit.prod_p[0:2] >= -obs.gen_max_ramp_down[0:2])
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        assert len(info['exception']) == 0
         # wait for setpoint to be reached
        obsfinal, reward, done, info = env.step(donothing)
        assert np.all(obsfinal.prod_p[0:2] - obs.prod_p[0:2] <= obs.gen_max_ramp_up[0:2])
        assert np.all(obsfinal.prod_p[0:2] - obs.prod_p[0:2] >= -obs.gen_max_ramp_down[0:2])
        assert np.abs(np.sum(obsfinal.actual_dispatch)) <= self.tol_one
        # pdb.set_trace()
        assert np.sum(np.abs(obsfinal.actual_dispatch)) <= self.tol_one  # redispatching should be canceled by now
        assert len(info['exception']) == 0

    def test_dispatch_still_not_zero(self):
        env = self.env

        max_iter = 27
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
            assert np.all(obs.prod_p[0:2] - obs_init.prod_p[0:2] <= obs.gen_max_ramp_up[0:2])
            assert np.all(obs.prod_p[0:2] - obs_init.prod_p[0:2] >= -obs.gen_max_ramp_down[0:2])
            assert np.all(obs.prod_p[0:2] <= obs.gen_pmax[0:2])
            assert np.all(obs.prod_p[0:2] >= -obs.gen_pmin[0:2])
            assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
            assert len(info['exception']) == 0, "error at iteration {}".format(i)
            i += 1
            obs_init = obs
            if i >= max_iter:
                break

        obs, reward, done, info = env.step(act)
        assert np.all(obs.prod_p[0:2] - obs_init.prod_p[0:2] <= obs.gen_max_ramp_up[0:2])
        assert np.all(obs.prod_p[0:2] - obs_init.prod_p[0:2] >= -obs.gen_max_ramp_down[0:2])
        assert np.all(obs.prod_p[0:2] <= obs.gen_pmax[0:2])
        assert np.all(obs.prod_p[0:2] >= -obs.gen_pmin[0:2])
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        assert len(info['exception']), "this redispatching should not be possible"


class TestLoadingAcceptAlmostZeroSumRedisp(HelperTests):
    def setUp(self):
        # powergrid
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("case14_test")

    def tearDown(self):
        self.env.close()

    def test_accept_almost_zerozum_too_high(self):
        redisp_act = self.env.action_space({"redispatch": [(0, 3), (1, -1), (-1, -2 + 1e-7)]})
        obs, reward, done, info = self.env.step(redisp_act)
        assert np.all(obs.prod_p[0:2] <= obs.gen_pmax[0:2])
        assert np.all(obs.prod_p[0:2] >= -obs.gen_pmin[0:2])
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        assert len(info['exception']) == 0

    def test_accept_almost_zerozum_too_low(self):
        redisp_act = self.env.action_space({"redispatch": [(0, 3), (1, -1), (-1, -2 - 1e-7)]})
        obs, reward, done, info = self.env.step(redisp_act)
        assert np.all(obs.prod_p[0:2] <= obs.gen_pmax[0:2])
        assert np.all(obs.prod_p[0:2] >= -obs.gen_pmin[0:2])
        assert np.abs(np.sum(obs.actual_dispatch)) <= self.tol_one
        assert len(info['exception']) == 0

    def test_accept_almost_zerozum_shouldnotbepossible_low(self):
        redisp_act = self.env.action_space({"redispatch": [(0, 3), (1, -1), (-1, -2 - 1e-1)]})
        obs, reward, done, info = self.env.step(redisp_act)
        assert np.all(obs.prod_p[0:2] <= obs.gen_pmax[0:2])
        assert np.all(obs.prod_p[0:2] >= -obs.gen_pmin[0:2])
        assert np.all(obs.actual_dispatch == 0.)
        assert len(info['exception'])

    def test_accept_almost_zerozum_shouldnotbepossible_high(self):
        redisp_act = self.env.action_space({"redispatch": [(0, 3), (1, -1), (-1, -2 + 1e-1)]})
        obs, reward, done, info = self.env.step(redisp_act)
        assert np.all(obs.prod_p[0:2] <= obs.gen_pmax[0:2])
        assert np.all(obs.prod_p[0:2] >= -obs.gen_pmin[0:2])
        assert np.all(obs.actual_dispatch == 0.)
        assert len(info['exception'])


if __name__ == "__main__":
    unittest.main()
