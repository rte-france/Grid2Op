# making some test that the backned is working as expected
import os
import sys
import unittest
import copy
import numpy as np
import pdb

from helper_path_test import PATH_DATA_TEST_PP, PATH_CHRONICS

from Exceptions import *
from Environment import Environment
from BackendPandaPower import PandaPowerBackend
from Parameters import Parameters
from ChronicsHandler import ChronicsHandler, GridStateFromFile
from Reward import L2RPNReward
from MakeEnv import make
from GameRules import GameRules, DefaultRules
from Action import Action
import time


DEBUG = False
PROFILE_CODE = False
if PROFILE_CODE:
    import cProfile


class TestLoadingBackendPandaPower(unittest.TestCase):
    def setUp(self):
        # powergrid
        self.backend = PandaPowerBackend()
        self.path_matpower = PATH_DATA_TEST_PP
        self.case_file = "test_case14.json"

        # chronics
        self.path_chron = os.path.join(PATH_CHRONICS, "chronics")
        self.chronics_handler = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path_chron)

        self.tolvect = 1e-2
        self.tol_one = 1e-5
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
                               actionClass=Action)
        self.array_double_dispatch = np.array([0.,  12.41833569,  10.89081339,   0., -23.30914908])

    def tearDown(self):
        pass

    def compare_vect(self, pred, true):
        return np.max(np.abs(pred- true)) <= self.tolvect

    def test_basic_redispatch_act(self):
        # test of the implementation of a simple case redispatching on one generator, bellow ramp min and ramp max
        act = self.env.action_space({"redispatch": [2, 5]})
        obs, reward, done, info = self.env.step(act)
        assert np.abs(np.sum(self.env.actual_dispatch)) <= self.tol_one
        th_dispatch = np.array([0., -1.30434783,  5.,  0., -3.69565217])
        assert self.compare_vect(self.env.actual_dispatch, th_dispatch)
        target_val = self.chronics_handler.real_data.prod_p[1, :] + self.env.actual_dispatch
        assert self.compare_vect(obs.prod_p[:-1], target_val[:-1])  # I remove last component which is the slack bus

        # check that the redispatching is apply in the right direction
        indx_ok = self.env.target_dispatch != 0.
        assert np.all(np.sign(self.env.actual_dispatch[indx_ok]) == np.sign(self.env.target_dispatch[indx_ok]))

    def test_redispatch_act_above_pmax(self):
        # in this test, the asked redispatching for generator 2 would make it above pmax, so the environment
        # need to "cut" it automatically, without invalidating the action
        act = self.env.action_space({"redispatch": [2, 20]})
        obs, reward, done, info = self.env.step(act)
        assert np.abs(np.sum(self.env.actual_dispatch)) <= self.tol_one
        th_dispatch = np.array([0., -2.69837697, 10.89081339,  0., -8.19243642])
        assert self.compare_vect(self.env.actual_dispatch, th_dispatch)
        target_val = self.chronics_handler.real_data.prod_p[1, :] + self.env.actual_dispatch
        assert self.compare_vect(obs.prod_p[:-1], target_val[:-1])  # I remove last component which is the slack bus

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

        th_dispatch = np.array([0. , 7.37325847,  10.38913319, 0., -17.76239165])
        assert self.compare_vect(self.env.actual_dispatch, th_dispatch)

        target_val = self.chronics_handler.real_data.prod_p[2, :] + self.env.actual_dispatch
        assert self.compare_vect(obs.prod_p[:-1], target_val[:-1])  # I remove last component which is the slack bus
        assert np.abs(np.sum(self.env.actual_dispatch)) <= self.tol_one

    def test_redispacth_two_gen(self):
        act = self.env.action_space({"redispatch": [(2, 20), (1, 10)]})
        obs, reward, done, info = self.env.step(act)

        th_dispatch = np.array([0., 10, 20., 0., 0.])
        assert self.compare_vect(self.env.target_dispatch, th_dispatch)

        assert self.compare_vect(self.env.actual_dispatch, self.array_double_dispatch)

        # check that the redispatching is apply in the right direction
        indx_ok = self.env.target_dispatch != 0.
        assert np.all(np.sign(self.env.actual_dispatch[indx_ok]) == np.sign(self.env.target_dispatch[indx_ok]))

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

    def test_count_turned_on(self):
        act = self.env.action_space()
        obs, reward, done, info = self.env.step(act)
        assert np.all(self.env.gen_uptime == np.array([0, 1, 1, 0, 1]))
        assert np.all(self.env.gen_downtime == np.array([1, 0, 0, 1, 0]))
        obs, reward, done, info = self.env.step(act)
        assert np.all(self.env.gen_uptime == np.array([0, 2, 2, 0, 2]))
        assert np.all(self.env.gen_downtime == np.array([2, 0, 0, 2, 0]))
        for i in range(63):
            obs, reward, done, info = self.env.step(act)

        obs, reward, done, info = self.env.step(act)
        assert np.all(self.env.gen_uptime == np.array([ 0, 66, 66,  1, 66]))
        assert np.all(self.env.gen_downtime == np.array([66, 0, 0, 0, 0]))

        obs, reward, done, info = self.env.step(act)
        assert np.all(self.env.gen_uptime == np.array([ 1, 67, 67,  2, 67]))
        assert np.all(self.env.gen_downtime == np.array([0, 0, 0, 0, 0]))


# TODO test that if i try to redispatched a turned off generator it breaks everything

if __name__ == "__main__":
    unittest.main()