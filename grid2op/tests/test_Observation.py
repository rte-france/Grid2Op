# making some test that the backned is working as expected
import os
import sys
import unittest
import datetime

import numpy as np
import pdb

# making sure test can be ran from:
# root package directory
# RL4Grid subdirectory
# RL4Grid/tests subdirectory
from helper_path_test import PATH_DATA_TEST_PP, PATH_CHRONICS

from Exceptions import *
from Observation import ObservationHelper, CompleteObservation, ObsEnv

from ChronicsHandler import ChronicsHandler, ChangeNothing, GridStateFromFile, GridStateFromFileWithForecasts

from Exceptions import *
from Action import HelperAction
from GameRules import GameRules
from Reward import L2RPNReward
from Parameters import Parameters

from BackendPandaPower import PandaPowerBackend
from Environment import Environment


# TODO add unit test for obs.connectity_matrix()
# TODO add unit test for obs.bus_connectivity_matrix()
# todo add unit test for the proper update the backend in the observation [for now there is a "data leakage" as
# the real backend is copied when the observation is built, but i need to make a test to check that's it's properly
# copied]

class TestLoadingBackendFunc(unittest.TestCase):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        # from ADNBackend import ADNBackend
        # self.backend = ADNBackend()
        # self.path_matpower = "/home/donnotben/Documents/RL4Grid/RL4Grid/data"
        # self.case_file = "ieee14_ADN.xml"
        # self.backend.load_grid(self.path_matpower, self.case_file)
        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.game_rules = GameRules()
        # pdb.set_trace()
        self.rewardClass = L2RPNReward
        self.reward_helper = self.rewardClass()
        self.obsClass = CompleteObservation
        self.parameters = Parameters()

        # powergrid
        self.backend = PandaPowerBackend()
        self.path_matpower = PATH_DATA_TEST_PP
        self.case_file = "test_case14.json"

        # chronics
        self.path_chron = os.path.join(PATH_CHRONICS, "chronics_with_forecast")
        self.chronics_handler = ChronicsHandler(chronicsClass=GridStateFromFileWithForecasts, path=self.path_chron)

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
                               rewardClass=self.rewardClass)

    def test_0_load_properly(self):
        # this test aims at checking that everything in setUp is working properly, eg that "ObsEnv" class has enough
        # information for example
        pass

    def test_1_generating_obs(self):
        # test that helper_obs is abl to generate a valid observation
        obs = self.env.helper_observation(self.env)
        pass

    def test_2_reset(self):
        # test that helper_obs is abl to generate a valid observation
        obs = self.env.helper_observation(self.env)
        assert obs.prod_p[0] is not None
        obs.reset()
        assert obs.prod_p is None

    def test_3_reset(self):
        # test that helper_obs is abl to generate a valid observation
        obs = self.env.helper_observation(self.env)
        obs2 = obs.copy()
        assert obs == obs2
        obs2.reset()
        assert obs2.prod_p is None
        assert obs.prod_p is not None

    def test_4_to_from_vect(self):
        # test that helper_obs is abl to generate a valid observation
        obs = self.env.helper_observation(self.env)
        obs2 = self.env.helper_observation(self.env)
        vect = obs.to_vect()
        assert vect.shape[0] == obs.size()
        obs2.reset()
        obs2.from_vect(vect)
        assert obs == obs2
        vect2 = obs2.to_vect()
        assert np.all(vect == vect2)

    def test_5_simulate_proper_timestep(self):
        obs_orig = self.env.helper_observation(self.env)
        action = self.env.helper_action_player({})
        action2 = self.env.helper_action_player({})

        simul_obs, simul_reward, simul_has_error, simul_info = obs_orig.simulate(action)
        real_obs, real_reward, real_has_error, real_info = self.env.step(action2)
        assert not real_has_error, "The powerflow diverged"

        # this is not true for every observation chronics, but we made sure in this files that the forecast were
        # without any noise
        assert simul_obs == real_obs, "there is a mismatch in the observation, though they are supposed to be equal"
        assert np.abs(simul_reward- real_reward) <= self.tol_one

    def test_6_simulate_dont_affect_env(self):
        obs_orig = self.env.helper_observation(self.env)
        obs_orig = obs_orig.copy()

        for i in range(self.env.backend.n_lines):
            # simulate lots of action
            tmp = np.full(self.env.backend.n_lines, fill_value=False, dtype=np.bool)
            tmp[i] = True
            action = self.env.helper_action_player({"change_line_status": tmp})
            simul_obs, simul_reward, simul_has_error, simul_info = obs_orig.simulate(action)

        obs_after = self.env.helper_observation(self.env)
        assert obs_orig == obs_after


if __name__ == "__main__":
    unittest.main()