# making some test that the backned is working as expected
import os
import sys
import unittest
import datetime

import time

import numpy as np
import pdb

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

from Agent import PowerLineSwitch, TopologyGreedy, DoNothingAgent

DEBUG = False


class TestAgent(unittest.TestCase):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.game_rules = GameRules()
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

    def _aux_test_agent(self, agent, i_max=30):
        done = False
        i = 0
        beg_ = time.time()
        cum_reward = 0.
        act = self.env.helper_action_player({})
        time_act = 0.
        while not done:
            obs, reward, done, info = self.env.step(act)  # should load the first time stamp
            beg__ = time.time()
            act = agent.act(obs, reward, done)
            end__ = time.time()
            time_act += end__ - beg__
            cum_reward += reward
            i += 1
            if i > i_max:
                break

        end_ = time.time()
        if DEBUG:
            li_text = ["Env: {:.2f}s", "\t - apply act {:.2f}s", "\t - run pf: {:.2f}s",
                       "\t - env update + observation: {:.2f}s", "Agent: {:.2f}s", "Total time: {:.2f}s",
                       "Cumulative reward: {:1f}"]
            msg_ = "\n".join(li_text)
            print(msg_.format(
                self.env._time_apply_act+self.env._time_powerflow+self.env._time_extract_obs,
                self.env._time_apply_act, self.env._time_powerflow, self.env._time_extract_obs,
                time_act, end_-beg_, cum_reward))
        return i, cum_reward

    def test_0_donothing(self):
        agent = DoNothingAgent(self.env.helper_action_player)
        i, cum_reward = self._aux_test_agent(agent)
        assert i == 31, "The powerflow diverged before step 30 for do nothing"
        assert np.abs(cum_reward - 619.994619) <= self.tol_one, "The reward has not been properly computed"

    def test_1_powerlineswitch(self):
        agent = PowerLineSwitch(self.env.helper_action_player)
        i, cum_reward = self._aux_test_agent(agent)
        assert i == 31, "The powerflow diverged before step 30 for powerline switch agent"
        assert np.abs(cum_reward - 619.9950) <= self.tol_one, "The reward has not been properly computed"

    def test_2_busswitch(self):
        agent = TopologyGreedy(self.env.helper_action_player)
        i, cum_reward = self._aux_test_agent(agent, i_max=10)
        assert i == 11, "The powerflow diverged before step 10 for greedy agent"
        assert np.abs(cum_reward - 219.99795) <= self.tol_one, "The reward has not been properly computed"


if __name__ == "__main__":
    unittest.main()
