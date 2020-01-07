# making some test that the backned is working as expected
import os
import sys
import unittest
import datetime
import json
from io import StringIO
import numpy as np
from numpy import dtype
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

        self.dict_ = {'name_gen': ['gen_1_0', 'gen_2_1', 'gen_5_2', 'gen_7_3', 'gen_0_4'],
                      'name_load': ['load_1_0', 'load_2_1', 'load_13_2', 'load_3_3', 'load_4_4', 'load_5_5', 'load_8_6',
                                    'load_9_7', 'load_10_8', 'load_11_9', 'load_12_10'],
                      'name_line': ['0_1_0', '0_4_1', '8_9_2', '8_13_3', '9_10_4', '11_12_5', '12_13_6', '1_2_7',
                                    '1_3_8', '1_4_9', '2_3_10', '3_4_11', '5_10_12', '5_11_13', '5_12_14', '3_6_15',
                                    '3_8_16', '4_5_17', '6_7_18', '6_8_19'],
                      'name_sub': ['sub_0', 'sub_1', 'sub_10', 'sub_11', 'sub_12', 'sub_13', 'sub_2', 'sub_3', 'sub_4',
                                   'sub_5', 'sub_6', 'sub_7', 'sub_8', 'sub_9'],
                 'sub_info': [3, 6, 4, 6, 5, 6, 3, 2, 5, 3, 3, 3, 4, 3],
                 'load_to_subid': [1, 2, 13, 3, 4, 5, 8, 9, 10, 11, 12],
                 'gen_to_subid': [1, 2, 5, 7, 0],
                 'line_or_to_subid': [0, 0, 8, 8, 9, 11, 12, 1, 1, 1, 2, 3, 5, 5, 5, 3, 3, 4, 6, 6],
                 'line_ex_to_subid': [1, 4, 9, 13, 10, 12, 13, 2, 3, 4, 3, 4, 10, 11, 12, 6, 8, 5, 7, 8],
                 'load_to_sub_pos': [5, 3, 2, 5, 4, 5, 4, 2, 2, 2, 3], 'gen_to_sub_pos': [4, 2, 4, 1, 2],
                 'line_or_to_sub_pos': [0, 1, 0, 1, 1, 0, 1, 1, 2, 3, 1, 2, 0, 1, 2, 3, 4, 3, 1, 2],
                 'line_ex_to_sub_pos': [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 1, 1, 2, 0, 2, 3, 0, 3],
                 'load_pos_topo_vect': [8, 12, 55, 18, 23, 29, 39, 42, 45, 48, 52],
                 'gen_pos_topo_vect': [7, 11, 28, 34, 2],
                 'line_or_pos_topo_vect': [0, 1, 35, 36, 41, 46, 50, 4, 5, 6, 10, 15, 24, 25, 26, 16, 17, 22, 31, 32],
                 'line_ex_pos_topo_vect': [3, 19, 40, 53, 43, 49, 54, 9, 13, 20, 14, 21, 44, 47, 51, 30, 37, 27, 33, 38],
                 'subtype': 'Observation.CompleteObservation'}
        self.dtypes = np.array([dtype('int64'), dtype('int64'), dtype('int64'), dtype('int64'),
                                           dtype('int64'), dtype('int64'), dtype('float64'), dtype('float64'),
                                           dtype('float64'), dtype('float64'), dtype('float64'),
                                           dtype('float64'), dtype('float64'), dtype('float64'),
                                           dtype('float64'), dtype('float64'), dtype('float64'),
                                           dtype('float64'), dtype('float64'), dtype('float64'),
                                           dtype('float64'), dtype('bool'), dtype('int64'), dtype('int64'),
                                           dtype('int64'), dtype('int64'), dtype('int64'), dtype('int64'),
                                           dtype('int64'), dtype('int64')], dtype=object)
        self.shapes = np.array([ 1,  1,  1,  1,  1,  1,  5,  5,  5, 11, 11, 11, 20, 20, 20, 20, 20,
                                            20, 20, 20, 20, 20, 20, 56, 20, 20, 14, 20, 20, 20])

    def test_sum_shape_equal_size(self):
        obs = self.env.helper_observation(self.env)
        assert obs.size() == np.sum(obs.shape())

    def test_conn_mat(self):
        obs = self.env.helper_observation(self.env)
        obs.bus_connectivity_matrix()
        obs.connectivity_matrix()

    def test_shape_correct(self):
        obs = self.env.helper_observation(self.env)
        assert obs.shape().shape == obs.dtype().shape
        assert np.all(obs.dtype() == self.dtypes)
        assert np.all(obs.shape() == self.shapes)

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
        assert np.all(np.isnan(obs.prod_p))
        assert np.all(obs.dtype() == self.dtypes)
        assert np.all(obs.shape() == self.shapes)

    def test_3_reset(self):
        # test that helper_obs is able to generate a valid observation
        obs = self.env.helper_observation(self.env)
        obs2 = obs.copy()
        assert obs == obs2
        obs2.reset()
        assert np.all(np.isnan(obs2.prod_p))
        assert np.all(obs2.dtype() == self.dtypes)
        assert np.all(obs2.shape() == self.shapes)
        # assert obs.prod_p is not None

    def test_shapes_types(self):
        obs = self.env.helper_observation(self.env)
        dtypes = obs.dtype()
        assert np.all(dtypes == self.dtypes)
        shapes = obs.shape()
        assert np.all(shapes == self.shapes)

    def test_4_to_from_vect(self):
        # test that helper_obs is abl to generate a valid observation
        obs = self.env.helper_observation(self.env)
        obs2 = self.env.helper_observation(self.env)
        vect = obs.to_vect()
        assert vect.shape[0] == obs.size()
        obs2.reset()
        obs2.from_vect(vect)

        assert np.all(obs.dtype() == self.dtypes)
        assert np.all(obs.shape() == self.shapes)

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
        # without any noise, maintenance, nor hazards
        assert simul_obs == real_obs, "there is a mismatch in the observation, though they are supposed to be equal"
        assert np.abs(simul_reward - real_reward) <= self.tol_one

    def test_6_simulate_dont_affect_env(self):
        obs_orig = self.env.helper_observation(self.env)
        obs_orig = obs_orig.copy()

        for i in range(self.env.backend.n_line):
            # simulate lots of action
            tmp = np.full(self.env.backend.n_line, fill_value=False, dtype=np.bool)
            tmp[i] = True
            action = self.env.helper_action_player({"change_line_status": tmp})
            simul_obs, simul_reward, simul_has_error, simul_info = obs_orig.simulate(action)

        obs_after = self.env.helper_observation(self.env)
        assert obs_orig == obs_after

    def test_inspect_load(self):
        obs = self.env.helper_observation(self.env)
        dict_ = obs.state_of(load_id=0)
        assert "p" in dict_
        assert dict_["p"] == 18.8
        assert "q" in dict_
        assert dict_["q"] == 13.4
        assert "v" in dict_
        assert dict_["v"] == 141.075
        assert "bus" in dict_
        assert dict_["bus"] == 1
        assert "sub_id" in dict_
        assert dict_["sub_id"] == 1

    def test_inspect_gen(self):
        obs = self.env.helper_observation(self.env)
        dict_ = obs.state_of(gen_id=0)
        assert "p" in dict_
        assert dict_["p"] == 0.0
        assert "q" in dict_
        assert np.abs(dict_["q"] - 47.48313177017934) <= self.tol_one
        assert "v" in dict_
        assert np.abs(dict_["v"] - 141.075) <= self.tol_one
        assert "bus" in dict_
        assert dict_["bus"] == 1
        assert "sub_id" in dict_
        assert dict_["sub_id"] == 1

    def test_inspect_line(self):
        obs = self.env.helper_observation(self.env)
        dict_both = obs.state_of(line_id=0)
        assert "origin" in dict_both
        dict_ = dict_both["origin"]

        assert "p" in dict_
        assert np.abs(dict_["p"] - 109.77536682689008) <= self.tol_one
        assert "q" in dict_
        assert np.abs(dict_["q"] - -8.7165023030358) <= self.tol_one
        assert "v" in dict_
        assert np.abs(dict_["v"] - 143.1) <= self.tol_one
        assert "bus" in dict_
        assert dict_["bus"] == 1
        assert "sub_id" in dict_
        assert dict_["sub_id"] == 0

        assert "extremity" in dict_both
        dict_ = dict_both["extremity"]
        assert "p" in dict_
        assert np.abs(dict_["p"] - -107.69115512018216) <= self.tol_one
        assert "q" in dict_
        assert np.abs(dict_["q"] - 9.230658220781127) <= self.tol_one
        assert "v" in dict_
        assert np.abs(dict_["v"] - 141.075) <= self.tol_one
        assert "bus" in dict_
        assert dict_["bus"] == 1
        assert "sub_id" in dict_
        assert dict_["sub_id"] == 1

    def test_inspect_topo(self):
        obs = self.env.helper_observation(self.env)
        dict_ = obs.state_of(substation_id=1)
        assert "topo_vect" in dict_
        assert np.all(dict_["topo_vect"] == [1, 1, 1, 1, 1, 1])
        assert "nb_bus" in dict_
        assert dict_["nb_bus"] == 1

    def test_get_obj_connect_to(self):
        dict_ = self.env.helper_observation.get_obj_connect_to(substation_id=1)
        assert 'loads_id' in dict_
        assert np.all(dict_['loads_id'] == 0)
        assert 'generators_id' in dict_
        assert np.all(dict_['generators_id'] == 0)
        assert 'lines_or_id' in dict_
        assert np.all(dict_['lines_or_id'] == [7, 8, 9])
        assert 'lines_ex_id' in dict_
        assert np.all(dict_['lines_ex_id'] == 0)
        assert 'nb_elements' in dict_
        assert dict_['nb_elements'] == 6

    def test_to_dict(self):
        dict_ = self.env.helper_observation.to_dict()
        assert dict_ == self.dict_

    def test_from_dict(self):
        res = ObservationHelper.from_dict(self.dict_)
        assert res.n_gen == self.env.helper_observation.n_gen
        assert res.n_load == self.env.helper_observation.n_load
        assert res.n_line == self.env.helper_observation.n_line
        assert np.all(res.sub_info == self.env.helper_observation.sub_info)
        assert np.all(res.load_to_subid == self.env.helper_observation.load_to_subid)
        assert np.all(res.gen_to_subid == self.env.helper_observation.gen_to_subid)
        assert np.all(res.line_or_to_subid == self.env.helper_observation.line_or_to_subid)
        assert np.all(res.line_ex_to_subid == self.env.helper_observation.line_ex_to_subid)
        assert np.all(res.load_to_sub_pos == self.env.helper_observation.load_to_sub_pos)
        assert np.all(res.gen_to_sub_pos == self.env.helper_observation.gen_to_sub_pos)
        assert np.all(res.line_or_to_sub_pos == self.env.helper_observation.line_or_to_sub_pos)
        assert np.all(res.line_ex_to_sub_pos == self.env.helper_observation.line_ex_to_sub_pos)
        assert np.all(res.load_pos_topo_vect == self.env.helper_observation.load_pos_topo_vect)
        assert np.all(res.gen_pos_topo_vect == self.env.helper_observation.gen_pos_topo_vect)
        assert np.all(res.line_or_pos_topo_vect == self.env.helper_observation.line_or_pos_topo_vect)
        assert np.all(res.line_ex_pos_topo_vect == self.env.helper_observation.line_ex_pos_topo_vect)
        assert np.all(res.observationClass == self.env.helper_observation.observationClass)

    def test_json_serializable(self):
        dict_ = self.env.helper_observation.to_dict()
        res = json.dumps(obj=dict_, indent=4, sort_keys=True)

    def test_json_loadable(self):
        dict_ = self.env.helper_observation.to_dict()
        tmp = json.dumps(obj=dict_, indent=4, sort_keys=True)
        res = ObservationHelper.from_dict(json.loads(tmp))

        assert res.n_gen == self.env.helper_observation.n_gen
        assert res.n_load == self.env.helper_observation.n_load
        assert res.n_line == self.env.helper_observation.n_line
        assert np.all(res.sub_info == self.env.helper_observation.sub_info)
        assert np.all(res.load_to_subid == self.env.helper_observation.load_to_subid)
        assert np.all(res.gen_to_subid == self.env.helper_observation.gen_to_subid)
        assert np.all(res.line_or_to_subid == self.env.helper_observation.line_or_to_subid)
        assert np.all(res.line_ex_to_subid == self.env.helper_observation.line_ex_to_subid)
        assert np.all(res.load_to_sub_pos == self.env.helper_observation.load_to_sub_pos)
        assert np.all(res.gen_to_sub_pos == self.env.helper_observation.gen_to_sub_pos)
        assert np.all(res.line_or_to_sub_pos == self.env.helper_observation.line_or_to_sub_pos)
        assert np.all(res.line_ex_to_sub_pos == self.env.helper_observation.line_ex_to_sub_pos)
        assert np.all(res.load_pos_topo_vect == self.env.helper_observation.load_pos_topo_vect)
        assert np.all(res.gen_pos_topo_vect == self.env.helper_observation.gen_pos_topo_vect)
        assert np.all(res.line_or_pos_topo_vect == self.env.helper_observation.line_or_pos_topo_vect)
        assert np.all(res.line_ex_pos_topo_vect == self.env.helper_observation.line_ex_pos_topo_vect)
        assert np.all(res.observationClass == self.env.helper_observation.observationClass)


class TestObservationHazard(unittest.TestCase):
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
        self.path_chron = os.path.join(PATH_CHRONICS, "chronics_with_hazards")
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
                               rewardClass=self.rewardClass)

    def test_1_generating_obs_withhazard(self):
        # test that helper_obs is abl to generate a valid observation
        obs = self.env.get_obs()
        assert np.all(obs.time_before_line_reconnectable == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        action = self.env.helper_action_player({})
        _ = self.env.step(action)
        obs = self.env.get_obs()
        assert np.all(obs.time_before_line_reconnectable == [0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        _ = self.env.step(action)
        obs = self.env.get_obs()
        assert np.all(obs.time_before_line_reconnectable == [0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


class TestObservationMaintenance(unittest.TestCase):
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
        self.path_chron = os.path.join(PATH_CHRONICS, "chronics_with_maintenance")
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
                               rewardClass=self.rewardClass)

    def test_1_generating_obs_withmaintenance(self):
        # test that helper_obs is abl to generate a valid observation
        obs = self.env.get_obs()
        assert np.all(obs.time_next_maintenance == np.array([ -1,  -1,  -1,  -1,   1,  -1, 276,  -1,  -1,  -1,  -1,
                                                              -1,  -1, -1,  -1,  -1,  -1,  -1,  -1,  -1]))
        assert np.all(obs.duration_next_maintenance == np.array([ 0,  0,  0,  0, 12,  0, 12,  0,  0,  0,  0,  0,  0,
                                                                  0,  0,  0,  0, 0,  0,  0]))
        action = self.env.helper_action_player({})
        _ = self.env.step(action)
        obs = self.env.get_obs()
        assert np.all(obs.time_next_maintenance == np.array([ -1,  -1,  -1,  -1,   0,  -1, 275,  -1,  -1,  -1,  -1,
                                                              -1,  -1, -1,  -1,  -1,  -1,  -1,  -1,  -1]))
        assert np.all(obs.duration_next_maintenance == np.array([ 0,  0,  0,  0, 12,  0, 12,  0,  0,  0,  0,  0,  0,
                                                                  0,  0,  0,  0, 0,  0,  0]))
        _ = self.env.step(action)
        obs = self.env.get_obs()
        assert np.all(obs.time_next_maintenance == np.array([ -1,  -1,  -1,  -1,   0,  -1, 274,  -1,  -1,  -1,  -1,
                                                              -1,  -1, -1,  -1,  -1,  -1,  -1,  -1,  -1]))
        assert np.all(obs.duration_next_maintenance == np.array([ 0,  0,  0,  0, 11,  0, 12,  0,  0,  0,  0,  0,  0,
                                                                  0,  0,  0,  0, 0,  0,  0]))

if __name__ == "__main__":
    unittest.main()