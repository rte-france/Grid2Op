# making some test that the backned is working as expected
import os
import sys
import unittest

import numpy as np
import pdb

# making sure test can be ran from:
# root package directory
# RL4Grid subdirectory
# RL4Grid/tests subdirectory
from helper_path_test import PATH_DATA_TEST_PP, PATH_CHRONICS

from Exceptions import *
from ChronicsHandler import ChronicsHandler, ChangeNothing, GridStateFromFile, GridStateFromFileWithForecasts, Multifolder
from ChronicsHandler import GridValue
from BackendPandaPower import PandaPowerBackend


class TestProperHandlingHazardsMaintenance(unittest.TestCase):

    def test_get_maintenance_time_1d(self):
        maintenance_time = GridValue.get_maintenance_time_1d(np.array([0 for _ in range(10)]))
        assert np.all(maintenance_time == np.array([-1  for _ in range(10)]))

        maintenance = np.array([0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0])
        maintenance_time = GridValue.get_maintenance_time_1d(maintenance)
        assert np.all(maintenance_time == np.array([5,4,3,2,1,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1]))

        maintenance = np.array([0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0])
        maintenance_time = GridValue.get_maintenance_time_1d(maintenance)
        assert np.all(maintenance_time == np.array([5,4,3,2,1,0,0,0,4,3,2,1,0,0,-1,-1,-1]))

    def test_get_maintenance_duration_1d(self):
        maintenance = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        maintenance_duration = GridValue.get_maintenance_duration_1d(maintenance)
        assert np.all(maintenance_duration == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
        maintenance = np.array([0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0])
        maintenance_duration = GridValue.get_maintenance_duration_1d(maintenance)
        assert np.all(maintenance_duration == np.array([3,3,3,3,3,3,2,1,0,0,0,0,0,0,0,0]))
        maintenance = np.array([0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0])
        maintenance_duration = GridValue.get_maintenance_duration_1d(maintenance)
        assert np.all(maintenance_duration == np.array([3,3,3,3,3,3,2,1,2,2,2,2,2,1,0,0,0]))

    def test_get_hazard_duration_1d(self):
        hazard = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        hazard_duration = GridValue.get_hazard_duration_1d(hazard)
        assert np.all(hazard_duration == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
        hazard = np.array([0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0])
        hazard_duration = GridValue.get_hazard_duration_1d(hazard)
        assert np.all(hazard_duration == np.array([0,0,0,0,0,3,2,1,0,0,0,0,0,0,0,0]))
        hazard = np.array([0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0])
        hazard_duration = GridValue.get_hazard_duration_1d(hazard)
        assert np.all(hazard_duration == np.array([0,0,0,0,0,3,2,1,0,0,0,0,2,1,0,0,0]))


class TestLoadingChronicsHandler(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(PATH_CHRONICS, "chronics")
        self.tolvect = 1e-2
        self.tol_one = 1e-5

        self.n_gen = 5
        self.n_load = 11
        self.n_lines = 20

        self.order_backend_loads = ['2_C-10.61', '3_C151.15', '4_C-9.47', '5_C201.84', '6_C-6.27', '9_C130.49',
                                    '10_C228.66', '11_C-138.89', '12_C-27.88', '13_C-13.33', '14_C63.6']
        self.order_backend_prods = ['1_G137.1', '2_G-56.47', '3_G36.31', '6_G63.29', '8_G40.43']
        self.order_backend_lines = ['1_2_1', '1_5_2', '2_3_3', '2_4_4', '2_5_5', '3_4_6', '4_5_7', '4_7_8', '4_9_9',
                                    '5_6_10', '6_11_11', '6_12_12', '6_13_13', '7_8_14', '7_9_15', '9_10_16', '9_14_17',
                                    '10_11_18', '12_13_19', '13_14_20']
        self.order_backend_subs = ['bus_1', 'bus_2', 'bus_3', 'bus_4', 'bus_5', 'bus_6', 'bus_7', 'bus_8', 'bus_9',
                                   'bus_10', 'bus_11', 'bus_12', 'bus_13', 'bus_14']

    # Cette méthode sera appelée après chaque test.
    def tearDown(self):
        pass

    def compare_vect(self, pred, true):
        return np.max(np.abs(pred- true)) <= self.tolvect

    def test_check_validity(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs)
        backend = PandaPowerBackend()
        path_matpower = PATH_DATA_TEST_PP
        case_file = "test_case14.json"
        backend.load_grid(path_matpower, case_file)
        chron_handl.check_validity(backend)

    def test_chronicsloading(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs)
        _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        vect = [18.8, 86.5, 44.5, 7.1, 10.4, 27.6, 8.1, 3.2, 5.6, 11.9, 13.6]
        assert self.compare_vect(res["injection"]['load_p'], vect)

    def test_chronicsloading_secondtimestep(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs)
        _ = chron_handl.next_time_step()  # should load the first time stamp
        _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        vect = [18.8, 85.1, 44.3, 7.1, 10.2, 27.1, 8.2, 3.2, 5.7, 11.8, 13.8]
        assert self.compare_vect(res["injection"]['load_p'], vect)

    def test_done(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs)
        for i in range(288):
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        vect = [19.0, 87.9, 44.4, 7.2, 10.4, 27.5, 8.4, 3.2, 5.7, 12.2, 13.6]
        assert self.compare_vect(res["injection"]['load_p'], vect)
        assert chron_handl.done()

    def test_stopiteration(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs)
        for i in range(288):
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        vect = [19.0, 87.9, 44.4, 7.2, 10.4, 27.5, 8.4, 3.2, 5.7, 12.2, 13.6]
        assert self.compare_vect(res["injection"]['load_p'], vect)
        try:
            res = chron_handl.next_time_step()  # should load the first time stamp
            raise RuntimeError("This should have thrown a StopIteration exception")
        except StopIteration:
            pass

    def test_name_invariant(self):
        """
        Test that the crhonics are loaded in whatever format, but the order returned is consistent with the one
        of the backend.
        :return:
        """
        path = os.path.join(PATH_CHRONICS, "chronics_reorder")
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs)
        _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        vect = [18.8, 86.5, 44.5, 7.1, 10.4, 27.6, 8.1, 3.2, 5.6, 11.9, 13.6]
        assert self.compare_vect(res["injection"]['load_p'], vect)
        for i in range(287):
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        vect = [19.0, 87.9, 44.4, 7.2, 10.4, 27.5, 8.4, 3.2, 5.7, 12.2, 13.6]
        assert self.compare_vect(res["injection"]['load_p'], vect)
        try:
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
            raise RuntimeError("This should have thrown a StopIteration exception")
        except StopIteration:
            pass


class TestLoadingChronicsHandlerWithForecast(unittest.TestCase):
    # Cette méthode sera appelée avant chaque test.
    def setUp(self):
        self.path = os.path.join(PATH_CHRONICS, "chronics_with_forecast")
        self.tolvect = 1e-2
        self.tol_one = 1e-5

        self.n_gen = 5
        self.n_load = 11
        self.n_lines = 20

        self.order_backend_loads = ['2_C-10.61', '3_C151.15', '4_C-9.47', '5_C201.84', '6_C-6.27', '9_C130.49',
                                    '10_C228.66', '11_C-138.89', '12_C-27.88', '13_C-13.33', '14_C63.6']
        self.order_backend_prods = ['1_G137.1', '2_G-56.47', '3_G36.31', '6_G63.29', '8_G40.43']
        self.order_backend_lines = ['1_2_1', '1_5_2', '2_3_3', '2_4_4', '2_5_5', '3_4_6', '4_5_7', '4_7_8', '4_9_9',
                                    '5_6_10', '6_11_11', '6_12_12', '6_13_13', '7_8_14', '7_9_15', '9_10_16', '9_14_17',
                                    '10_11_18', '12_13_19', '13_14_20']
        self.order_backend_subs = ['bus_1', 'bus_2', 'bus_3', 'bus_4', 'bus_5', 'bus_6', 'bus_7', 'bus_8', 'bus_9',
                                   'bus_10', 'bus_11', 'bus_12', 'bus_13', 'bus_14']

    # Cette méthode sera appelée après chaque test.
    def tearDown(self):
        pass

    def compare_vect(self, pred, true):
        return np.max(np.abs(pred- true)) <= self.tolvect

    def test_check_validity(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFileWithForecasts, path=self.path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs)
        backend = PandaPowerBackend()
        path_matpower = PATH_DATA_TEST_PP
        case_file = "test_case14.json"
        backend.load_grid(path_matpower, case_file)
        chron_handl.check_validity(backend)


class TestLoadingChronicsHandlerPP(unittest.TestCase):
    # Cette méthode sera appelée avant chaque test.
    def setUp(self):
        self.pathfake = os.path.join(PATH_CHRONICS, "chronics")
        self.path = os.path.join(PATH_CHRONICS, "chronics")

        self.tolvect = 1e-2
        self.tol_one = 1e-5

        self.n_gen = 5
        self.n_load = 11
        self.n_lines = 20

        self.order_backend_loads = ['load_1_0', 'load_2_1', 'load_13_2', 'load_3_3', 'load_4_4', 'load_5_5',
                                    'load_8_6', 'load_9_7', 'load_10_8', 'load_11_9', 'load_12_10']
        self.order_backend_prods = ['gen_1_0', 'gen_2_1', 'gen_5_2', 'gen_7_3', "gen_0_4"]
        self.order_backend_lines = ['0_1_0', '0_4_1', '8_9_2', '8_13_3', '9_10_4', '11_12_5', '12_13_6',
                                    '1_2_7', '1_3_8', '1_4_9', '2_3_10',
                                    '3_4_11', '5_10_12', '5_11_13', '5_12_14', '3_6_15', '3_8_16',
                                    '4_5_17', '6_7_18', '6_8_19']

        self.order_backend_subs = ['sub_0', 'sub_1', 'sub_10', 'sub_11', 'sub_12', 'sub_13', 'sub_2', 'sub_3', 'sub_4',
                                   'sub_5', 'sub_6', 'sub_7', 'sub_8', 'sub_9']

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

        self.id_chron_to_back_load = np.array([0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9])

    def compare_vect(self, pred, true):
        return np.max(np.abs(pred- true)) <= self.tolvect

    def test_check_validity(self):
        # load a "fake" chronics with name in the correct order
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.pathfake)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs,
                               self.names_chronics_to_backend)
        backend = PandaPowerBackend()
        path_matpower = PATH_DATA_TEST_PP
        case_file = "test_case14.json"
        backend.load_grid(path_matpower, case_file)
        chron_handl.check_validity(backend)

    def test_check_validity_withdiffname(self):
        #  load a real chronics with different names
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs,
                               self.names_chronics_to_backend
                               )
        backend = PandaPowerBackend()
        path_matpower = PATH_DATA_TEST_PP
        case_file = "test_case14.json"
        backend.load_grid(path_matpower, case_file)
        chron_handl.check_validity(backend)

    def test_chronicsloading(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs,
                               self.names_chronics_to_backend)
        _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        vect = np.array([18.8, 86.5, 44.5, 7.1, 10.4, 27.6, 8.1, 3.2, 5.6, 11.9, 13.6])  # what is written on the file
        backend_th = vect[self.id_chron_to_back_load]  # what should be in backend
        assert self.compare_vect(res["injection"]['load_p'], backend_th)

    def test_chronicsloading_secondtimestep(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs,
                               self.names_chronics_to_backend)
        _ = chron_handl.next_time_step()  # should load the first time stamp
        _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        vect = np.array([18.8, 85.1, 44.3, 7.1, 10.2, 27.1, 8.2, 3.2, 5.7, 11.8, 13.8])
        vect = vect[self.id_chron_to_back_load]
        assert self.compare_vect(res["injection"]['load_p'], vect)

    def test_done(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs,
                               self.names_chronics_to_backend)
        for i in range(288):
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        vect = np.array([19.0, 87.9, 44.4, 7.2, 10.4, 27.5, 8.4, 3.2, 5.7, 12.2, 13.6])
        vect = vect[self.id_chron_to_back_load]
        assert self.compare_vect(res["injection"]['load_p'], vect)
        assert chron_handl.done()

    def test_stopiteration(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs,
                               self.names_chronics_to_backend)
        for i in range(288):
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        vect = np.array([19.0, 87.9, 44.4, 7.2, 10.4, 27.5, 8.4, 3.2, 5.7, 12.2, 13.6])
        vect = vect[self.id_chron_to_back_load]
        assert self.compare_vect(res["injection"]['load_p'], vect)
        try:
            res = chron_handl.next_time_step()  # should load the first time stamp
            raise RuntimeError("This should have thrown a StopIteration exception")
        except StopIteration:
            pass

    def test_name_invariant(self):
        """
        Test that the crhonics are loaded in whatever format, but the order returned is consistent with the one
        of the backend.
        :return:
        """
        path = os.path.join(PATH_CHRONICS, "chronics_reorder")
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=path)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs,
                               self.names_chronics_to_backend)
        _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        vect = np.array([18.8, 86.5, 44.5, 7.1, 10.4, 27.6, 8.1, 3.2, 5.6, 11.9, 13.6])
        vect = vect[self.id_chron_to_back_load]
        assert self.compare_vect(res["injection"]['load_p'], vect)
        for i in range(287):
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        vect = np.array([19.0, 87.9, 44.4, 7.2, 10.4, 27.5, 8.4, 3.2, 5.7, 12.2, 13.6])
        vect = vect[self.id_chron_to_back_load]
        assert self.compare_vect(res["injection"]['load_p'], vect)
        try:
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
            raise RuntimeError("This should have thrown a StopIteration exception")
        except StopIteration:
            pass

class TestLoadingMultiFolder(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(PATH_CHRONICS, "test_multi_chronics")
        self.tolvect = 1e-2
        self.tol_one = 1e-5

        self.n_gen = 5
        self.n_load = 11
        self.n_lines = 20

        self.order_backend_loads = ['load_1_0', 'load_2_1', 'load_13_2', 'load_3_3', 'load_4_4', 'load_5_5',
                                    'load_8_6', 'load_9_7', 'load_10_8', 'load_11_9', 'load_12_10']
        self.order_backend_prods = ['gen_1_0', 'gen_2_1', 'gen_5_2', 'gen_7_3', "gen_0_4"]
        self.order_backend_lines = ['0_1_0', '0_4_1', '8_9_2', '8_13_3', '9_10_4', '11_12_5', '12_13_6',
                                    '1_2_7', '1_3_8', '1_4_9', '2_3_10',
                                    '3_4_11', '5_10_12', '5_11_13', '5_12_14', '3_6_15', '3_8_16',
                                    '4_5_17', '6_7_18', '6_8_19']

        self.order_backend_subs = ['sub_0', 'sub_1', 'sub_10', 'sub_11', 'sub_12', 'sub_13', 'sub_2', 'sub_3', 'sub_4',
                                   'sub_5', 'sub_6', 'sub_7', 'sub_8', 'sub_9']

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

        self.max_iter = 10

    # Cette méthode sera appelée après chaque test.
    def tearDown(self):
        pass

    def test_stopiteration(self):
        chron_handl = ChronicsHandler(chronicsClass=Multifolder,
                                      path=self.path,
                                      gridvalueClass=GridStateFromFileWithForecasts,
                                      max_iter=self.max_iter)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                               self.order_backend_lines, self.order_backend_subs,
                               self.names_chronics_to_backend)
        _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        for i in range(self.max_iter ):
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp

        try:
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
            raise RuntimeError("This should have thrown a StopIteration exception")
        except StopIteration:
            pass


if __name__ == "__main__":
    unittest.main()


