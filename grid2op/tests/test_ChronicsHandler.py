# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import warnings
import pandas as pd
import tempfile
from grid2op.tests.helper_path_test import *

from grid2op.dtypes import dt_int, dt_float
from grid2op.MakeEnv import make
from grid2op.Exceptions import *
from grid2op.Chronics import ChronicsHandler, GridStateFromFile, GridStateFromFileWithForecasts, Multifolder, GridValue
from grid2op.Chronics import MultifolderWithCache
from grid2op.Backend import PandaPowerBackend
from grid2op.Parameters import Parameters
from grid2op.Rules import AlwaysLegal
from grid2op.Chronics import GridStateFromFileWithForecastsWithoutMaintenance

import warnings
warnings.simplefilter("error")


class TestProperHandlingHazardsMaintenance(HelperTests):
    def setUp(self):
        self.path_hazard = os.path.join(PATH_CHRONICS, "chronics_with_hazards")
        self.path_maintenance = os.path.join(PATH_CHRONICS, "chronics_with_maintenance")

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

    def test_get_maintenance_time_1d(self):
        maintenance_time = GridValue.get_maintenance_time_1d(np.array([0 for _ in range(10)]))
        assert np.all(maintenance_time == np.array([-1  for _ in range(10)]))

        maintenance = np.array([0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0])
        maintenance_time = GridValue.get_maintenance_time_1d(maintenance)
        assert np.all(maintenance_time == np.array([5,4,3,2,1,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1]))

        maintenance = np.array([0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0])
        maintenance_time = GridValue.get_maintenance_time_1d(maintenance)
        assert np.all(maintenance_time == np.array([5,4,3,2,1,0,0,0,4,3,2,1,0,0,-1,-1,-1]))

        maintenance = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1])
        maintenance_duration = GridValue.get_maintenance_time_1d(maintenance)
        assert np.all(maintenance_duration == np.array([12,11,10,9,8,7,6,5,4,3,2,1,0,0,0,0,0]))

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

        maintenance = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1])
        maintenance_duration = GridValue.get_maintenance_duration_1d(maintenance)
        assert np.all(maintenance_duration == np.array([5,5,5,5,5,5,5,5,5,5,5,5,5,4,3,2,1]))

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

        hazard = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1])
        hazard_duration = GridValue.get_hazard_duration_1d(hazard)
        assert np.all(hazard_duration == np.array([0,0,0,0,0,0,0,0,0,0,0,0,5,4,3,2,1]))

    def test_loadchornics_hazard_ok(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path_hazard)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs)
        current_datetime, dict_, maintenance_time, maintenance_duration, hazard_duration, prod_v = chron_handl.next_time_step()
        assert np.all(hazard_duration == 0)

        for i in range(12):
            current_datetime, dict_, maintenance_time, maintenance_duration, hazard_duration, prod_v = chron_handl.next_time_step()
            assert np.sum(hazard_duration == 0) == 19
            assert hazard_duration[17] == 12-i, "error at iteration {}".format(i)

        current_datetime, dict_, maintenance_time, maintenance_duration, hazard_duration, prod_v = chron_handl.next_time_step()
        assert np.all(hazard_duration == 0)

    def test_loadchornics_maintenance_ok(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path_maintenance)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                     self.order_backend_lines, self.order_backend_subs)
        current_datetime, dict_, maintenance_time, maintenance_duration, hazard_duration, prod_v = chron_handl.next_time_step()

        assert np.sum(maintenance_duration == 0) == 18
        assert maintenance_duration[17] == 12, "incorrect duration of maintenance on powerline 17"
        assert maintenance_duration[19] == 12, "incorrect duration of maintenance on powerline 19"

        assert np.sum(maintenance_time == -1) == 18
        assert maintenance_time[17] == 1, "incorrect time for next maintenance on powerline 17"
        assert maintenance_time[19] == 276, "incorrect time for next maintenance on powerline 19"

        for i in range(12):
            current_datetime, dict_, maintenance_time, maintenance_duration, hazard_duration, prod_v = chron_handl.next_time_step()
            assert np.sum(maintenance_duration == 0) == 18
            assert int(maintenance_duration[17]) == int(12-i), "incorrect duration of maintenance on powerline 17 at iteration {}: it is {} and should be {}".format(i, maintenance_duration[17], int(12-i))
            assert maintenance_duration[19] == 12, "incorrect duration of maintenance on powerline 19 at iteration {}".format(i)

            assert np.sum(maintenance_time == -1) == 18
            assert maintenance_time[17] == 0, "incorrect time for next maintenance on powerline 17 at iteration {}".format(i)
            assert maintenance_time[19] == 275-i, "incorrect time for next maintenance on powerline 19 at iteration {}".format(i)

        current_datetime, dict_, maintenance_time, maintenance_duration, hazard_duration, prod_v = chron_handl.next_time_step()
        assert np.sum(maintenance_duration == 0) == 19
        assert maintenance_duration[19] == 12, "incorrect duration of maintenance on powerline 19 at finish"

        assert np.sum(maintenance_time == -1) == 19
        assert maintenance_time[19] == 263, "incorrect time for next maintenance on powerline 19 at finish"


class TestLoadingChronicsHandler(HelperTests):
    def setUp(self):
        self.path = os.path.join(PATH_CHRONICS, "chronics")

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

    def test_chronicsloading_chunk(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path, chunk_size=5)
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

    def test_chronicsloading_secondtimestep_chunksize(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path, chunk_size=1)
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


class TestLoadingChronicsHandlerWithForecast(HelperTests):
    # Cette méthode sera appelée avant chaque test.
    def setUp(self):
        self.path = os.path.join(PATH_CHRONICS, "chronics_with_forecast")

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


class TestLoadingChronicsHandlerPP(HelperTests):
    # Cette méthode sera appelée avant chaque test.
    def setUp(self):
        self.pathfake = os.path.join(PATH_CHRONICS, "chronics")
        self.path = os.path.join(PATH_CHRONICS, "chronics")

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

    def test_done_chunk_size(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path, chunk_size=1)
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

    def test_stopiteration_chunk_size(self):
        chron_handl = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path, chunk_size=1)
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


class TestLoadingMultiFolder(HelperTests):
    def setUp(self):
        self.path = os.path.join(PATH_CHRONICS, "test_multi_chronics")

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
        for i in range(self.max_iter):
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp

        try:
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
            raise RuntimeError("This should have thrown a StopIteration exception")
        except StopIteration:
            pass

    def test_stopiteration_chunksize(self):
        chron_handl = ChronicsHandler(chronicsClass=Multifolder,
                                      path=self.path,
                                      gridvalueClass=GridStateFromFileWithForecasts,
                                      max_iter=self.max_iter,
                                      chunk_size=5)
        chron_handl.initialize(self.order_backend_loads, self.order_backend_prods,
                               self.order_backend_lines, self.order_backend_subs,
                               self.names_chronics_to_backend)
        _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
        for i in range(self.max_iter):
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp

        try:
            _, res, *_ = chron_handl.next_time_step()  # should load the first time stamp
            raise RuntimeError("This should have thrown a StopIteration exception")
        except StopIteration:
            pass


class TestEnvChunk(HelperTests):
    def setUp(self):
        self.max_iter = 10
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("rte_case14_realistic", test=True)
            self.env.chronics_handler.set_max_iter(self.max_iter)

    def tearDown(self):
        self.env.close()

    def test_normal(self):
        # self.env.set_chunk_size()
        self.env.reset()
        i = 0
        done = False
        while not done:
            obs, reward, done, info = self.env.step(self.env.action_space())
            i += 1
        assert i == self.max_iter  # I used 1 data to intialize the environment

    def test_normal_chunck(self):
        self.env.set_chunk_size(1)
        self.env.reset()
        i = 0
        done = False
        while not done:
            obs, reward, done, info = self.env.step(self.env.action_space())
            i += 1
        assert i == self.max_iter  # I used 1 data to intialize the environment


class TestMissingData(HelperTests):
    def test_load_error(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with self.assertRaises(EnvError):
                with make("rte_case14_realistic", test=True, chronics_path="/answer/life/42"):
                    pass

    def run_env_till_over(self, env, max_iter):
        nb_it = 0
        done = False
        obs = None
        while not done:
            obs, reward, done, info = env.step(env.action_space())
            nb_it += 1
        # assert i == max_iter
        return nb_it, obs

    def test_load_still(self):
        max_iter = 10
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True,
                          chronics_path=os.path.join(PATH_CHRONICS, "5bus_example_some_missing", "chronics")) \
                    as env:
                # test a first time without chunks
                env.set_id(0)
                env.chronics_handler.set_max_iter(max_iter)
                obs = env.reset()

                # check that simulate is working
                simul_obs, *_ = obs.simulate(env.action_space())
                # check that load_p is indeed modified
                assert np.all(simul_obs.load_p == env.chronics_handler.real_data.data.load_p_forecast[1, :])

                # check that the environment goes till the end
                nb_it, final_obs = self.run_env_till_over(env, max_iter)
                assert nb_it == max_iter
                # check the that the "missing" files is properly handled: data is not modified
                assert np.all(obs.load_q == final_obs.load_q)
                # check that other data are modified properly
                assert np.any(obs.prod_p != final_obs.prod_p)

                # test a second time with chunk
                env.set_id(0)
                env.set_chunk_size(3)
                obs = env.reset()
                nb_it, obs = self.run_env_till_over(env, max_iter)
                assert nb_it == max_iter
                pass


class TestCFFWFWM(HelperTests):
    def test_load(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(os.path.join(PATH_DATA_TEST, "ieee118_R2subgrid_wcci_test_maintenance"), param=param) as env:
                env.seed(123456)  # for reproducible tests !
                obs = env.reset()
                #get input data, to check they were correctly applied in
                linesPossiblyInMaintenance = env.chronics_handler.real_data.data.line_to_maintenance
                assert np.all(np.array(sorted(linesPossiblyInMaintenance)) ==
                              ['11_12_13', '12_13_14', '16_18_23', '16_21_27', '22_26_39', '26_30_56',
                               '2_3_0', '30_31_45', '7_9_9', '9_16_18'])
                ChronicMonth = env.chronics_handler.real_data.data.start_datetime.month
                assert ChronicMonth == 8
                maxMaintenancePerDay = env.chronics_handler.real_data.data.max_daily_number_per_month_maintenance[(ChronicMonth-1)]
                assert maxMaintenancePerDay == 2

                envLines = env.name_line
                idx_linesPossiblyInMaintenance = [i for i in range(len(envLines)) if envLines[i] in linesPossiblyInMaintenance]
                idx_linesNotInMaintenance = [i for i in range(len(envLines)) if envLines[i] not in linesPossiblyInMaintenance]
                
                 
                #maintenance dataFrame
                maintenanceChronic = maintenances_df = pd.DataFrame(env.chronics_handler.real_data.data.maintenance,
                                                                    columns=envLines)
                nb_timesteps = maintenanceChronic.shape[0]
                # identify the timestamps of the chronics to find out the month and day of the week
                freq = str(int(env.chronics_handler.real_data.data.time_interval.total_seconds())) + "s"  # should be in the timedelta frequency format in pandas
                datelist = pd.date_range(env.chronics_handler.real_data.data.start_datetime,
                                         periods=nb_timesteps,
                                         freq=freq)
               
                maintenances_df.index = datelist
                assert (maintenanceChronic[envLines[idx_linesNotInMaintenance]].sum().sum() == 0)

                assert (maintenanceChronic[linesPossiblyInMaintenance].sum().sum() >= 1)

                nb_mainteance_timestep = maintenanceChronic.sum(axis=1)
                assert np.all(nb_mainteance_timestep <= maxMaintenancePerDay)

    def test_maintenance_multiple_timesteps(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(os.path.join(PATH_DATA_TEST, "ieee118_R2subgrid_wcci_test_maintenance"),
                      param=param) as env:
                env.seed(0)
                envLines = env.name_line
                linesPossiblyInMaintenance = np.array(['11_12_13', '12_13_14', '16_18_23', '16_21_27', '22_26_39',
                                                       '26_30_56', '2_3_0', '30_31_45', '7_9_9', '9_16_18'])
                idx_linesPossiblyInMaintenance=[i for i in range(len(envLines))
                                                if envLines[i] in linesPossiblyInMaintenance]
                idx_linesNotInMaintenance = [i for i in range(len(envLines)) if
                                             envLines[i] not in linesPossiblyInMaintenance]
                maxMaintenancePerDay = 2

                obs = env.reset()
                ####check that at least one line that can go in maintenance actuamlly goes in maintenance
                assert (np.sum(obs.duration_next_maintenance[idx_linesPossiblyInMaintenance]) >= 1)
                ####check that at no line that can not go in maintenance actually goes in maintenance
                assert (np.sum(obs.duration_next_maintenance[idx_linesNotInMaintenance]) == 0)

                done = False
                max_it = 10
                it_num = 0
                while not done:
                    if np.any(obs.time_next_maintenance > 0):
                        timestep_nextMaintenance = np.min(obs.time_next_maintenance[obs.time_next_maintenance!=-1])
                    else:
                        break
                    env.fast_forward_chronics(timestep_nextMaintenance)
                    obs, reward, done, info = env.step(env.action_space())
                    assert np.sum(obs.time_next_maintenance == 0) <= maxMaintenancePerDay
                    assert np.all(np.abs(obs.a_or[obs.time_next_maintenance == 0] - 0.) <= self.tol_one)
                    assert np.all(~obs.line_status[obs.time_next_maintenance == 0])
                    it_num += 1
                    if it_num >= max_it:
                        break

    def test_proba(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(os.path.join(PATH_DATA_TEST, "ieee118_R2subgrid_wcci_test_maintenance_2"),
                      param=param) as env:
                env.seed(0)
                # input data
                nb_scenario = 30   # if too low then i don't have 1e-3 beteween theory and practice
                nb_line_in_maintenance = 10
                assert len(env.chronics_handler.real_data.data.line_to_maintenance) == nb_line_in_maintenance
                proba = 0.06

                nb_th = proba * 5/7  # for day of week
                nb_th *= 8/24  # maintenance only between 9 and 17

                nb_maintenance = np.zeros(env.n_line, dtype=dt_float)
                nb_ts_ = 0
                for i in range(nb_scenario):
                    obs = env.reset()
                    nb_maintenance += np.sum(env.chronics_handler.real_data.data.maintenance, axis=0)
                    nb_ts_ += env.chronics_handler.real_data.data.maintenance.shape[0]
                total_maintenance = np.sum(nb_maintenance)
                total_maintenance /= nb_ts_ * nb_line_in_maintenance
                assert np.abs(total_maintenance - nb_th) <= 1e-3

    def test_load_fake_january(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(os.path.join(PATH_DATA_TEST, "ieee118_R2subgrid_wcci_test_maintenance_3"),
                      param=param) as env:
                env.seed(0)
                # get input data, to check they were correctly applied in
                linesPossiblyInMaintenance = env.chronics_handler.real_data.data.line_to_maintenance
                assert np.all(np.array(sorted(linesPossiblyInMaintenance)) ==
                              ['11_12_13', '12_13_14', '16_18_23', '16_21_27', '22_26_39', '26_30_56',
                               '2_3_0', '30_31_45', '7_9_9', '9_16_18'])
                ChronicMonth = env.chronics_handler.real_data.data.start_datetime.month
                assert ChronicMonth == 1
                maxMaintenancePerDay = env.chronics_handler.real_data.data.max_daily_number_per_month_maintenance[
                    (ChronicMonth - 1)]
                assert maxMaintenancePerDay == 0
                assert np.sum(env.chronics_handler.real_data.data.maintenance) == 0


    def test_split_and_save(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(os.path.join(PATH_DATA_TEST, "ieee118_R2subgrid_wcci_test_maintenance"), param=param) as env:
                env.seed(0)
                env.set_id(0)
                obs = env.reset()
                start_d = {
                    "Scenario_august_00": "2012-08-10 00:00"
                }
                end_d = {
                    "Scenario_august_00": "2012-08-19 00:00"
                }
                ts_beg = 2591
                ts_end = 5184

                # generate some reference data
                chronics_outdir = tempfile.mkdtemp()
                env.chronics_handler.real_data.split_and_save(start_d,
                                                              end_d,
                                                              chronics_outdir)
                maintenance_0_0 = pd.read_csv(os.path.join(chronics_outdir,
                                                         "Scenario_august_00",
                                                         "maintenance.csv.bz2"),
                                            sep=";").values

                # test that i got a different result with a different seed
                env.seed(1)
                env.set_id(0)
                obs = env.reset()
                chronics_outdir2 = tempfile.mkdtemp()
                env.chronics_handler.real_data.split_and_save(start_d,
                                                              end_d,
                                                              chronics_outdir2)
                maintenance_1 = pd.read_csv(os.path.join(chronics_outdir2,
                                                         "Scenario_august_00",
                                                         "maintenance.csv.bz2"),
                                            sep=";").values
                assert np.any(maintenance_0_0 != maintenance_1)

                # and now test that i have the same results with the same seed
                env.seed(0)
                env.set_id(0)
                obs = env.reset()
                chronics_outdir3 = tempfile.mkdtemp()
                env.chronics_handler.real_data.split_and_save(start_d,
                                                              end_d,
                                                              chronics_outdir3)
                maintenance_0_1 = pd.read_csv(os.path.join(chronics_outdir3,
                                                         "Scenario_august_00",
                                                         "maintenance.csv.bz2"),
                                              sep=";").values
                assert np.all(maintenance_0_0 == maintenance_0_1)

                # make sure i can reload the environment
                env2 = make(os.path.join(PATH_DATA_TEST, "ieee118_R2subgrid_wcci_test_maintenance"),
                            param=param,
                            data_feeding_kwargs={"gridvalueClass": GridStateFromFileWithForecasts},
                            chronics_path=chronics_outdir3)
                env2.set_id(0)
                obs = env2.reset()
                # and that it has the right chroncis
                assert np.all(env2.chronics_handler.real_data.data.maintenance.astype(dt_float) ==
                              maintenance_0_0)

    def test_seed(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(os.path.join(PATH_DATA_TEST, "ieee118_R2subgrid_wcci_test_maintenance"), param=param) as env:
                nb_scenario = 10
                nb_maintenance = np.zeros((nb_scenario, env.n_line), dtype=dt_float)
                nb_maintenance1 = np.zeros((nb_scenario, env.n_line), dtype=dt_float)

                env.seed(0)
                for i in range(nb_scenario):
                    obs = env.reset()
                    nb_maintenance[i, :] = np.sum(env.chronics_handler.real_data.data.maintenance, axis=0)

                env.seed(0)
                for i in range(nb_scenario):
                    obs = env.reset()
                    nb_maintenance1[i, :] = np.sum(env.chronics_handler.real_data.data.maintenance, axis=0)
                assert np.all(nb_maintenance == nb_maintenance)

    def test_chunk_size(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(os.path.join(PATH_DATA_TEST, "ieee118_R2subgrid_wcci_test_maintenance_3"),
                      param=param) as env:
                env.seed(0)
                obs = env.reset()
                maint = env.chronics_handler.real_data.data.maintenance

                env.seed(0)
                env.set_chunk_size(10)
                obs = env.reset()
                maint2 = env.chronics_handler.real_data.data.maintenance
                assert np.all(maint == maint2)


class TestWithCache(HelperTests):
    def test_load(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(os.path.join(PATH_DATA_TEST, "5bus_example_some_missing"),
                      param=param,
                      chronics_class=MultifolderWithCache) as env:
                env.seed(123456)  # for reproducible tests !
                nb_steps = 10
                # I test that the reset ... reset also the date time and the chronics "state"
                obs = env.reset()
                assert obs.minute_of_hour == 5
                assert env.chronics_handler.real_data.data.current_index == 0
                assert env.chronics_handler.real_data.data.curr_iter == 1
                for i in range(nb_steps):
                    obs, reward, done, info = env.step(env.action_space())
                assert obs.minute_of_hour == 55
                obs = env.reset()
                assert obs.minute_of_hour == 5
                assert env.chronics_handler.real_data.data.current_index == 0
                assert env.chronics_handler.real_data.data.curr_iter == 1


class TestMaintenanceBehavingNormally(HelperTests):
    def test_withrealistic(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(os.path.join(PATH_CHRONICS, "env_14_test_maintenance"),
                      test=True,
                      param=param) as env:
                l_id = 11
                obs = env.reset()
                assert np.all(obs.time_before_cooldown_line == 0)
                obs, reward, done, info = env.step(env.action_space())
                assert np.all(obs.time_before_cooldown_line ==
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int))
                obs, reward, done, info = env.step(env.action_space({"set_line_status": [(11, 1)]}))
                assert info["is_illegal"]
                assert not obs.line_status[l_id]
                assert np.all(obs.time_before_cooldown_line ==
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int))

                obs, reward, done, info = env.step(env.action_space({"set_bus": {"lines_or_id": [(11, 1)]}}))
                assert info["is_illegal"] is True  # it is legal -> no more because of cooldown (and because now this action is counted as an action on a line)
                assert not obs.line_status[l_id]  # but has no effect (line is still disconnected)

                assert np.all(obs.time_before_cooldown_line ==
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int))
                for i in range(10):
                    obs, reward, done, info = env.step(env.action_space())
                    arr_ = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int)
                    arr_[l_id] = 9-i
                    assert np.all(obs.time_before_cooldown_line == arr_), "error at time step {}".format(i)
                # i have no more cooldown on line now
                assert np.all(obs.time_before_cooldown_line == 0)

                obs, reward, done, info = env.step(env.action_space({"set_bus": {"lines_or_id": [(11, 1)]}}))
                assert info["is_illegal"] is False  # it is legal
                assert obs.line_status[l_id]  #  reconnecting only one end has still no effect -> change of behaviour

                obs, reward, done, info = env.step(env.action_space({"set_line_status": [(11, 1)]}))
                assert info["is_illegal"] is False  # it's legal

                assert obs.line_status[l_id]  # it has reconnected the powerline
                assert np.all(obs.time_before_cooldown_line ==
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int))
                # nothing is in maintenance

                for i in range(obs.time_next_maintenance[12]-1):
                    obs, reward, done, info = env.step(env.action_space())
                # no maintenance yet
                assert np.all(obs.time_before_cooldown_line ==
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int))
                # a maintenance now
                obs, reward, done, info = env.step(env.action_space())
                assert np.all(obs.time_before_cooldown_line ==
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int))

    def test_with_alwayslegal(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(os.path.join(PATH_CHRONICS, "env_14_test_maintenance"),
                      test=True,
                      param=param,
                      gamerules_class=AlwaysLegal) as env:
                l_id = 11
                obs = env.reset()
                assert np.all(obs.time_before_cooldown_line == 0)
                obs, reward, done, info = env.step(env.action_space())
                assert np.all(obs.time_before_cooldown_line ==
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int))
                obs, reward, done, info = env.step(env.action_space({"set_line_status": [(11, 1)]}))
                assert not info["is_illegal"]  # it is legal
                assert not obs.line_status[l_id]  # yet maintenance should have stayed
                assert np.all(obs.time_before_cooldown_line ==
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int))

                obs, reward, done, info = env.step(env.action_space({"set_bus": {"lines_or_id": [(11, 1)]}}))
                assert not info["is_illegal"]  # it is legal
                assert not obs.line_status[l_id]  # but has no effect (line is still disconnected)
                assert np.all(obs.time_before_cooldown_line ==
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int))
                for i in range(10):
                    obs, reward, done, info = env.step(env.action_space())
                    arr_ = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int)
                    arr_[l_id] = 9-i
                    assert np.all(obs.time_before_cooldown_line == arr_), "error at time step {}".format(i)

                # i have no more cooldown on line now
                assert np.all(obs.time_before_cooldown_line == 0)

                obs, reward, done, info = env.step(env.action_space({"set_bus": {"lines_or_id": [(11, 1)]}}))
                assert not info["is_illegal"]  # it is legal -> no more because of cooldown
                # assert not obs.line_status[11]  # reconnecting only one end has still no effect -> change of behaviour now

                obs, reward, done, info = env.step(env.action_space({"set_line_status": [(11, 1)]}))
                assert not info["is_illegal"]  # it is legal
                assert obs.line_status[l_id]  # it has reconnected the powerline
                assert np.all(obs.time_before_cooldown_line ==
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int))
                # nothing is in maintenance

                for i in range(obs.time_next_maintenance[12]-1):
                    obs, reward, done, info = env.step(env.action_space())
                # no maintenance yet
                assert np.all(obs.time_before_cooldown_line ==
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int))
                # a maintenance now
                obs, reward, done, info = env.step(env.action_space())
                assert np.all(obs.time_before_cooldown_line ==
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0], dtype=dt_int))


class TestMultiFolder(HelperTests):
    def get_multifolder_class(self):
        return Multifolder

    def _reset_chron_handl(self, chronics_handler):
        pass

    def setUp(self) -> None:
        chronics_class = self.get_multifolder_class()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("rte_case14_realistic", test=True, chronics_class=chronics_class)
        root_path = self.env.chronics_handler.real_data.path
        self.chronics_paths = np.array([os.path.join(root_path, '000'),
                                        os.path.join(root_path, '001')])
        self._reset_chron_handl(self.env.chronics_handler)
        self.env.seed(0)

    def test_real_data(self):
        real_data = self.env.chronics_handler.real_data

    def test_filter(self):
        class MyFilterForTest:
            def __init__(self):
                self.i = -1

            def __call__(self, path):
                self.i += 1
                return self.i == 0

        assert np.all(self.env.chronics_handler.real_data.subpaths == self.chronics_paths)
        assert np.all(self.env.chronics_handler.real_data._order == [0, 1])
        my_filt = MyFilterForTest()
        self.env.chronics_handler.set_filter(my_filt)
        obs = self.env.chronics_handler.reset()
        assert np.all(self.env.chronics_handler.real_data.subpaths == self.chronics_paths)
        assert np.all(self.env.chronics_handler.real_data._order == [0])

    def tearDown(self) -> None:
        self.env.close()

    def test_the_tests(self):
        assert isinstance(self.env.chronics_handler.real_data, Multifolder)

    def test_shuffler(self):
        def shuffler(list):
            return list[[1, 0]]

        # without shuffling it's read this way
        assert np.all(self.env.chronics_handler.real_data.subpaths == self.chronics_paths)
        assert np.all(self.env.chronics_handler.real_data._order == [0, 1])

        # i check that shuffling is done properly
        self.env.chronics_handler.shuffle(shuffler)
        assert np.all(self.env.chronics_handler.real_data.subpaths == self.chronics_paths)
        assert np.all(self.env.chronics_handler.real_data._order == [1, 0])

        # i check that the env is working
        obs = self.env.reset()
        # at first chronics is 0, now it's 1
        # and i have put the subpaths 0 at the element 1 of the order
        assert self.env.chronics_handler.real_data.data.path == self.chronics_paths[0]

    def test_get_id(self):
        assert self.env.chronics_handler.get_id() == self.chronics_paths[0]
        obs = self.env.reset()
        assert self.env.chronics_handler.get_id() == self.chronics_paths[1]

    def test_set_id(self):
        self.env.set_id(0)
        obs = self.env.reset()
        assert self.env.chronics_handler.get_id() == self.chronics_paths[0]
        def shuffler(list):
            return list[[1, 0]]
        self.env.chronics_handler.shuffle(shuffler)
        obs = self.env.reset()
        assert self.env.chronics_handler.get_id() == self.chronics_paths[0]
        obs = self.env.reset()
        assert self.env.chronics_handler.get_id() == self.chronics_paths[1]
        obs = self.env.reset()
        assert self.env.chronics_handler.get_id() == self.chronics_paths[0]
        self.env.set_id(0)
        obs = self.env.reset()
        assert self.env.chronics_handler.get_id() == self.chronics_paths[1]

    def test_reset(self):
        assert self.env.chronics_handler.get_id() == self.chronics_paths[0]
        self.env.chronics_handler.real_data.reset()
        assert np.all(self.env.chronics_handler.real_data.subpaths == self.chronics_paths)
        assert np.all(self.env.chronics_handler.real_data._order == [0, 1])

        def shuffler(list):
            return list[[1, 0]]
        self.env.chronics_handler.real_data.reset()
        self.env.chronics_handler.shuffle(shuffler)
        assert np.all(self.env.chronics_handler.real_data.subpaths == self.chronics_paths)
        assert np.all(self.env.chronics_handler.real_data._order == [1, 0])
        obs = self.env.reset()
        assert self.env.chronics_handler.get_id() == self.chronics_paths[0]

    def test_sample_next_chronics(self):
        res_th = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
        for i in range(20):
            assert self.env.chronics_handler.sample_next_chronics([0.9, 0.1]) == res_th[i], "error for iteration {}" \
                                                                                            "".format(i)
        # reseed to test it's working accordingly
        self.env.seed(0)
        for i in range(20):
            assert self.env.chronics_handler.sample_next_chronics([0.9, 0.1]) == res_th[i], "error for iteration {}" \
                                                                                            "".format(i)

    def test_sample_next_chronics_withfilter(self):
        chronics_class = self.get_multifolder_class()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("rte_case5_example", test=True, chronics_class=chronics_class)
        self._reset_chron_handl(env.chronics_handler)
        env.seed(0)

        # test than the filtering
        class MyFilterForTest:
            def __init__(self):
                self.i = -1

            def __call__(self, path):
                self.i += 1
                return self.i % 2 == 0
        root_path = env.chronics_handler.real_data.path
        chronics_paths = np.array([os.path.join(root_path, "{:02d}".format(i)) for i in range(20)])
        # if i don't do anything, then everything is normal
        assert np.all(env.chronics_handler.real_data.subpaths == chronics_paths)
        assert np.all(env.chronics_handler.real_data._order == [i for i in range(20)])
        # now i will apply a filter and check it has the correct behaviour
        my_filt = MyFilterForTest()
        env.chronics_handler.set_filter(my_filt)
        obs = env.chronics_handler.reset()
        # check that reset do what need to be done
        assert np.all(env.chronics_handler.real_data.subpaths == chronics_paths)
        assert np.all(env.chronics_handler.real_data._order == [2*i for i in range(10)])
        # now check the id is correct
        id_ = self.env.chronics_handler.sample_next_chronics([0.9, 0.1])
        assert id_ == 0
        assert np.all(env.chronics_handler.real_data._order == [2*i for i in range(10)])


class TestMultiFolderWithCache(TestMultiFolder):
    def get_multifolder_class(self):
        return MultifolderWithCache

    def test_the_tests(self):
        assert isinstance(self.env.chronics_handler.real_data, MultifolderWithCache)

    def _reset_chron_handl(self, chronics_handler):
        # by default when using the cache, only the first data is kept...
        # I make sure to keep everything
        chronics_handler.set_filter(lambda x: True)
        chronics_handler.reset()


class TestDeactivateMaintenance(HelperTests):
    def test_maintenance_deactivated(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(os.path.join(PATH_CHRONICS, "env_14_test_maintenance"),
                      test=True,
                      param=param) as env:
                obs = env.reset()
                # there is a maintenance by default for some powerlines
                assert np.any(obs.time_next_maintenance != -1)

            with make(os.path.join(PATH_CHRONICS, "env_14_test_maintenance"),
                      test=True,
                      param=param,
                      data_feeding_kwargs={"gridvalueClass": GridStateFromFileWithForecastsWithoutMaintenance}
                      ) as env:
                obs = env.reset()
                # all maintenance are deactivated
                assert np.all(obs.time_next_maintenance == -1)


if __name__ == "__main__":
    unittest.main()
