# making some test that the backned is working as expected
import os
import sys
import unittest

import numpy as np
import pdb

from helper_path_test import *

from grid2op.Exceptions import *
from grid2op.Environment import Environment
from grid2op.Backend import PandaPowerBackend
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChronicsHandler, GridStateFromFile
from grid2op.Rules import *
from grid2op.MakeEnv import make


class TestLoadingBackendFunc(unittest.TestCase):
    def setUp(self):
        # powergrid
        self.adn_backend = PandaPowerBackend()
        self.path_matpower = PATH_DATA_TEST_PP
        self.case_file = "test_case14.json"

        # data
        self.path_chron = os.path.join(PATH_CHRONICS, "chronics")
        self.chronics_handler = ChronicsHandler(chronicsClass=GridStateFromFile, path=self.path_chron)

        self.tolvect = 1e-2
        self.tol_one = 1e-5

        # force the verbose backend
        self.adn_backend.detailed_infos_for_cascading_failures = True

        # _parameters for the environment
        self.env_params = Parameters()

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

        self.env = Environment(init_grid_path=os.path.join(self.path_matpower, self.case_file),
                               backend=self.adn_backend,
                               chronics_handler=self.chronics_handler,
                               parameters=self.env_params,
                               names_chronics_to_backend=self.names_chronics_to_backend)

        self.helper_action = self.env.helper_action_env

    def test_AlwaysLegal(self):
        # build a random action acting on everything
        new_vect = np.random.randn(self.helper_action.n_load)
        new_vect2 = np.random.randn(self.helper_action.n_load)

        change_status_orig = np.random.randint(0, 2, self.helper_action.n_line).astype(np.bool)
        set_status_orig = np.random.randint(-1, 2, self.helper_action.n_line)
        set_status_orig[change_status_orig] = 0

        change_topo_vect_orig = np.random.randint(0, 2, self.helper_action.dim_topo).astype(np.bool)
        # powerline that are set to be reconnected, can't be moved to another bus
        change_topo_vect_orig[self.helper_action.line_or_pos_topo_vect[set_status_orig == 1]] = False
        change_topo_vect_orig[self.helper_action.line_ex_pos_topo_vect[set_status_orig == 1]] = False
        # powerline that are disconnected, can't be moved to the other bus
        change_topo_vect_orig[self.helper_action.line_or_pos_topo_vect[set_status_orig == -1]] = False
        change_topo_vect_orig[self.helper_action.line_ex_pos_topo_vect[set_status_orig == -1]] = False

        set_topo_vect_orig = np.random.randint(0, 3, self.helper_action.dim_topo)
        set_topo_vect_orig[change_topo_vect_orig] = 0  # don't both change and set
        # I need to make sure powerlines that are reconnected are indeed reconnected to a bus
        set_topo_vect_orig[self.helper_action.line_or_pos_topo_vect[set_status_orig == 1]] = 1
        set_topo_vect_orig[self.helper_action.line_ex_pos_topo_vect[set_status_orig == 1]] = 1
        # I need to make sure powerlines that are disconnected are not assigned to a bus
        set_topo_vect_orig[self.helper_action.line_or_pos_topo_vect[set_status_orig == -1]] = 0
        set_topo_vect_orig[self.helper_action.line_ex_pos_topo_vect[set_status_orig == -1]] = 0

        action = self.helper_action({"change_bus": change_topo_vect_orig,
                                     "set_bus": set_topo_vect_orig,
                                      "injection": {"load_p": new_vect, "load_q": new_vect2},
                                     "change_line_status": change_status_orig,
                                     "set_line_status": set_status_orig})

        # game rules
        gr = GameRules()
        assert gr.legal_action(action, self.env)

    def test_LookParam(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=np.bool)
        arr_line1[id_line] = True
        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=np.int)
        arr_line2[id_line2] = 2

        self.helper_action.legal_action = GameRules(legalActClass=LookParam).legal_action
        self.env.time_remaining_before_reconnection = np.full(shape=(self.env.backend.n_line,),
                                                              fill_value=0,
                                                              dtype=np.int)

        self.env.parameters.MAX_SUB_CHANGED = 2
        self.env.parameters.MAX_LINE_STATUS_CHANGED = 2
        _ = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                "set_bus": {"substations_id": [(id_2, arr2)]},
                                "change_line_status": arr_line1,
                                "set_line_status": arr_line2},
                               env=self.env,
                               check_legal=True)

        try:
            self.env.parameters.MAX_SUB_CHANGED = 1
            self.env.parameters.MAX_LINE_STATUS_CHANGED = 2
            _ = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                                             "set_bus": {"substations_id": [(id_2, arr2)]},
                                                             "change_line_status": arr_line1,
                                                             "set_line_status": arr_line2},
                                                            env=self.env,
                                                            check_legal=True)
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass

        try:
            self.env.parameters.MAX_SUB_CHANGED = 2
            self.env.parameters.MAX_LINE_STATUS_CHANGED = 1
            _ = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                                             "set_bus": {"substations_id": [(id_2, arr2)]},
                                                             "change_line_status": arr_line1,
                                                             "set_line_status": arr_line2},
                                                            env=self.env,
                                                            check_legal=True)
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass


        self.env.parameters.MAX_SUB_CHANGED = 1
        self.env.parameters.MAX_LINE_STATUS_CHANGED = 1
        _ = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                "set_line_status": arr_line2},
                               env=self.env,
                               check_legal=True)

    def test_PreventReconection(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=np.bool)
        arr_line1[id_line] = True
        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=np.int)
        arr_line2[id_line2] = 2

        self.helper_action.legal_action = GameRules(legalActClass=PreventReconection).legal_action
        self.env.time_remaining_before_reconnection = np.full(shape=(self.env.backend.n_line,),
                                                              fill_value=0,
                                                              dtype=np.int)

        self.env.parameters.MAX_SUB_CHANGED = 1
        self.env.parameters.MAX_LINE_STATUS_CHANGED = 2
        _ = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                "set_bus": {"substations_id": [(id_2, arr2)]},
                                "change_line_status": arr_line1,
                                "set_line_status": arr_line2},
                               env=self.env,
                               check_legal=True)


        try:
            self.env.parameters.MAX_SUB_CHANGED = 2
            self.env.parameters.MAX_LINE_STATUS_CHANGED = 1
            self.env.time_remaining_before_reconnection[id_line] = 1
            _ = self.helper_action({"change_bus": {"substations": [(id_1, arr1)]},
                                                             "set_bus": {"substations_id": [(id_2, arr2)]},
                                                             "change_line_status": arr_line1,
                                                             "set_line_status": arr_line2},
                                                            env=self.env,
                                                            check_legal=True)
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass

        self.env.time_remaining_before_reconnection = np.full(shape=(self.env.backend.n_line,),
                                                              fill_value=0,
                                                              dtype=np.int)
        self.env.parameters.MAX_SUB_CHANGED = 2
        self.env.parameters.MAX_LINE_STATUS_CHANGED = 1
        self.env.time_remaining_before_reconnection[1] = 1
        _ = self.helper_action({"change_bus": {"substations": [(id_1, arr1)]},
                                                         "set_bus": {"substations_id": [(id_2, arr2)]},
                                                         "change_line_status": arr_line1,
                                                         "set_line_status": arr_line2},
                                                        env=self.env,
                                                        check_legal=True)

    def test_linereactionnable_throw(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=np.bool)
        arr_line1[id_line] = True

        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=np.int)
        arr_line2[id_line2] = -1

        self.env.max_timestep_line_status_deactivated = 1
        self.helper_action.legal_action = GameRules(legalActClass=PreventReconection).legal_action
        self.env.time_remaining_before_reconnection = np.full(shape=(self.env.backend.n_line,),
                                                              fill_value=0,
                                                              dtype=np.int)

        # i act a first time on powerline 15
        act = self.helper_action({"set_line_status": arr_line2},
                               env=self.env,
                               check_legal=True)
        self.env.step(act)
        try:
            # i try to react on it, it should throw an IllegalAction exception.
            act = self.helper_action({"set_line_status": arr_line2},
                                     env=self.env,
                                     check_legal=True)
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass

    def test_linereactionnable_nothrow(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=np.bool)
        arr_line1[id_line] = True

        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=np.int)
        arr_line2[id_line2] = -1

        self.env.max_timestep_line_status_deactivated = 1
        self.helper_action.legal_action = GameRules(legalActClass=PreventReconection).legal_action
        self.env.time_remaining_before_reconnection = np.full(shape=(self.env.backend.n_line,),
                                                              fill_value=0,
                                                              dtype=np.int)

        # i act a first time on powerline 15
        act = self.helper_action({"set_line_status": arr_line2},
                               env=self.env,
                               check_legal=True)
        self.env.step(act)
        # i compute another time step without doing anything
        self.env.step(self.helper_action({}))

        # i try to react on it, it should NOT throw an IllegalAction exception, but
        act = self.helper_action({"set_line_status": arr_line2},
                                 env=self.env,
                                 check_legal=True)

    def test_linereactionnable_throw_longerperiod(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=np.bool)
        arr_line1[id_line] = True

        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=np.int)
        arr_line2[id_line2] = -1

        self.env.max_timestep_line_status_deactivated = 2
        self.helper_action.legal_action = GameRules(legalActClass=PreventReconection).legal_action
        self.env.time_remaining_before_reconnection = np.full(shape=(self.env.backend.n_line,),
                                                              fill_value=0,
                                                              dtype=np.int)

        # i act a first time on powerline 15
        act = self.helper_action({"set_line_status": arr_line2},
                               env=self.env,
                               check_legal=True)
        self.env.step(act)
        # i compute another time step without doing anything
        self.env.step(self.helper_action({}))

        # i try to react on it, it should throw an IllegalAction exception because we ask the environment to wait
        # at least 2 time steps
        try:
            # i try to react on it, it should throw an IllegalAction exception.
            act = self.helper_action({"set_line_status": arr_line2},
                                     env=self.env,
                                     check_legal=True)
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass

    def test_toporeactionnable_throw(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=np.bool)
        arr_line1[id_line] = True

        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=np.int)
        arr_line2[id_line2] = -1

        self.env.max_timestep_topology_deactivated = 1
        self.helper_action.legal_action = GameRules(legalActClass=PreventReconection).legal_action
        self.env.time_remaining_before_reconnection = np.full(shape=(self.env.backend.n_line,),
                                                              fill_value=0,
                                                              dtype=np.int)

        # i act a first time on powerline 15
        act = self.helper_action({"set_bus": {"substations_id": [(id_2, arr2)]}},
                                 env=self.env,
                                 check_legal=True)
        self.env.step(act)
        try:
            # i try to react on it, it should throw an IllegalAction exception.
            act = self.helper_action({"set_bus": {"substations_id": [(id_2, arr2)]}},
                                     env=self.env,
                                     check_legal=True)
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass

    def test_toporeactionnable_nothrow(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=np.bool)
        arr_line1[id_line] = True

        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=np.int)
        arr_line2[id_line2] = -1

        self.env.max_timestep_topology_deactivated = 1
        self.helper_action.legal_action = GameRules(legalActClass=PreventReconection).legal_action
        self.env.time_remaining_before_reconnection = np.full(shape=(self.env.backend.n_line,),
                                                              fill_value=0,
                                                              dtype=np.int)

        # i act a first time on powerline 15
        act = self.helper_action({"set_bus": {"substations_id": [(id_2, arr2)]}},
                                 env=self.env,
                                 check_legal=True)
        self.env.step(act)
        # i compute another time step without doing anything
        self.env.step(self.helper_action({}))

        # i try to react on it, it should NOT throw an IllegalAction exception, but
        act = self.helper_action({"set_bus": {"substations_id": [(id_2, arr2)]}},
                                 env=self.env,
                                 check_legal=True)

    def test_toporeactionnable_throw_longerperiod(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=np.bool)
        arr_line1[id_line] = True

        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=np.int)
        arr_line2[id_line2] = -1

        self.env.max_timestep_topology_deactivated = 2
        self.helper_action.legal_action = GameRules(legalActClass=PreventReconection).legal_action
        self.env.time_remaining_before_reconnection = np.full(shape=(self.env.backend.n_line,),
                                                              fill_value=0,
                                                              dtype=np.int)

        # i act a first time on powerline 15
        act = self.helper_action({"set_bus": {"substations_id": [(id_2, arr2)]}},
                                 env=self.env,
                                 check_legal=True)
        self.env.step(act)
        # i compute another time step without doing anything
        self.env.step(self.helper_action({}))

        # i try to react on it, it should throw an IllegalAction exception because we ask the environment to wait
        # at least 2 time steps
        try:
            # i try to react on it, it should throw an IllegalAction exception.
            act = self.helper_action({"set_bus": {"substations_id": [(id_2, arr2)]}},
                                     env=self.env,
                                     check_legal=True)
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass


class TestCooldown(unittest.TestCase):
    def setUp(self):
        self.env = make("case5_example", gamerules_class=DefaultRules)

    def test_cooldown_sub(self):
        act = self.env.action_space({"set_bus": {"substations_id": [(2, [1,1,2,2])]} })
        obs, *_ = self.env.step(act)
        # TODO do these kind of test with modified parameters !!!


if __name__ == "__main__":
    unittest.main()
