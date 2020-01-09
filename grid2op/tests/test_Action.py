# making some test that the backned is working as expected
import os
import sys
import unittest
import json
import numpy as np
import pdb

import helper_path_test  # usefull to set poperly the sys.path

from Exceptions import *
from Action import HelperAction, Action
from GameRules import GameRules
from Space import GridObjects

from grid2op.MakeEnv import make

# TODO test that "twice change" is reset to normal. when i update an action twice, nothing is done.
# TODO test for all class of Action

# TODO clean the test to have it for all class of actions without recoding everything each time


class TestLoadingBackendFunc(unittest.TestCase):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.game_rules = GameRules()
        self.gridobj = GridObjects()
        self.gridobj.init_grid_vect(name_prod=["gen_{}".format(i) for i in range(5)],
                                          name_load=["load_{}".format(i) for i in range(11)],
                                          name_line=["line_{}".format(i) for i in range(20)],
                                          name_sub=["sub_{}".format(i) for i in range(14)],
                                          sub_info=np.array([3, 6, 4, 6, 5, 6, 3, 2, 5, 3, 3, 3, 4, 3], dtype=np.int),
                                          load_to_subid=np.array([1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 13]),
                                          gen_to_subid=np.array([0, 1, 2, 5, 7]),
                                          line_or_to_subid=np.array([ 0,  0,  1,  1,  1,  2,  3,  3,  3,  4,  5,  5,
                                                                       5,  6,  6,  8,  8, 9, 11, 12]),
                                          line_ex_to_subid=np.array([ 1,  4,  2,  3,  4,  3,  4,  6,  8,  5, 10, 11,
                                                                       12,  7,  8,  9, 13, 10, 12, 13]),  #####
                                          load_to_sub_pos=np.array([4, 2, 5, 4, 4, 4, 1, 1, 1, 2, 1]),
                                          gen_to_sub_pos=np.array([2, 5, 3, 5, 1]),
                                          line_or_to_sub_pos=np.array([0, 1, 1, 2, 3, 1, 2, 3, 4, 3, 1, 2, 3, 1, 2, 2,
                                                                        3, 0, 0, 1]),
                                          line_ex_to_sub_pos=np.array([0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 3, 0, 1, 2, 2, 0, 0, 0]),  #####
                                          load_pos_topo_vect=np.array([ 7, 11, 18, 23, 28, 39, 41, 44, 47, 51, 54]),
                                          gen_pos_topo_vect=np.array([ 2,  8, 12, 29, 34]),
                                          line_or_pos_topo_vect=np.array([ 0,  1,  4,  5,  6, 10, 15, 16, 17, 22, 25, 26, 27, 31, 32, 37, 38, 40, 46, 50]),
                                          line_ex_pos_topo_vect=np.array([ 3, 19,  9, 13, 20, 14, 21, 30, 35, 24, 45, 48, 52, 33, 36, 42, 55, 43, 49, 53]))
        # pdb.set_trace()
        self.helper_action = HelperAction(self.gridobj,
                                          game_rules=self.game_rules)


        self.helper_action_env = HelperAction(self.gridobj,
                                              game_rules=self.game_rules,
                                              actionClass=Action)

        self.res = {'name_gen': ['gen_0', 'gen_1', 'gen_2', 'gen_3', 'gen_4'],
               'name_load': ['load_0', 'load_1', 'load_2', 'load_3', 'load_4', 'load_5', 'load_6',
                             'load_7', 'load_8', 'load_9', 'load_10'],
               'name_line': ['line_0', 'line_1', 'line_2', 'line_3', 'line_4', 'line_5', 'line_6', 'line_7',
                             'line_8', 'line_9', 'line_10', 'line_11', 'line_12', 'line_13', 'line_14', 'line_15',
                             'line_16', 'line_17', 'line_18', 'line_19'],
                'name_sub': ["sub_0", "sub_1", "sub_2", "sub_3", "sub_4", "sub_5", "sub_6", "sub_7", "sub_8", "sub_9",
                             "sub_10", "sub_11", "sub_12", "sub_13"],
               'sub_info': [3, 6, 4, 6, 5, 6, 3, 2, 5, 3, 3, 3, 4, 3],
               'load_to_subid': [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13],
               'gen_to_subid': [0, 1, 2, 5, 7],
               'line_or_to_subid': [0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 8, 8, 9, 11, 12],
               'line_ex_to_subid': [1, 4, 2, 3, 4, 3, 4, 6, 8, 5, 10, 11, 12, 7, 8, 9, 13, 10, 12, 13],
               'load_to_sub_pos': [4, 2, 5, 4, 4, 4, 1, 1, 1, 2, 1], 'gen_to_sub_pos': [2, 5, 3, 5, 1],
               'line_or_to_sub_pos': [0, 1, 1, 2, 3, 1, 2, 3, 4, 3, 1, 2, 3, 1, 2, 2, 3, 0, 0, 1],
               'line_ex_to_sub_pos': [0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 3, 0, 1, 2, 2, 0, 0, 0],
               'load_pos_topo_vect': [7, 11, 18, 23, 28, 39, 41, 44, 47, 51, 54],
               'gen_pos_topo_vect': [2, 8, 12, 29, 34],
               'line_or_pos_topo_vect': [0, 1, 4, 5, 6, 10, 15, 16, 17, 22, 25, 26, 27, 31, 32, 37, 38, 40, 46, 50],
               'line_ex_pos_topo_vect': [3, 19, 9, 13, 20, 14, 21, 30, 35, 24, 45, 48, 52, 33, 36, 42, 55, 43, 49, 53],
               'subtype': 'Action.Action'}

    def tearDown(self):
        pass
        # self.backend._grid.delete()

    def compare_vect(self, pred, true):
        return np.max(np.abs(pred - true)) <= self.tolvect

    def test_instanciate_action(self):
        """
        test i can instanciate an action without crashing
        :return:
        """
        action = self.helper_action()

    def test_size(self):
        action = self.helper_action()
        action.size()

    def test_proper_size(self):
        action = self.helper_action()
        assert action.size() == 224

    def test_size_action_space(self):
        assert self.helper_action.size() == 224

    def test_print_notcrash(self):
        """
        test the conversion to str does not crash
        :return:
        """
        action = self.helper_action({})
        a = "{}".format(action)

    def test_change_p(self):
        """

        :return:
        """
        new_vect = np.random.randn(self.helper_action.n_load)
        action = self.helper_action({"injection": {"load_p": new_vect}})
        self.compare_vect(action._dict_inj["load_p"], new_vect)
        for i in range(self.helper_action.n_load):
            assert action.effect_on(load_id=i)["new_p"] == new_vect[i]

    def test_change_v(self):
        """

        :return:
        """
        new_vect = np.random.randn(self.helper_action.n_gen)
        action = self.helper_action({"injection": {"prod_v": new_vect}})
        self.compare_vect(action._dict_inj["prod_v"], new_vect)
        for i in range(self.helper_action.n_gen):
            assert action.effect_on(gen_id=i)["new_v"] == new_vect[i]

    def test_change_p_q(self):
        """

        :return:
        """
        new_vect = np.random.randn(self.helper_action.n_load)
        new_vect2 = np.random.randn(self.helper_action.n_load)
        action = self.helper_action({"injection": {"load_p": new_vect, "load_q": new_vect2}})
        assert self.compare_vect(action._dict_inj["load_p"], new_vect)
        assert self.compare_vect(action._dict_inj["load_q"], new_vect2)
        for i in range(self.helper_action.n_load):
            assert action.effect_on(load_id=i)["new_p"] == new_vect[i]
            assert action.effect_on(load_id=i)["new_q"] == new_vect2[i]

    def test_update_disconnection_1(self):
        """
        Test if the disconnection is working properly
        :return:
        """
        for i in range(self.helper_action.n_line):
            disco = np.full(shape=self.helper_action.n_line, fill_value=0, dtype=np.int)
            disco[i] = 1
            action = self.helper_action({"set_line_status": disco})
            for j in range(self.helper_action.n_line):
                assert action.effect_on(line_id=j)["set_line_status"] == disco[j], "problem with line {} if line {} is disconnected".format(j, i)
                assert action.effect_on(line_id=j)["change_line_status"] == False

    def test_update_disconnection_m1(self):
        """
        Test if the disconnection is working properly
        :return:
        """
        for i in range(self.helper_action.n_line):
            disco = np.full(shape=self.helper_action.n_line, fill_value=0, dtype=np.int)
            disco[i] = -1
            action = self.helper_action({"set_line_status": disco})
            for j in range(self.helper_action.n_line):
                assert action.effect_on(line_id=j)["set_line_status"] == disco[j], "problem with line {} if line {} is disconnected".format(j, i)
                assert action.effect_on(line_id=j)["change_line_status"] == False

    def test_update_hazard(self):
        """
        Same test as above, but with hazard
        :return:
        """
        for i in range(self.helper_action.n_line):
            disco = np.full(shape=self.helper_action.n_line, fill_value=False, dtype=np.bool)
            disco[i] = True
            action = self.helper_action({"hazards": disco})
            for j in range(self.helper_action.n_line):
                expected_res = -1 if j == i else 0
                assert action.effect_on(line_id=j)["set_line_status"] == expected_res, "problem with line {} if line {} is disconnected".format(j, i)
                assert action.effect_on(line_id=j)["change_line_status"] == False

    def test_update_status(self):
        for i in range(self.helper_action.n_line):
            disco = np.full(shape=self.helper_action.n_line, fill_value=False, dtype=np.bool)
            disco[i] = True
            action = self.helper_action({"change_line_status": disco})
            for j in range(self.helper_action.n_line):
                expected_res = j == i
                assert action.effect_on(line_id=j)["set_line_status"] == 0
                assert action.effect_on(line_id=j)["change_line_status"] == expected_res

    def test_update_set_topo_by_dict_obj(self):
        action = self.helper_action({"set_bus": {"loads_id": [(1, 3)]}})
        assert action.effect_on(load_id=1)["set_bus"] == 3
        assert action.effect_on(load_id=1)["change_bus"] == False
        assert action.effect_on(load_id=0)["set_bus"] == 0
        assert action.effect_on(load_id=0)["change_bus"] == False

    def test_update_set_topo_by_dict_sub(self):
        arr = np.array([1, 1, 1, 2, 2, 2], dtype=np.int)
        action = self.helper_action({"set_bus": {"substations_id": [(1, arr)]}})
        assert action.effect_on(line_id=2)["set_bus_or"] == 1
        assert action.effect_on(line_id=3)["set_bus_or"] == 1
        assert action.effect_on(line_id=4)["set_bus_or"] == 2
        assert action.effect_on(line_id=0)["set_bus_ex"] == 1
        assert action.effect_on(load_id=0)["set_bus"] == 2
        assert action.effect_on(gen_id=1)["set_bus"] == 2

        assert action.effect_on(load_id=1)["set_bus"] == 0
        assert action.effect_on(gen_id=0)["set_bus"] == 0

    def test_update_set_topo_by_dict_sub2(self):
        arr = np.array([1, 1, 1, 2, 2, 2], dtype=np.int)
        arr3 = np.array([1, 2, 1, 2, 1, 2], dtype=np.int)
        action = self.helper_action({"set_bus": {"substations_id": [(3, arr3), (1, arr)]}})
        assert action.effect_on(line_id=2)["set_bus_or"] == 1
        assert action.effect_on(line_id=3)["set_bus_or"] == 1
        assert action.effect_on(line_id=4)["set_bus_or"] == 2
        assert action.effect_on(line_id=0)["set_bus_ex"] == 1
        assert action.effect_on(load_id=0)["set_bus"] == 2
        assert action.effect_on(gen_id=1)["set_bus"] == 2

        assert action.effect_on(load_id=1)["set_bus"] == 0
        assert action.effect_on(gen_id=0)["set_bus"] == 0

    def test_update_change_bus_by_dict_obj(self):
        action = self.helper_action({"change_bus": {"loads_id": [1]}})
        assert action.effect_on(load_id=1)["set_bus"] == 0
        assert action.effect_on(load_id=1)["change_bus"] == True
        assert action.effect_on(load_id=0)["set_bus"] == 0
        assert action.effect_on(load_id=0)["change_bus"] == False

    def test_update_change_bus_by_dict_sub(self):
        arr = np.array([True, True, True, False, False, False], dtype=np.bool)
        action = self.helper_action({"change_bus": {"substations_id": [(1, arr)]}})
        assert action.effect_on(line_id=2)["change_bus_or"] == True
        assert action.effect_on(line_id=3)["change_bus_or"] == True
        assert action.effect_on(line_id=4)["change_bus_or"] == False
        assert action.effect_on(line_id=0)["change_bus_ex"] == True
        assert action.effect_on(load_id=0)["change_bus"] == False
        assert action.effect_on(gen_id=1)["change_bus"] == False

        assert action.effect_on(load_id=1)["change_bus"] == False
        assert action.effect_on(gen_id=0)["change_bus"] == False

    def test_update_change_bus_by_dict_sub2(self):
        arr = np.array([True, True, True, False, False, False], dtype=np.bool)
        arr3 = np.array([True, False, True, False, True, False], dtype=np.bool)
        action = self.helper_action({"change_bus": {"substations_id": [(3, arr3), (1, arr)]}})
        assert action.effect_on(line_id=2)["change_bus_or"] == True
        assert action.effect_on(line_id=3)["change_bus_or"] == True
        assert action.effect_on(line_id=4)["change_bus_or"] == False
        assert action.effect_on(line_id=0)["change_bus_ex"] == True
        assert action.effect_on(load_id=0)["change_bus"] == False
        assert action.effect_on(gen_id=1)["change_bus"] == False

        assert action.effect_on(load_id=1)["change_bus"] == False
        assert action.effect_on(gen_id=0)["change_bus"] == False

    def test_ambiguity_topo(self):
        action = self.helper_action({"change_bus": {"lines_or_id": [1]}})  # i switch the bus of the origin of powerline 1
        action.update({"set_bus": {"lines_or_id": [(1,1)]}})  # i set the origin of powerline 1 to bus 1
        try:
            action()
            raise RuntimeError("This should hav thrown an InvalidBusStatus error")
        except InvalidBusStatus as e:
            pass

    def test_ambiguity_line_status_when_set_and_change(self):
        arr = np.zeros(self.helper_action.n_line)
        arr[1] = -1
        action = self.helper_action({"set_line_status": arr})  # i switch set the status of powerline 1 to "disconnected"
        action.update({"change_line_status": [1]})  # i asked to change this status
        try:
            action()
            raise RuntimeError("This should hav thrown an InvalidBusStatus error")
        except InvalidLineStatus as e:
            pass

    def test_ambiguity_line_reconnected_without_bus(self):
        arr = np.zeros(self.helper_action.n_line)
        arr[1] = 1
        action = self.helper_action({"set_line_status": arr})  # i switch set the status of powerline 1 to "connected"
        # and i don't say on which bus to connect it
        try:
            action()
            raise RuntimeError("This should have thrown an InvalidBusStatus error")
        except InvalidLineStatus as e:
            pass

    def test_set_status_and_setbus_isambiguous(self):
        """

        :return:
        """
        arr = np.array([1, 1, 1, 2, 2, 2], dtype=np.int)
        id_=2
        action = self.helper_action({"set_bus": {"substations_id": [(1, arr)]}})
        arr2 = np.zeros(self.helper_action.n_line)
        arr2[id_] = -1
        action.update({"set_line_status": arr2})
        try:
            action()
            raise RuntimeError("This should have thrown an InvalidBusStatus error")
        except InvalidLineStatus as e:
            pass

    def test_hazard_overides_setbus(self):
        """

        :return:
        """
        arr = np.array([1, 1, 1, 2, 2, 2], dtype=np.int)
        id_ = 2
        action = self.helper_action({"set_bus": {"substations_id": [(1, arr)]}})
        assert action.effect_on(line_id=id_)["set_bus_or"] == 1
        action.update({"hazards": [id_]})
        assert action.effect_on(line_id=id_)["set_bus_or"] == 0
        assert action.effect_on(line_id=id_)["set_line_status"] == -1
        assert action.effect_on(line_id=id_)["set_bus_ex"] == 0

    def test_action_str(self):
        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        id_1 = 1
        id_2 = 12
        action = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                     "set_bus": {"substations_id": [(id_2, arr2)]}})
        res = action.__str__()
        act_str = 'This action will:\n\t - NOT change anything to the injections\n\t - NOT force any line status\n\t - NOT switch any line status\n\t - Change the bus of the following element:\n\t \t - switch bus of line (origin) 4 [on substation 1]\n\t \t - switch bus of load 0 [on substation 1]\n\t \t - switch bus of generator 1 [on substation 1]\n\t - Set the bus of the following element:\n\t \t - assign bus 1 to line (extremity) 18 [on substation 12]\n\t \t - assign bus 1 to line (origin) 19 [on substation 12]\n\t \t - assign bus 2 to load 9 [on substation 12]\n\t \t - assign bus 2 to line (extremity) 12 [on substation 12]'
        assert res == act_str

    def test_to_vect(self):
        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        id_1 = 1
        id_2 = 12
        action = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                     "set_bus": {"substations_id": [(id_2, arr2)]}})
        res = action.to_vect()
        tmp = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
                        np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
                        np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  1.,  1.,  2.,  2.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.])
        assert np.all(res[np.isfinite(tmp)] == tmp[np.isfinite(tmp)])
        assert np.all(np.isfinite(res) == np.isfinite(tmp))

    def test__eq__(self):
        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        id_1 = 1
        id_2 = 12
        action1 = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                     "set_bus": {"substations_id": [(id_2, arr2)]}})
        action2 = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                     "set_bus": {"substations_id": [(id_2, arr2)]}})
        action3 = self.helper_action()
        assert action1 == action2
        assert action1 != action3

    def test_from_vect(self):
        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        id_1 = 1
        id_2 = 12
        action1 = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                     "set_bus": {"substations_id": [(id_2, arr2)]}})
        action2 = self.helper_action({})

        vect_act1 = action1.to_vect()
        action2.from_vect(vect_act1)
        # if i load an action with from_vect it's equal to the original one
        assert action1 == action2
        vect_act2 = action2.to_vect()

        # if i convert it back to a vector, it's equal to the original converted vector
        assert np.all(vect_act1[np.isfinite(vect_act2)] == vect_act2[np.isfinite(vect_act2)])
        assert np.all(np.isfinite(vect_act1) == np.isfinite(vect_act2))

    def test_call_change_set(self):
        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        id_1 = 1
        id_2 = 12
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
        dict_injection, set_status, change_status, set_topo_vect, switcth_topo_vect = action()
        assert "load_p" in dict_injection
        assert np.all(dict_injection["load_p"] == new_vect)
        assert "load_q" in dict_injection
        assert np.all(dict_injection["load_q"] == new_vect2)

        assert np.all(set_status == set_status_orig)
        assert np.all(change_status == change_status_orig)
        assert np.all(set_topo_vect == set_topo_vect_orig)
        assert np.all(switcth_topo_vect == change_topo_vect_orig)

    def test_get_topological_impact(self):
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

        do_nothing = self.helper_action({})
        aff_lines, aff_subs = do_nothing.get_topological_impact()
        assert np.sum(aff_lines) == 0
        assert np.sum(aff_subs) == 0

        act_sub1 = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]}})
        aff_lines, aff_subs = act_sub1.get_topological_impact()
        assert np.sum(aff_lines) == 0
        assert np.sum(aff_subs) == 1
        assert aff_subs[id_1]

        act_sub1_sub12 = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                             "set_bus": {"substations_id": [(id_2, arr2)]}})
        aff_lines, aff_subs = act_sub1_sub12.get_topological_impact()
        assert np.sum(aff_lines) == 0
        assert np.sum(aff_subs) == 2
        assert aff_subs[id_1]
        assert aff_subs[id_2]

        act_sub1_sub12_line1 = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                                   "set_bus": {"substations_id": [(id_2, arr2)]},
                                                   "change_line_status": arr_line1})
        aff_lines, aff_subs = act_sub1_sub12_line1.get_topological_impact()
        assert np.sum(aff_lines) == 1
        assert aff_lines[id_line] == 1
        assert np.sum(aff_subs) == 2
        assert aff_subs[id_1]
        assert aff_subs[id_2]

        act_sub1_sub12_line1_line2 = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                                   "set_bus": {"substations_id": [(id_2, arr2)]},
                                                   "change_line_status": arr_line1,
                                                   "set_line_status": arr_line2})
        aff_lines, aff_subs = act_sub1_sub12_line1_line2.get_topological_impact()
        assert np.sum(aff_lines) == 2
        assert aff_lines[id_line] == 1
        assert aff_lines[id_line2] == 1
        assert np.sum(aff_subs) == 2
        assert aff_subs[id_1]
        assert aff_subs[id_2]

    def test_to_dict(self):
        dict_ = self.helper_action.to_dict()
        assert dict_ == self.res

    def test_from_dict(self):
        res = HelperAction.from_dict(self.res )
        assert np.all(res.name_gen == self.helper_action.name_gen)
        assert np.all(res.name_load == self.helper_action.name_load)
        assert np.all(res.name_line == self.helper_action.name_line)
        assert np.all(res.sub_info == self.helper_action.sub_info)
        assert np.all(res.load_to_subid == self.helper_action.load_to_subid)
        assert np.all(res.gen_to_subid == self.helper_action.gen_to_subid)
        assert np.all(res.line_or_to_subid == self.helper_action.line_or_to_subid)
        assert np.all(res.line_ex_to_subid == self.helper_action.line_ex_to_subid)
        assert np.all(res.load_to_sub_pos == self.helper_action.load_to_sub_pos)
        assert np.all(res.gen_to_sub_pos == self.helper_action.gen_to_sub_pos)
        assert np.all(res.line_or_to_sub_pos == self.helper_action.line_or_to_sub_pos)
        assert np.all(res.line_ex_to_sub_pos == self.helper_action.line_ex_to_sub_pos)
        assert np.all(res.load_pos_topo_vect == self.helper_action.load_pos_topo_vect)
        assert np.all(res.gen_pos_topo_vect == self.helper_action.gen_pos_topo_vect)
        assert np.all(res.line_or_pos_topo_vect == self.helper_action.line_or_pos_topo_vect)
        assert np.all(res.line_ex_pos_topo_vect == self.helper_action.line_ex_pos_topo_vect)
        # pdb.set_trace()
        assert np.all(res.actionClass == self.helper_action.actionClass)

    def test_json_serializable(self):
        dict_ = self.helper_action.to_dict()
        res = json.dumps(obj=dict_, indent=4, sort_keys=True)

    def test_json_loadable(self):
        dict_ = self.helper_action.to_dict()
        tmp = json.dumps(obj=dict_, indent=4, sort_keys=True)
        res = HelperAction.from_dict(json.loads(tmp))

        assert np.all(res.name_gen == self.helper_action.name_gen)
        assert np.all(res.name_load == self.helper_action.name_load)
        assert np.all(res.name_line == self.helper_action.name_line)
        assert np.all(res.sub_info == self.helper_action.sub_info)
        assert np.all(res.load_to_subid == self.helper_action.load_to_subid)
        assert np.all(res.gen_to_subid == self.helper_action.gen_to_subid)
        assert np.all(res.line_or_to_subid == self.helper_action.line_or_to_subid)
        assert np.all(res.line_ex_to_subid == self.helper_action.line_ex_to_subid)
        assert np.all(res.load_to_sub_pos == self.helper_action.load_to_sub_pos)
        assert np.all(res.gen_to_sub_pos == self.helper_action.gen_to_sub_pos)
        assert np.all(res.line_or_to_sub_pos == self.helper_action.line_or_to_sub_pos)
        assert np.all(res.line_ex_to_sub_pos == self.helper_action.line_ex_to_sub_pos)
        assert np.all(res.load_pos_topo_vect == self.helper_action.load_pos_topo_vect)
        assert np.all(res.gen_pos_topo_vect == self.helper_action.gen_pos_topo_vect)
        assert np.all(res.line_or_pos_topo_vect == self.helper_action.line_or_pos_topo_vect)
        assert np.all(res.line_ex_pos_topo_vect == self.helper_action.line_ex_pos_topo_vect)
        assert np.all(res.actionClass == self.helper_action.actionClass)

    def test_as_dict(self):
        act = self.helper_action_env({})
        dict_ = act.as_dict()
        assert dict_ == {}

    def test_to_from_vect_action(self):
        act = self.helper_action_env({})
        vect_ = act.to_vect()
        act2 = self.helper_action_env.from_vect(vect_)
        assert act == act2

    def test_sum_shape_equal_size(self):
        act = self.helper_action_env({})
        assert act.size() == np.sum(act.shape())

    def test_shape_correct(self):
        act = self.helper_action_env({})
        assert act.shape().shape == act.dtype().shape

if __name__ == "__main__":
    unittest.main()
