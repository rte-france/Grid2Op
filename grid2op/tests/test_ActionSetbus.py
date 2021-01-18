# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import re

from grid2op.tests.helper_path_test import *

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import *
from grid2op.Action import *
from grid2op.Rules import RulesChecker
from grid2op.Space.space_utils import save_to_dict
from grid2op.tests.test_Action import _get_action_grid_class
import pdb


class TestSetBus(unittest.TestCase):
    """test the property to set the bus of the action"""

    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.game_rules = RulesChecker()

        GridObjects_cls, self.res = _get_action_grid_class()
        self.gridobj = GridObjects_cls()
        self.n_line = self.gridobj.n_line


        # self.size_act = 229
        self.ActionSpaceClass = ActionSpace.init_grid(self.gridobj)
        # self.helper_action = ActionSpace(self.gridobj, legal_action=self.game_rules.legal_action)
        self.helper_action = self.ActionSpaceClass(self.gridobj,
                                                   legal_action=self.game_rules.legal_action,
                                                   actionClass=CompleteAction)
        self.helper_action.seed(42)
        # save_to_dict(self.res, self.helper_action, "subtype", lambda x: re.sub("(<class ')|('>)", "", "{}".format(x)))
        save_to_dict(self.res, self.helper_action,
                     "_init_subtype",
                     lambda x: re.sub("(<class ')|(\\.init_grid\\.<locals>\\.res)|('>)", "", "{}".format(x)))

        self.authorized_keys = self.helper_action().authorized_keys
        self.size_act = self.helper_action.size()

    def test_load_set_bus_array(self):
        li_orig = [1, 2, -1, 2, 1, 0, 2, 1, 0, 1, 2]  # because i have 11 loads
        tmp = np.array(li_orig)

        # first set of tests, with numpy array
        act = self.helper_action()
        act.load_set_bus = tmp  # ok
        assert np.all(act.load_set_bus == tmp)

        # array too short
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            act.load_set_bus = tmp[:-1]
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # array too big
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp2 = np.concatenate((tmp, (1,)))
            act.load_set_bus = tmp2
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # float vect
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp3 = np.array(li_orig).astype(dt_float)
            act.load_set_bus = tmp3
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # one of the value too small
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp4 = np.array(li_orig)
            tmp4[2] = -2
            act.load_set_bus = tmp4
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # one of the value too large
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp5 = np.array(li_orig)
            tmp5[2] = 3
            act.load_set_bus = tmp5
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # wrong type
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp6 = np.array(li_orig).astype(str)
            tmp6[2] = "toto"
            act.load_set_bus = tmp6
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

    def test_load_set_bus_tuple(self):
        # second set of tests, with tuple
        act = self.helper_action()
        act.load_set_bus = (1, 1)
        assert np.all(act.load_set_bus == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (3.0, 1)
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (False, 1)
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = ("toto", 1)
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (1, "toto")
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (11, 1)
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (-1, 1)
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # not enough element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (1, )
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # too much element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (1, 2, 3)
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

    def test_load_set_bus_list_asarray(self):
        """test the set attribute when list are given (list convertible to array)"""
        li_orig = [1, 2, -1, 2, 1, 0, 2, 1, 0, 1, 2]  # because i have 11 loads
        tmp = np.array(li_orig)

        # ok
        act = self.helper_action()
        act.load_set_bus = li_orig
        assert np.all(act.load_set_bus == tmp)

        # list too short
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp0 = copy.deepcopy(li_orig)
            tmp0.pop(0)
            act.load_set_bus = tmp0
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp1 = copy.deepcopy(li_orig)
            tmp1.append(2)
            act.load_set_bus = tmp1
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [float(el) for el in li_orig]
            act.load_set_bus = tmp3
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[2] = -2
            act.load_set_bus = tmp4
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[2] = 3
            act.load_set_bus = tmp5
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = [str(el) for el in li_orig]
            tmp6[2] = "toto"
            act.load_set_bus = tmp6
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

    def test_load_set_bus_list_oftuple(self):
        """test the set attribute when list are given (list of tuple)"""
        li_orig = [(0, 1), (2, -1), (5, 2)]
        # ok
        act = self.helper_action()
        act.load_set_bus = li_orig
        assert np.all(act.load_set_bus == [1, 0, -1, 0, 0, 2, 0, 0, 0, 0, 0])

        # list of float (for the el_id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [(float(id_), new_bus) for id_, new_bus in li_orig]
            act.load_set_bus = tmp3
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[2] = (3, -2)
            act.load_set_bus = tmp4
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[2] = (3, 3)
            act.load_set_bus = tmp5
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(li_orig)
            tmp6[2] = ("toto", 1)
            act.load_set_bus = tmp6
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(li_orig)
            tmp7[2]= (3, "toto")
            act.load_set_bus = tmp7
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(li_orig)
            tmp8.append((11, 1))
            act.load_set_bus = tmp8
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(li_orig)
            tmp9.append((-1, 1))
            act.load_set_bus = tmp9
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

        # last test, when we give a list of tuple of exactly the right size
        act = self.helper_action()
        act.load_set_bus = [(el, 2) for el in range(act.n_load)]
        assert np.all(act.load_set_bus == 2)

    def test_load_set_bus_dict_with_id(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {0: 1, 2: -1, 5: 2}
        # ok
        act = self.helper_action()
        act.load_set_bus = dict_orig
        assert np.all(act.load_set_bus == [1, 0, -1, 0, 0, 2, 0, 0, 0, 0, 0])

        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = {float(id_): new_bus for id_, new_bus in dict_orig.items()}
            act.load_set_bus = tmp3
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(dict_orig)
            tmp4[2] = -2
            act.load_set_bus = tmp4
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(dict_orig)
            tmp5[2] = 3
            act.load_set_bus = tmp5
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1
            act.load_set_bus = tmp6
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(dict_orig)
            tmp7[3] = "tata"
            act.load_set_bus = tmp7
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(dict_orig)
            tmp8[11] = 1
            act.load_set_bus = tmp8
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(dict_orig)
            tmp9[-1] = 1
            act.load_set_bus = tmp9
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

    def test_load_set_bus_dict_with_name(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {"load_0": 1, "load_2": -1, "load_5": 2}
        # ok
        act = self.helper_action()
        act.load_set_bus = dict_orig
        assert np.all(act.load_set_bus == [1, 0, -1, 0, 0, 2, 0, 0, 0, 0, 0])

        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1  # unknown load
            act.load_set_bus = tmp6
        assert np.all(act.load_set_bus == 0), "a load has been modified by an illegal action"

    def test_gen_set_bus_array(self):
        li_orig = [1, 2, -1, 2, 1]  # because i have 5 gens
        tmp = np.array(li_orig)

        # first set of tests, with numpy array
        act = self.helper_action()
        act.gen_set_bus = tmp  # ok
        assert np.all(act.gen_set_bus == tmp)

        # array too short
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            act.gen_set_bus = tmp[:-1]
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # array too big
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp2 = np.concatenate((tmp, (1,)))
            act.gen_set_bus = tmp2
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # float vect
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp3 = np.array(li_orig).astype(dt_float)
            act.gen_set_bus = tmp3
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # one of the value too small
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp4 = np.array(li_orig)
            tmp4[2] = -2
            act.gen_set_bus = tmp4
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # one of the value too large
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp5 = np.array(li_orig)
            tmp5[2] = 3
            act.gen_set_bus = tmp5
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # wrong type
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp6 = np.array(li_orig).astype(str)
            tmp6[2] = "toto"
            act.gen_set_bus = tmp6
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

    def test_gen_set_bus_tuple(self):
        # second set of tests, with tuple
        act = self.helper_action()
        act.gen_set_bus = (1, 1)
        assert np.all(act.gen_set_bus == [0, 1, 0, 0, 0])

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (3.0, 1)
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (False, 1)
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = ("toto", 1)
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (1, "toto")
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (6, 1)
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (-1, 1)
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # not enough element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (1, )
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # too much element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (1, 2, 3)
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

    def test_gen_set_bus_list_asarray(self):
        """test the set attribute when list are given (list convertible to array)"""
        li_orig = [1, 2, -1, 2, 1]  # because i have 5 gens
        tmp = np.array(li_orig)

        # ok
        act = self.helper_action()
        act.gen_set_bus = li_orig
        assert np.all(act.gen_set_bus == tmp)

        # list too short
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp0 = copy.deepcopy(li_orig)
            tmp0.pop(0)
            act.gen_set_bus = tmp0
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp1 = copy.deepcopy(li_orig)
            tmp1.append(2)
            act.gen_set_bus = tmp1
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [float(el) for el in li_orig]
            act.gen_set_bus = tmp3
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[2] = -2
            act.gen_set_bus = tmp4
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[2] = 3
            act.gen_set_bus = tmp5
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = [str(el) for el in li_orig]
            tmp6[2] = "toto"
            act.gen_set_bus = tmp6
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

    def test_gen_set_bus_list_oftuple(self):
        """test the set attribute when list are given (list of tuple)"""
        li_orig = [(0, 1), (2, -1), (4, 2)]
        # ok
        act = self.helper_action()
        act.gen_set_bus = li_orig
        assert np.all(act.gen_set_bus == [1, 0, -1, 0, 2])

        # list of float (for the el_id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [(float(id_), new_bus) for id_, new_bus in li_orig]
            act.gen_set_bus = tmp3
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[2] = (3, -2)
            act.gen_set_bus = tmp4
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[2] = (3, 3)
            act.gen_set_bus = tmp5
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(li_orig)
            tmp6[2] = ("toto", 1)
            act.gen_set_bus = tmp6
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(li_orig)
            tmp7[2] = (3, "toto")
            act.gen_set_bus = tmp7
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(li_orig)
            tmp8.append((5, 1))
            act.gen_set_bus = tmp8
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(li_orig)
            tmp9.append((-1, 1))
            act.gen_set_bus = tmp9
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

        # when the list has exactly the same size
        act = self.helper_action()
        act.gen_set_bus = [(el, 2) for el in range(act.n_gen)]
        assert np.all(act.gen_set_bus == 2)

    def test_gen_set_bus_dict_with_id(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {0: 1, 2: -1, 4: 2}
        # ok
        act = self.helper_action()
        act.gen_set_bus = dict_orig
        assert np.all(act.gen_set_bus == [1, 0, -1, 0, 2])

        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = {float(id_): new_bus for id_, new_bus in dict_orig.items()}
            act.gen_set_bus = tmp3
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(dict_orig)
            tmp4[2] = -2
            act.gen_set_bus = tmp4
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(dict_orig)
            tmp5[2] = 3
            act.gen_set_bus = tmp5
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1
            act.gen_set_bus = tmp6
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(dict_orig)
            tmp7[3] = "tata"
            act.gen_set_bus = tmp7
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(dict_orig)
            tmp8[11] = 1
            act.gen_set_bus = tmp8
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(dict_orig)
            tmp9[-1] = 1
            act.gen_set_bus = tmp9
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

    def test_gen_set_bus_dict_with_name(self):
        """test the set attribute when dict are given with key = names"""
        dict_orig = {"gen_0": 1, "gen_2": -1, "gen_4": 2}
        # ok
        act = self.helper_action()
        act.gen_set_bus = dict_orig
        assert np.all(act.gen_set_bus == [1, 0, -1, 0, 2])

        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1  # unknown gen
            act.gen_set_bus = tmp6
        assert np.all(act.gen_set_bus == 0), "a gen has been modified by an illegal action"

    def test_storage_set_bus_array(self):
        li_orig = [1, 2]  # because i have 2 loads
        tmp = np.array(li_orig)

        # first set of tests, with numpy array
        act = self.helper_action()
        act.storage_set_bus = tmp  # ok
        assert np.all(act.storage_set_bus == tmp)

        # array too short
        act = self.helper_action()

        with self.assertRaises(IllegalAction):
            act.storage_set_bus = tmp[0]
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # array too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp2 = np.concatenate((tmp, (1,)))
            act.storage_set_bus = tmp2
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # float vect
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = np.array(li_orig).astype(dt_float)
            act.storage_set_bus = tmp3
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = np.array(li_orig)
            tmp4[1] = -2
            act.storage_set_bus = tmp4
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = np.array(li_orig)
            tmp5[1] = 3
            act.storage_set_bus = tmp5
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = np.array(li_orig).astype(str)
            tmp6[1] = "toto"
            act.storage_set_bus = tmp6
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

    def test_storage_set_bus_tuple(self):
        # second set of tests, with tuple
        act = self.helper_action()
        act.storage_set_bus = (1, 1)
        assert np.all(act.storage_set_bus == [0, 1])

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (1.0, 1)
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (False, 1)
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = ("toto", 1)
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (1, "toto")
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (11, 1)
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (-1, 1)
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # not enough element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (1, )
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # too much element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (1, 2, 3)
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

    def test_storage_set_bus_list_asarray(self):
        """test the set attribute when list are given (list convertible to array)"""
        li_orig = [1, 2]  # because i have 2 storage unit
        tmp = np.array(li_orig)

        # ok
        act = self.helper_action()
        act.storage_set_bus = li_orig
        assert np.all(act.storage_set_bus == tmp)

        # list too short
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp0 = copy.deepcopy(li_orig)
            tmp0.pop(0)
            act.storage_set_bus = tmp0
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp1 = copy.deepcopy(li_orig)
            tmp1.append(2)
            act.storage_set_bus = tmp1
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [float(el) for el in li_orig]
            act.storage_set_bus = tmp3
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = -2
            act.storage_set_bus = tmp4
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = 3
            act.storage_set_bus = tmp5
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = [str(el) for el in li_orig]
            tmp6[1] = "toto"
            act.storage_set_bus = tmp6
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

    def test_storage_set_bus_list_oftuple(self):
        """test the set attribute when list are given (list of tuple)"""
        li_orig = [(0, 1), (1, 2)]
        # ok
        act = self.helper_action()
        act.storage_set_bus = li_orig
        assert np.all(act.storage_set_bus == [1, 2])

        # list of float (for the el_id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [(float(id_), new_bus) for id_, new_bus in li_orig]
            act.storage_set_bus = tmp3
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = (1, -2)
            act.storage_set_bus = tmp4
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = (1, 3)
            act.storage_set_bus = tmp5
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(li_orig)
            tmp6[1] = ("toto", 1)
            act.storage_set_bus = tmp6
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(li_orig)
            tmp7[1] = (3, "toto")
            act.storage_set_bus = tmp7
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(li_orig)
            tmp8.append((2, 1))
            act.storage_set_bus = tmp8
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(li_orig)
            tmp9.append((-1, 1))
            act.storage_set_bus = tmp9
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

        # last test, when we give a list of tuple of exactly the right size
        act = self.helper_action()
        act.storage_set_bus = [(el, 2) for el in range(act.n_storage)]
        assert np.all(act.storage_set_bus == 2)

    def test_storage_set_bus_dict_with_id(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {0: 1}
        # ok
        act = self.helper_action()
        act.storage_set_bus = dict_orig
        assert np.all(act.storage_set_bus == [1, 0])

        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = {float(id_): new_bus for id_, new_bus in dict_orig.items()}
            act.storage_set_bus = tmp3
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(dict_orig)
            tmp4[1] = -2
            act.storage_set_bus = tmp4
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(dict_orig)
            tmp5[1] = 3
            act.storage_set_bus = tmp5
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1
            act.storage_set_bus = tmp6
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(dict_orig)
            tmp7[1] = "tata"
            act.storage_set_bus = tmp7
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(dict_orig)
            tmp8[2] = 1
            act.storage_set_bus = tmp8
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(dict_orig)
            tmp9[-1] = 1
            act.storage_set_bus = tmp9
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

    def test_storage_set_bus_dict_with_name(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {"storage_0": 1}
        # ok
        act = self.helper_action()
        act.storage_set_bus = dict_orig
        assert np.all(act.storage_set_bus == [1, 0])

        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1  # unknown load
            act.storage_set_bus = tmp6
        assert np.all(act.storage_set_bus == 0), "a storage unit has been modified by an illegal action"

    def test_line_or_set_bus_array(self):
        li_orig = [1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # because i have 20 lines
        tmp = np.array(li_orig)

        # first set of tests, with numpy array
        act = self.helper_action()
        act.line_or_set_bus = tmp  # ok
        assert np.all(act.line_or_set_bus == tmp)

        # array too short
        act = self.helper_action()

        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = tmp[0]
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # array too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp2 = np.concatenate((tmp, (1,)))
            act.line_or_set_bus = tmp2
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # float vect
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = np.array(li_orig).astype(dt_float)
            act.line_or_set_bus = tmp3
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = np.array(li_orig)
            tmp4[1] = -2
            act.line_or_set_bus = tmp4
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = np.array(li_orig)
            tmp5[1] = 3
            act.line_or_set_bus = tmp5
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = np.array(li_orig).astype(str)
            tmp6[1] = "toto"
            act.line_or_set_bus = tmp6
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

    def test_line_or_set_bus_tuple(self):
        # second set of tests, with tuple
        act = self.helper_action()
        act.line_or_set_bus = (1, 1)
        assert np.all(act.line_or_set_bus == [0, 1] + [0 for _ in range(18)])

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (1.0, 1)
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (False, 1)
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = ("toto", 1)
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (1, "toto")
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (21, 1)
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (-1, 1)
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # not enough element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (1, )
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # too much element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (1, 2, 3)
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

    def test_line_or_set_bus_list_asarray(self):
        """test the set attribute when list are given (list convertible to array)"""
        li_orig = [1, 2, -1] + [0 for _ in range(17)]  # because i have 2 storage unit
        tmp = np.array(li_orig)

        # ok
        act = self.helper_action()
        act.line_or_set_bus = li_orig
        assert np.all(act.line_or_set_bus == tmp)

        # list too short
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp0 = copy.deepcopy(li_orig)
            tmp0.pop(0)
            act.line_or_set_bus = tmp0
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp1 = copy.deepcopy(li_orig)
            tmp1.append(2)
            act.line_or_set_bus = tmp1
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [float(el) for el in li_orig]
            act.line_or_set_bus = tmp3
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = -2
            act.line_or_set_bus = tmp4
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = 3
            act.line_or_set_bus = tmp5
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = [str(el) for el in li_orig]
            tmp6[1] = "toto"
            act.line_or_set_bus = tmp6
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

    def test_line_or_set_bus_list_oftuple(self):
        """test the set attribute when list are given (list of tuple)"""
        li_orig = [(0, 1), (1, 2)]
        # ok
        act = self.helper_action()
        act.line_or_set_bus = li_orig
        assert np.all(act.line_or_set_bus == [1, 2] + [0 for _ in range(18)])

        # list of float (for the el_id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [(float(id_), new_bus) for id_, new_bus in li_orig]
            act.line_or_set_bus = tmp3
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = (1, -2)
            act.line_or_set_bus = tmp4
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = (1, 3)
            act.line_or_set_bus = tmp5
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(li_orig)
            tmp6[1] = ("toto", 1)
            act.line_or_set_bus = tmp6
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(li_orig)
            tmp7[1] = (3, "toto")
            act.line_or_set_bus = tmp7
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(li_orig)
            tmp8.append((21, 1))
            act.line_or_set_bus = tmp8
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(li_orig)
            tmp9.append((-1, 1))
            act.line_or_set_bus = tmp9
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

        # last test, when we give a list of tuple of exactly the right size
        act = self.helper_action()
        act.line_or_set_bus = [(el, 2) for el in range(act.n_line)]
        assert np.all(act.line_or_set_bus == 2)

    def test_line_or_set_bus_dict_with_id(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {0: 1}
        # ok
        act = self.helper_action()
        act.line_or_set_bus = dict_orig
        assert np.all(act.line_or_set_bus == [1, 0] + [0 for _ in range(18)])

        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = {float(id_): new_bus for id_, new_bus in dict_orig.items()}
            act.line_or_set_bus = tmp3
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(dict_orig)
            tmp4[1] = -2
            act.line_or_set_bus = tmp4
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(dict_orig)
            tmp5[1] = 3
            act.line_or_set_bus = tmp5
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1
            act.line_or_set_bus = tmp6
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(dict_orig)
            tmp7[1] = "tata"
            act.line_or_set_bus = tmp7
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(dict_orig)
            tmp8[21] = 1
            act.line_or_set_bus = tmp8
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(dict_orig)
            tmp9[-1] = 1
            act.line_or_set_bus = tmp9
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

    def test_line_or_set_bus_dict_with_name(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {"line_0": 1}
        # ok
        act = self.helper_action()
        act.line_or_set_bus = dict_orig
        assert np.all(act.line_or_set_bus == [1, 0] + [0 for _ in range(18)])

        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1  # unknown load
            act.line_or_set_bus = tmp6
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"

    def test_line_ex_set_bus_array(self):
        li_orig = [1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # because i have 20 lines
        tmp = np.array(li_orig)

        # first set of tests, with numpy array
        act = self.helper_action()
        act.line_ex_set_bus = tmp  # ok
        assert np.all(act.line_ex_set_bus == tmp)

        # array too short
        act = self.helper_action()

        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = tmp[0]
        assert np.all(act.line_ex_set_bus == 0), "a line (ext) unit has been modified by an illegal action"

        # array too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp2 = np.concatenate((tmp, (1,)))
            act.line_ex_set_bus = tmp2
        assert np.all(act.line_ex_set_bus == 0), "a line (ext) unit has been modified by an illegal action"

        # float vect
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = np.array(li_orig).astype(dt_float)
            act.line_ex_set_bus = tmp3
        assert np.all(act.line_ex_set_bus == 0), "a line (ext) unit has been modified by an illegal action"

        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = np.array(li_orig)
            tmp4[1] = -2
            act.line_ex_set_bus = tmp4
        assert np.all(act.line_ex_set_bus == 0), "a line (ext) unit has been modified by an illegal action"

        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = np.array(li_orig)
            tmp5[1] = 3
            act.line_ex_set_bus = tmp5
        assert np.all(act.line_ex_set_bus == 0), "a line (ext) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = np.array(li_orig).astype(str)
            tmp6[1] = "toto"
            act.line_ex_set_bus = tmp6
        assert np.all(act.line_ex_set_bus == 0), "a line (ext) unit has been modified by an illegal action"

    def test_line_ex_set_bus_tuple(self):
        # second set of tests, with tuple
        act = self.helper_action()
        act.line_ex_set_bus = (1, 1)
        assert np.all(act.line_ex_set_bus == [0, 1] + [0 for _ in range(18)])

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (1.0, 1)
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (False, 1)
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = ("toto", 1)
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (1, "toto")
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (21, 1)
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (-1, 1)
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"

        # not enough element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (1, )
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"

        # too much element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (1, 2, 3)
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"

    def test_line_ex_set_bus_list_asarray(self):
        """test the set attribute when list are given (list convertible to array)"""
        li_orig = [1, 2, -1] + [0 for _ in range(17)]  # because i have 2 storage unit
        tmp = np.array(li_orig)

        # ok
        act = self.helper_action()
        act.line_ex_set_bus = li_orig
        assert np.all(act.line_ex_set_bus == tmp)

        # list too short
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp0 = copy.deepcopy(li_orig)
            tmp0.pop(0)
            act.line_ex_set_bus = tmp0
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp1 = copy.deepcopy(li_orig)
            tmp1.append(2)
            act.line_ex_set_bus = tmp1
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [float(el) for el in li_orig]
            act.line_ex_set_bus = tmp3
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = -2
            act.line_ex_set_bus = tmp4
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = 3
            act.line_ex_set_bus = tmp5
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = [str(el) for el in li_orig]
            tmp6[1] = "toto"
            act.line_ex_set_bus = tmp6
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"

    def test_line_ex_set_bus_list_oftuple(self):
        """test the set attribute when list are given (list of tuple)"""
        li_orig = [(0, 1), (1, 2)]
        # ok
        act = self.helper_action()
        act.line_ex_set_bus = li_orig
        assert np.all(act.line_ex_set_bus == [1, 2] + [0 for _ in range(18)])

        # list of float (for the el_id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [(float(id_), new_bus) for id_, new_bus in li_orig]
            act.line_ex_set_bus = tmp3
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = (1, -2)
            act.line_ex_set_bus = tmp4
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = (1, 3)
            act.line_ex_set_bus = tmp5
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(li_orig)
            tmp6[1] = ("toto", 1)
            act.line_ex_set_bus = tmp6
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(li_orig)
            tmp7[1] = (3, "toto")
            act.line_ex_set_bus = tmp7
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(li_orig)
            tmp8.append((21, 1))
            act.line_ex_set_bus = tmp8
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(li_orig)
            tmp9.append((-1, 1))
            act.line_ex_set_bus = tmp9
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"

        # last test, when we give a list of tuple of exactly the right size
        act = self.helper_action()
        act.line_ex_set_bus = [(el, 2) for el in range(act.n_line)]
        assert np.all(act.line_ex_set_bus == 2)

    def test_line_ex_set_bus_dict_with_id(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {0: 1}
        # ok
        act = self.helper_action()
        act.line_ex_set_bus = dict_orig
        assert np.all(act.line_ex_set_bus == [1, 0] + [0 for _ in range(18)])

        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = {float(id_): new_bus for id_, new_bus in dict_orig.items()}
            act.line_ex_set_bus = tmp3
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(dict_orig)
            tmp4[1] = -2
            act.line_ex_set_bus = tmp4
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(dict_orig)
            tmp5[1] = 3
            act.line_ex_set_bus = tmp5
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1
            act.line_ex_set_bus = tmp6
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(dict_orig)
            tmp7[1] = "tata"
            act.line_ex_set_bus = tmp7
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(dict_orig)
            tmp8[21] = 1
            act.line_ex_set_bus = tmp8
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(dict_orig)
            tmp9[-1] = 1
            act.line_ex_set_bus = tmp9
        assert np.all(act.line_ex_set_bus == 0), "a line (ex) unit has been modified by an illegal action"

    def test_line_ex_set_bus_dict_with_name(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {"line_0": 1}
        # ok
        act = self.helper_action()
        act.line_or_set_bus = dict_orig
        assert np.all(act.line_or_set_bus == [1, 0] + [0 for _ in range(18)])

        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1  # unknown load
            act.line_or_set_bus = tmp6
        assert np.all(act.line_or_set_bus == 0), "a line (origin) unit has been modified by an illegal action"