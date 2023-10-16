# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import re
import unittest

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
        self.helper_action = self.ActionSpaceClass(
            self.gridobj,
            legal_action=self.game_rules.legal_action,
            actionClass=CompleteAction,
        )  # TopologySetAndStorageAction would be better
        self.helper_action.seed(42)
        # save_to_dict(self.res, self.helper_action, "subtype", lambda x: re.sub("(<class ')|('>)", "", "{}".format(x)))
        save_to_dict(
            self.res,
            self.helper_action,
            "_init_subtype",
            lambda x: re.sub(
                "(<class ')|(\\.init_grid\\.<locals>\\.res)|('>)", "", "{}".format(x)
            ),
        )

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
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

        # array too big
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp2 = np.concatenate((tmp, (1,)))
            act.load_set_bus = tmp2
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

        # float vect
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp3 = np.array(li_orig).astype(dt_float)
            act.load_set_bus = tmp3
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

        # one of the value too small
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp4 = np.array(li_orig)
            tmp4[2] = -2
            act.load_set_bus = tmp4
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

        # one of the value too large
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp5 = np.array(li_orig)
            tmp5[2] = 3
            act.load_set_bus = tmp5
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

        # wrong type
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp6 = np.array(li_orig).astype(str)
            tmp6[2] = "toto"
            act.load_set_bus = tmp6
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

    def test_load_set_bus_tuple(self):
        # second set of tests, with tuple
        act = self.helper_action()
        act.load_set_bus = (1, 1)
        assert np.all(act.load_set_bus == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (3.0, 1)
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (False, 1)
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = ("toto", 1)
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (1, "toto")
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (11, 1)
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (-1, 1)
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

        # not enough element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (1,)
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

        # too much element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = (1, 2, 3)
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

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
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp1 = copy.deepcopy(li_orig)
            tmp1.append(2)
            act.load_set_bus = tmp1
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [float(el) for el in li_orig]
            act.load_set_bus = tmp3
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[2] = -2
            act.load_set_bus = tmp4
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[2] = 3
            act.load_set_bus = tmp5
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = [str(el) for el in li_orig]
            tmp6[2] = "toto"
            act.load_set_bus = tmp6
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

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
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[2] = (3, -2)
            act.load_set_bus = tmp4
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[2] = (3, 3)
            act.load_set_bus = tmp5
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(li_orig)
            tmp6[2] = ("toto", 1)
            act.load_set_bus = tmp6
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(li_orig)
            tmp7[2] = (3, "toto")
            act.load_set_bus = tmp7
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(li_orig)
            tmp8.append((11, 1))
            act.load_set_bus = tmp8
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(li_orig)
            tmp9.append((-1, 1))
            act.load_set_bus = tmp9
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

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
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(dict_orig)
            tmp4[2] = -2
            act.load_set_bus = tmp4
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(dict_orig)
            tmp5[2] = 3
            act.load_set_bus = tmp5
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1
            act.load_set_bus = tmp6
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(dict_orig)
            tmp7[3] = "tata"
            act.load_set_bus = tmp7
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(dict_orig)
            tmp8[11] = 1
            act.load_set_bus = tmp8
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(dict_orig)
            tmp9[-1] = 1
            act.load_set_bus = tmp9
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

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
        assert np.all(
            act.load_set_bus == 0
        ), "a load has been modified by an illegal action"

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
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

        # array too big
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp2 = np.concatenate((tmp, (1,)))
            act.gen_set_bus = tmp2
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

        # float vect
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp3 = np.array(li_orig).astype(dt_float)
            act.gen_set_bus = tmp3
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

        # one of the value too small
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp4 = np.array(li_orig)
            tmp4[2] = -2
            act.gen_set_bus = tmp4
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

        # one of the value too large
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp5 = np.array(li_orig)
            tmp5[2] = 3
            act.gen_set_bus = tmp5
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

        # wrong type
        with self.assertRaises(IllegalAction):
            act = self.helper_action()
            tmp6 = np.array(li_orig).astype(str)
            tmp6[2] = "toto"
            act.gen_set_bus = tmp6
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

    def test_gen_set_bus_tuple(self):
        # second set of tests, with tuple
        act = self.helper_action()
        act.gen_set_bus = (1, 1)
        assert np.all(act.gen_set_bus == [0, 1, 0, 0, 0])

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (3.0, 1)
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (False, 1)
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = ("toto", 1)
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (1, "toto")
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (6, 1)
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (-1, 1)
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

        # not enough element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (1,)
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

        # too much element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = (1, 2, 3)
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

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
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp1 = copy.deepcopy(li_orig)
            tmp1.append(2)
            act.gen_set_bus = tmp1
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [float(el) for el in li_orig]
            act.gen_set_bus = tmp3
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[2] = -2
            act.gen_set_bus = tmp4
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[2] = 3
            act.gen_set_bus = tmp5
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = [str(el) for el in li_orig]
            tmp6[2] = "toto"
            act.gen_set_bus = tmp6
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

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
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[2] = (3, -2)
            act.gen_set_bus = tmp4
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[2] = (3, 3)
            act.gen_set_bus = tmp5
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(li_orig)
            tmp6[2] = ("toto", 1)
            act.gen_set_bus = tmp6
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(li_orig)
            tmp7[2] = (3, "toto")
            act.gen_set_bus = tmp7
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(li_orig)
            tmp8.append((5, 1))
            act.gen_set_bus = tmp8
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(li_orig)
            tmp9.append((-1, 1))
            act.gen_set_bus = tmp9
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

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
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(dict_orig)
            tmp4[2] = -2
            act.gen_set_bus = tmp4
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(dict_orig)
            tmp5[2] = 3
            act.gen_set_bus = tmp5
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1
            act.gen_set_bus = tmp6
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(dict_orig)
            tmp7[3] = "tata"
            act.gen_set_bus = tmp7
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(dict_orig)
            tmp8[11] = 1
            act.gen_set_bus = tmp8
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(dict_orig)
            tmp9[-1] = 1
            act.gen_set_bus = tmp9
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

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
        assert np.all(
            act.gen_set_bus == 0
        ), "a gen has been modified by an illegal action"

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
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

        # array too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp2 = np.concatenate((tmp, (1,)))
            act.storage_set_bus = tmp2
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

        # float vect
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = np.array(li_orig).astype(dt_float)
            act.storage_set_bus = tmp3
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = np.array(li_orig)
            tmp4[1] = -2
            act.storage_set_bus = tmp4
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = np.array(li_orig)
            tmp5[1] = 3
            act.storage_set_bus = tmp5
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = np.array(li_orig).astype(str)
            tmp6[1] = "toto"
            act.storage_set_bus = tmp6
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

    def test_storage_set_bus_tuple(self):
        # second set of tests, with tuple
        act = self.helper_action()
        act.storage_set_bus = (1, 1)
        assert np.all(act.storage_set_bus == [0, 1])

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (1.0, 1)
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (False, 1)
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = ("toto", 1)
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (1, "toto")
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (11, 1)
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (-1, 1)
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

        # not enough element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (1,)
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

        # too much element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = (1, 2, 3)
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

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
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp1 = copy.deepcopy(li_orig)
            tmp1.append(2)
            act.storage_set_bus = tmp1
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [float(el) for el in li_orig]
            act.storage_set_bus = tmp3
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = -2
            act.storage_set_bus = tmp4
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = 3
            act.storage_set_bus = tmp5
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = [str(el) for el in li_orig]
            tmp6[1] = "toto"
            act.storage_set_bus = tmp6
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

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
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = (1, -2)
            act.storage_set_bus = tmp4
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = (1, 3)
            act.storage_set_bus = tmp5
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(li_orig)
            tmp6[1] = ("toto", 1)
            act.storage_set_bus = tmp6
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(li_orig)
            tmp7[1] = (3, "toto")
            act.storage_set_bus = tmp7
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(li_orig)
            tmp8.append((2, 1))
            act.storage_set_bus = tmp8
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(li_orig)
            tmp9.append((-1, 1))
            act.storage_set_bus = tmp9
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

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
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(dict_orig)
            tmp4[1] = -2
            act.storage_set_bus = tmp4
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(dict_orig)
            tmp5[1] = 3
            act.storage_set_bus = tmp5
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1
            act.storage_set_bus = tmp6
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(dict_orig)
            tmp7[1] = "tata"
            act.storage_set_bus = tmp7
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(dict_orig)
            tmp8[2] = 1
            act.storage_set_bus = tmp8
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(dict_orig)
            tmp9[-1] = 1
            act.storage_set_bus = tmp9
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

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
        assert np.all(
            act.storage_set_bus == 0
        ), "a storage unit has been modified by an illegal action"

    def test_line_or_set_bus_array(self):
        li_orig = [
            1,
            2,
            -1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # because i have 20 lines
        tmp = np.array(li_orig)

        # first set of tests, with numpy array
        act = self.helper_action()
        act.line_or_set_bus = tmp  # ok
        assert np.all(act.line_or_set_bus == tmp)

        # array too short
        act = self.helper_action()

        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = tmp[0]
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

        # array too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp2 = np.concatenate((tmp, (1,)))
            act.line_or_set_bus = tmp2
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

        # float vect
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = np.array(li_orig).astype(dt_float)
            act.line_or_set_bus = tmp3
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = np.array(li_orig)
            tmp4[1] = -2
            act.line_or_set_bus = tmp4
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = np.array(li_orig)
            tmp5[1] = 3
            act.line_or_set_bus = tmp5
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = np.array(li_orig).astype(str)
            tmp6[1] = "toto"
            act.line_or_set_bus = tmp6
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

    def test_line_or_set_bus_tuple(self):
        # second set of tests, with tuple
        act = self.helper_action()
        act.line_or_set_bus = (1, 1)
        assert np.all(act.line_or_set_bus == [0, 1] + [0 for _ in range(18)])

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (1.0, 1)
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (False, 1)
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = ("toto", 1)
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (1, "toto")
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (21, 1)
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (-1, 1)
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

        # not enough element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (1,)
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

        # too much element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = (1, 2, 3)
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

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
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp1 = copy.deepcopy(li_orig)
            tmp1.append(2)
            act.line_or_set_bus = tmp1
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [float(el) for el in li_orig]
            act.line_or_set_bus = tmp3
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = -2
            act.line_or_set_bus = tmp4
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = 3
            act.line_or_set_bus = tmp5
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = [str(el) for el in li_orig]
            tmp6[1] = "toto"
            act.line_or_set_bus = tmp6
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

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
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = (1, -2)
            act.line_or_set_bus = tmp4
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = (1, 3)
            act.line_or_set_bus = tmp5
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(li_orig)
            tmp6[1] = ("toto", 1)
            act.line_or_set_bus = tmp6
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(li_orig)
            tmp7[1] = (3, "toto")
            act.line_or_set_bus = tmp7
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(li_orig)
            tmp8.append((21, 1))
            act.line_or_set_bus = tmp8
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(li_orig)
            tmp9.append((-1, 1))
            act.line_or_set_bus = tmp9
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

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
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(dict_orig)
            tmp4[1] = -2
            act.line_or_set_bus = tmp4
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(dict_orig)
            tmp5[1] = 3
            act.line_or_set_bus = tmp5
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1
            act.line_or_set_bus = tmp6
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(dict_orig)
            tmp7[1] = "tata"
            act.line_or_set_bus = tmp7
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(dict_orig)
            tmp8[21] = 1
            act.line_or_set_bus = tmp8
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(dict_orig)
            tmp9[-1] = 1
            act.line_or_set_bus = tmp9
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

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
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

    def test_line_ex_set_bus_array(self):
        li_orig = [
            1,
            2,
            -1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # because i have 20 lines
        tmp = np.array(li_orig)

        # first set of tests, with numpy array
        act = self.helper_action()
        act.line_ex_set_bus = tmp  # ok
        assert np.all(act.line_ex_set_bus == tmp)

        # array too short
        act = self.helper_action()

        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = tmp[0]
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ext) unit has been modified by an illegal action"

        # array too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp2 = np.concatenate((tmp, (1,)))
            act.line_ex_set_bus = tmp2
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ext) unit has been modified by an illegal action"

        # float vect
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = np.array(li_orig).astype(dt_float)
            act.line_ex_set_bus = tmp3
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ext) unit has been modified by an illegal action"

        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = np.array(li_orig)
            tmp4[1] = -2
            act.line_ex_set_bus = tmp4
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ext) unit has been modified by an illegal action"

        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = np.array(li_orig)
            tmp5[1] = 3
            act.line_ex_set_bus = tmp5
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ext) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = np.array(li_orig).astype(str)
            tmp6[1] = "toto"
            act.line_ex_set_bus = tmp6
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ext) unit has been modified by an illegal action"

    def test_line_ex_set_bus_tuple(self):
        # second set of tests, with tuple
        act = self.helper_action()
        act.line_ex_set_bus = (1, 1)
        assert np.all(act.line_ex_set_bus == [0, 1] + [0 for _ in range(18)])

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (1.0, 1)
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (False, 1)
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = ("toto", 1)
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (1, "toto")
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (21, 1)
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (-1, 1)
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"

        # not enough element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (1,)
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"

        # too much element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = (1, 2, 3)
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"

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
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp1 = copy.deepcopy(li_orig)
            tmp1.append(2)
            act.line_ex_set_bus = tmp1
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [float(el) for el in li_orig]
            act.line_ex_set_bus = tmp3
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = -2
            act.line_ex_set_bus = tmp4
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = 3
            act.line_ex_set_bus = tmp5
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = [str(el) for el in li_orig]
            tmp6[1] = "toto"
            act.line_ex_set_bus = tmp6
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"

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
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = (1, -2)
            act.line_ex_set_bus = tmp4
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = (1, 3)
            act.line_ex_set_bus = tmp5
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(li_orig)
            tmp6[1] = ("toto", 1)
            act.line_ex_set_bus = tmp6
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(li_orig)
            tmp7[1] = (3, "toto")
            act.line_ex_set_bus = tmp7
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(li_orig)
            tmp8.append((21, 1))
            act.line_ex_set_bus = tmp8
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(li_orig)
            tmp9.append((-1, 1))
            act.line_ex_set_bus = tmp9
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"

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
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(dict_orig)
            tmp4[1] = -2
            act.line_ex_set_bus = tmp4
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(dict_orig)
            tmp5[1] = 3
            act.line_ex_set_bus = tmp5
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1
            act.line_ex_set_bus = tmp6
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(dict_orig)
            tmp7[1] = "tata"
            act.line_ex_set_bus = tmp7
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(dict_orig)
            tmp8[21] = 1
            act.line_ex_set_bus = tmp8
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(dict_orig)
            tmp9[-1] = 1
            act.line_ex_set_bus = tmp9
        assert np.all(
            act.line_ex_set_bus == 0
        ), "a line (ex) unit has been modified by an illegal action"

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
        assert np.all(
            act.line_or_set_bus == 0
        ), "a line (origin) unit has been modified by an illegal action"

    def test_set_by_sub(self):
        # TODO more thorough testing !!!
        act = self.helper_action()
        act.sub_set_bus = (1, (1, 1, -1, 1, 2, 1, -1))
        aff_lines, aff_subs = act.get_topological_impact()
        assert aff_subs[1]
        assert np.sum(aff_subs) == 1

        with self.assertRaises(IllegalAction):
            act.sub_set_bus = (1, (1, 1, -1, 1, 2, 3, -1))  # one too high
        with self.assertRaises(IllegalAction):
            act.sub_set_bus = (1, (1, 1, -1, 1, 2, -2, -1))  # one too low
        with self.assertRaises(IllegalAction):
            act.sub_set_bus = (1, (1, 1, -1, 1, 2, -1))  # too short
        with self.assertRaises(IllegalAction):
            act.sub_set_bus = (1, (1, 1, -1, 1, 2, 1, 2, 2))  # too big

        with self.assertRaises(IllegalAction):
            act.sub_set_bus = np.zeros(act.dim_topo + 1, dtype=int)  # too long
        with self.assertRaises(IllegalAction):
            act.sub_set_bus = np.zeros(act.dim_topo - 1, dtype=int)  # too short

        # ok
        tmp = np.zeros(act.dim_topo, dtype=int)  # too short
        tmp[:10] = 1
        act.sub_set_bus = tmp
        aff_lines, aff_subs = act.get_topological_impact()
        assert aff_subs[0]
        assert aff_subs[1]
        assert np.sum(aff_subs) == 2

    def test_change_by_sub(self):
        # TODO more thorough testing !!!
        act = self.helper_action()
        act.sub_change_bus = (1, (True, True, True, False, False, True, False))
        aff_lines, aff_subs = act.get_topological_impact()
        assert aff_subs[1]
        assert np.sum(aff_subs) == 1

        with self.assertRaises(IllegalAction):
            act.sub_change_bus = (
                1,
                (True, True, True, False, False, True),
            )  # too short
        with self.assertRaises(IllegalAction):
            act.sub_change_bus = (
                1,
                (True, True, True, False, False, True, False, True),
            )  # too big

        with self.assertRaises(IllegalAction):
            act.sub_change_bus = np.zeros(act.dim_topo + 1, dtype=int)  # too long
        with self.assertRaises(IllegalAction):
            act.sub_change_bus = np.zeros(act.dim_topo - 1, dtype=int)  # too short
        with self.assertRaises(IllegalAction):
            act.sub_change_bus = np.zeros(act.dim_topo - 1, dtype=int)  # wrong type
        with self.assertRaises(IllegalAction):
            act.sub_change_bus = np.zeros(act.dim_topo - 1, dtype=float)  # wrong type

        # ok
        tmp = np.zeros(act.dim_topo, dtype=bool)  # too short
        tmp[:10] = True
        act.sub_change_bus = tmp
        aff_lines, aff_subs = act.get_topological_impact()
        assert aff_subs[0]
        assert aff_subs[1]
        assert np.sum(aff_subs) == 2


class TestSetStatus(unittest.TestCase):
    """test the property to set the status of the action"""

    # TODO test the act.set_bus too here !
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
        self.helper_action = self.ActionSpaceClass(
            self.gridobj,
            legal_action=self.game_rules.legal_action,
            actionClass=PowerlineSetAction,
        )
        self.helper_action.seed(42)
        # save_to_dict(self.res, self.helper_action, "subtype", lambda x: re.sub("(<class ')|('>)", "", "{}".format(x)))
        save_to_dict(
            self.res,
            self.helper_action,
            "_init_subtype",
            lambda x: re.sub(
                "(<class ')|(\\.init_grid\\.<locals>\\.res)|('>)", "", "{}".format(x)
            ),
        )

        self.authorized_keys = self.helper_action().authorized_keys
        self.size_act = self.helper_action.size()

    def test_line_set_status_array(self):
        li_orig = [
            1,
            1,
            -1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # because i have 20 lines
        tmp = np.array(li_orig)

        # first set of tests, with numpy array
        act = self.helper_action()
        act.line_set_status = tmp  # ok
        assert np.all(act.line_set_status == tmp)

        # array too short
        act = self.helper_action()

        with self.assertRaises(IllegalAction):
            act.line_set_status = tmp[0]
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # array too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp2 = np.concatenate((tmp, (1,)))
            act.line_set_status = tmp2
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # float vect
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = np.array(li_orig).astype(dt_float)
            act.line_set_status = tmp3
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = np.array(li_orig)
            tmp4[1] = -2
            act.line_set_status = tmp4
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = np.array(li_orig)
            tmp5[1] = 2
            act.line_set_status = tmp5
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = np.array(li_orig).astype(str)
            tmp6[1] = "toto"
            act.line_ex_set_bus = tmp6
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

    def test_line_set_status_tuple(self):
        # second set of tests, with tuple
        act = self.helper_action()
        act.line_set_status = (1, 1)
        assert np.all(act.line_set_status == [0, 1] + [0 for _ in range(18)])

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_set_status = (1.0, 1)
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_set_status = (False, 1)
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_set_status = ("toto", 1)
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_set_status = (1, "toto")
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_set_status = (21, 1)
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_set_status = (-1, 1)
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # not enough element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_set_status = (1,)
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # too much element in the tuple
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            act.line_set_status = (1, 2, 3)
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

    def test_line_set_status_list_asarray(self):
        """test the set attribute when list are given (list convertible to array)"""
        li_orig = [1, 1, -1] + [0 for _ in range(17)]  # because i have 2 storage unit
        tmp = np.array(li_orig)

        # ok
        act = self.helper_action()
        act.line_set_status = li_orig
        assert np.all(act.line_set_status == tmp)

        # list too short
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp0 = copy.deepcopy(li_orig)
            tmp0.pop(0)
            act.line_set_status = tmp0
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp1 = copy.deepcopy(li_orig)
            tmp1.append(1)
            act.line_set_status = tmp1
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [float(el) for el in li_orig]
            act.line_set_status = tmp3
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # one of the value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = -2
            act.line_set_status = tmp4
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # one of the value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = 2
            act.line_set_status = tmp5
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = [str(el) for el in li_orig]
            tmp6[1] = "toto"
            act.line_set_status = tmp6
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

    def test_line_set_status_list_oftuple(self):
        """test the set attribute when list are given (list of tuple)"""
        li_orig = [(0, 1), (1, 1)]
        # ok
        act = self.helper_action()
        act.line_set_status = li_orig
        assert np.all(act.line_set_status == [1, 1] + [0 for _ in range(18)])

        # list of float (for the el_id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [(float(id_), new_bus) for id_, new_bus in li_orig]
            act.line_set_status = tmp3
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4[1] = (1, -2)
            act.line_set_status = tmp4
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5[1] = (1, 2)
            act.line_set_status = tmp5
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(li_orig)
            tmp6[1] = ("toto", 1)
            act.line_set_status = tmp6
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(li_orig)
            tmp7[1] = (3, "toto")
            act.line_set_status = tmp7
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(li_orig)
            tmp8.append((21, 1))
            act.line_set_status = tmp8
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(li_orig)
            tmp9.append((-1, 1))
            act.line_set_status = tmp9
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

        # last test, when we give a list of tuple of exactly the right size
        act = self.helper_action()
        act.line_set_status = [(el, 1) for el in range(act.n_line)]
        assert np.all(act.line_set_status == 1)

    def test_line_set_status_dict_with_id(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {0: 1}
        # ok
        act = self.helper_action()
        act.line_set_status = dict_orig
        assert np.all(act.line_set_status == [1, 0] + [0 for _ in range(18)])

        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = {float(id_): new_bus for id_, new_bus in dict_orig.items()}
            act.line_set_status = tmp3
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(dict_orig)
            tmp4[1] = -2
            act.line_set_status = tmp4
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(dict_orig)
            tmp5[1] = 3
            act.line_set_status = tmp5
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1
            act.line_set_status = tmp6
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(dict_orig)
            tmp7[1] = "tata"
            act.line_set_status = tmp7
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(dict_orig)
            tmp8[21] = 1
            act.line_set_status = tmp8
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(dict_orig)
            tmp9[-1] = 1
            act.line_set_status = tmp9
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

    def test_line_set_status_dict_with_name(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {"line_0": 1}
        # ok
        act = self.helper_action()
        act.line_set_status = dict_orig
        assert np.all(act.line_set_status == [1, 0] + [0 for _ in range(18)])

        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1  # unknown load
            act.line_set_status = tmp6
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"


class TestChangeBus(unittest.TestCase):
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
        self.helper_action = self.ActionSpaceClass(
            self.gridobj,
            legal_action=self.game_rules.legal_action,
            actionClass=CompleteAction,
        )  # TopologyChangeAndStorageAction would be better
        self.helper_action.seed(42)
        # save_to_dict(self.res, self.helper_action, "subtype", lambda x: re.sub("(<class ')|('>)", "", "{}".format(x)))
        save_to_dict(
            self.res,
            self.helper_action,
            "_init_subtype",
            lambda x: re.sub(
                "(<class ')|(\\.init_grid\\.<locals>\\.res)|('>)", "", "{}".format(x)
            ),
        )

        self.authorized_keys = self.helper_action().authorized_keys
        self.size_act = self.helper_action.size()

    def _aux_change_bus_int(self, name_el, nb_el, prop="change_bus"):
        """first set of test by giving the id of the object i want to change"""
        act = self.helper_action()
        prop_name = f"{name_el}_{prop}"
        setattr(act, prop_name, 1)
        assert np.all(
            getattr(act, prop_name) == [False, True] + [False for _ in range(nb_el - 2)]
        )

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, 3.0)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, False)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, "toto")
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (1, "toto"))
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, nb_el + 1)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, -1)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

    def test_load_change_bus_int(self):
        self._aux_change_bus_int("load", self.helper_action.n_load)

    def test_gen_change_bus_int(self):
        self._aux_change_bus_int("gen", nb_el=self.helper_action.n_gen)

    def test_storage_change_bus_int(self):
        self._aux_change_bus_int("storage", nb_el=self.helper_action.n_storage)

    def test_line_or_change_bus_int(self):
        self._aux_change_bus_int("line_or", nb_el=self.helper_action.n_line)

    def test_line_ex_change_bus_int(self):
        self._aux_change_bus_int("line_ex", nb_el=self.helper_action.n_line)

    def test_line_change_status_bus_int(self):
        self._aux_change_bus_int(
            "line", nb_el=self.helper_action.n_line, prop="change_status"
        )

    def _aux_change_bus_tuple(self, name_el, nb_el, prop="change_bus"):
        """first set of test by giving the a tuple: should be deactivated!"""
        act = self.helper_action()
        prop_name = f"{name_el}_{prop}"
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (1,))
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (1, False))
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (1, False, 3))
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

    def test_load_change_bus_tuple(self):
        self._aux_change_bus_tuple("load", self.helper_action.n_load)

    def test_gen_change_bus_tuple(self):
        self._aux_change_bus_tuple("gen", nb_el=self.helper_action.n_gen)

    def test_storage_change_bus_tuple(self):
        self._aux_change_bus_tuple("storage", nb_el=self.helper_action.n_storage)

    def test_line_or_change_bus_tuple(self):
        self._aux_change_bus_tuple("line_or", nb_el=self.helper_action.n_line)

    def test_line_ex_change_bus_tuple(self):
        self._aux_change_bus_tuple("line_ex", nb_el=self.helper_action.n_line)

    def test_line_change_status_bus_tuple(self):
        self._aux_change_bus_tuple(
            "line", nb_el=self.helper_action.n_line, prop="change_status"
        )

    def _aux_change_bus_arraybool(self, name_el, nb_el, prop="change_bus"):
        """test by giving the a complete an array of bool (all the vector)"""
        prop_name = f"{name_el}_{prop}"
        li_orig = [False, True] + [False for _ in range(nb_el - 2)]
        tmp = np.array(li_orig)
        tmp_dt_bool = np.array(li_orig).astype(dt_bool)
        tmp_bool = np.array(li_orig).astype(bool)

        act = self.helper_action()
        setattr(act, prop_name, tmp)
        assert np.all(
            getattr(act, prop_name) == [False, True] + [False for _ in range(nb_el - 2)]
        )

        act = self.helper_action()
        setattr(act, prop_name, tmp_dt_bool)
        assert np.all(
            getattr(act, prop_name) == [False, True] + [False for _ in range(nb_el - 2)]
        )

        act = self.helper_action()
        setattr(act, prop_name, tmp_bool)
        assert np.all(
            getattr(act, prop_name) == [False, True] + [False for _ in range(nb_el - 2)]
        )

        # list too short
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, tmp[:-1])
        assert np.all(
            ~getattr(act, prop_name)
        ), "a load has been modified by an illegal action"

        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp_1 = np.concatenate((tmp, (False,)))
            setattr(act, prop_name, tmp_1)
        assert np.all(
            ~getattr(act, prop_name)
        ), "a load has been modified by an illegal action"

    def test_load_change_bus_arraybool(self):
        self._aux_change_bus_arraybool("load", self.helper_action.n_load)

    def test_gen_change_bus_arraybool(self):
        self._aux_change_bus_arraybool("gen", nb_el=self.helper_action.n_gen)

    def test_storage_change_bus_arraybool(self):
        self._aux_change_bus_arraybool("storage", nb_el=self.helper_action.n_storage)

    def test_line_or_change_bus_arraybool(self):
        self._aux_change_bus_arraybool("line_or", nb_el=self.helper_action.n_line)

    def test_line_ex_change_bus_arraybool(self):
        self._aux_change_bus_arraybool("line_ex", nb_el=self.helper_action.n_line)

    def test_line_change_status_bus_arraybool(self):
        self._aux_change_bus_arraybool(
            "line", nb_el=self.helper_action.n_line, prop="change_status"
        )

    def _aux_change_bus_arrayint(self, name_el, nb_el, prop="change_bus"):
        """test by giving the a numpy array of int"""
        prop_name = f"{name_el}_{prop}"
        li_orig = [0, 1]
        tmp = np.array(li_orig)
        tmp_dt_int = np.array(li_orig).astype(dt_int)
        tmp_int = np.array(li_orig).astype(int)

        act = self.helper_action()
        setattr(act, prop_name, tmp)
        assert np.all(
            getattr(act, prop_name) == [True, True] + [False for _ in range(nb_el - 2)]
        )

        act = self.helper_action()
        setattr(act, prop_name, tmp_dt_int)
        assert np.all(
            getattr(act, prop_name) == [True, True] + [False for _ in range(nb_el - 2)]
        )

        act = self.helper_action()
        setattr(act, prop_name, tmp_int)
        assert np.all(
            getattr(act, prop_name) == [True, True] + [False for _ in range(nb_el - 2)]
        )

        # one id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp2 = np.concatenate((tmp, (-1,)))
            setattr(act, prop_name, tmp2)
        assert np.all(
            ~getattr(act, prop_name)
        ), "a load has been modified by an illegal action"

        # one id too high
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = np.concatenate((tmp, (nb_el,)))
            setattr(act, prop_name, tmp3)
        assert np.all(
            ~getattr(act, prop_name)
        ), "a load has been modified by an illegal action"

    def test_load_change_bus_arrayint(self):
        self._aux_change_bus_arrayint("load", self.helper_action.n_load)

    def test_gen_change_bus_arrayint(self):
        self._aux_change_bus_arrayint("gen", nb_el=self.helper_action.n_gen)

    def test_storage_change_bus_arrayint(self):
        self._aux_change_bus_arrayint("storage", nb_el=self.helper_action.n_storage)

    def test_line_or_change_bus_arrayint(self):
        self._aux_change_bus_arrayint("line_or", nb_el=self.helper_action.n_line)

    def test_line_ex_change_bus_arrayint(self):
        self._aux_change_bus_arrayint("line_ex", nb_el=self.helper_action.n_line)

    def test_line_change_status_bus_arrayint(self):
        self._aux_change_bus_arrayint(
            "line", nb_el=self.helper_action.n_line, prop="change_status"
        )

    def _aux_change_bus_listbool(self, name_el, nb_el, prop="change_bus"):
        """
        test by giving the a complete a list of bool (all the vector)
        has been deactivate because of impossibility to check the test with `li_5`
        below
        """
        prop_name = f"{name_el}_{prop}"
        li_orig = [False, True] + [False for _ in range(nb_el)]

        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, li_orig)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # list too short
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, li_orig[:-1])
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            li_2 = copy.deepcopy(li_orig)
            li_2.append(True)
            setattr(act, prop_name, li_2)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # list mixed types (str)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            li_3 = copy.deepcopy(li_orig)
            li_3.append("toto")
            setattr(act, prop_name, li_3)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # list mixed types (float)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            li_4 = copy.deepcopy(li_orig)
            li_4.append(1.0)
            setattr(act, prop_name, li_4)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # list mixed types (int)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            li_5 = copy.deepcopy(li_orig)
            li_5.append(1)
            setattr(act, prop_name, li_5)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

    def test_load_change_bus_listbool(self):
        self._aux_change_bus_listbool("load", nb_el=self.helper_action.n_load)

    def test_gen_change_bus_listbool(self):
        self._aux_change_bus_listbool("gen", nb_el=self.helper_action.n_gen)

    def test_storage_change_bus_listbool(self):
        self._aux_change_bus_listbool("storage", nb_el=self.helper_action.n_storage)

    def test_line_or_change_bus_listbool(self):
        self._aux_change_bus_listbool("line_or", nb_el=self.helper_action.n_line)

    def test_line_ex_change_bus_listbool(self):
        self._aux_change_bus_listbool("line_ex", nb_el=self.helper_action.n_line)

    def test_line_change_status_bus_listbool(self):
        self._aux_change_bus_listbool(
            "line", nb_el=self.helper_action.n_line, prop="change_status"
        )

    def _aux_change_bus_listint(self, name_el, nb_el, prop="change_bus"):
        """
        test by giving the a a list of int
        """
        prop_name = f"{name_el}_{prop}"
        li_orig = [0]
        act = self.helper_action()
        setattr(act, prop_name, li_orig)
        assert np.all(
            getattr(act, prop_name) == [True, False] + [False for _ in range(nb_el - 2)]
        )

        # one id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp2 = copy.deepcopy(li_orig)
            tmp2.append(-1)
            setattr(act, prop_name, tmp2)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # one id too high
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = copy.deepcopy(li_orig)
            tmp3.append(nb_el)
            setattr(act, prop_name, tmp3)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # one string
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(li_orig)
            tmp4.append("toto")
            setattr(act, prop_name, tmp4)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # one float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5.append(1.0)
            setattr(act, prop_name, tmp5)
        assert np.all(
            ~getattr(act, prop_name)
        ), f"a {name_el} has been modified by an illegal action"

        # test it revert back to proper thing
        act = self.helper_action()
        setattr(act, prop_name, li_orig)
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(li_orig)
            tmp5.append(1.0)
            setattr(act, prop_name, tmp5)
        assert np.all(
            getattr(act, prop_name) == [True, False] + [False for _ in range(nb_el - 2)]
        )

        # test if change twice it's equivalent to not changing at all
        act = self.helper_action()
        setattr(act, prop_name, li_orig)
        assert np.all(
            getattr(act, prop_name) == [True, False] + [False for _ in range(nb_el - 2)]
        )
        setattr(act, prop_name, li_orig)
        assert np.all(
            getattr(act, prop_name)
            == [False, False] + [False for _ in range(nb_el - 2)]
        )

    def test_load_change_bus_listint(self):
        self._aux_change_bus_listint("load", nb_el=self.helper_action.n_load)

    def test_gen_change_bus_listint(self):
        self._aux_change_bus_listint("gen", nb_el=self.helper_action.n_gen)

    def test_storage_change_bus_listint(self):
        self._aux_change_bus_listint("storage", nb_el=self.helper_action.n_storage)

    def test_line_or_change_bus_listint(self):
        self._aux_change_bus_listint("line_or", nb_el=self.helper_action.n_line)

    def test_line_ex_change_bus_listint(self):
        self._aux_change_bus_listint("line_ex", nb_el=self.helper_action.n_line)

    def test_line_change_status_bus_listint(self):
        self._aux_change_bus_listint(
            "line", nb_el=self.helper_action.n_line, prop="change_status"
        )


class TestSetValues(unittest.TestCase):
    """test the property to set continuous values"""

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
        self.helper_action = self.ActionSpaceClass(
            self.gridobj,
            legal_action=self.game_rules.legal_action,
            actionClass=CompleteAction,
        )
        self.helper_action.seed(42)
        # save_to_dict(self.res, self.helper_action, "subtype", lambda x: re.sub("(<class ')|('>)", "", "{}".format(x)))
        save_to_dict(
            self.res,
            self.helper_action,
            "_init_subtype",
            lambda x: re.sub(
                "(<class ')|(\\.init_grid\\.<locals>\\.res)|('>)", "", "{}".format(x)
            ),
        )

        self.authorized_keys = self.helper_action().authorized_keys
        self.size_act = self.helper_action.size()

    def _aux_change_val_tuple(self, name_el, nb_el, prop_name):
        """first set of test by giving the id of the object i want to change"""
        this_zero = [0.0 for _ in range(nb_el)]

        # regular modification
        act = self.helper_action()
        setattr(act, prop_name, (1, 1.0))
        assert np.all(
            getattr(act, prop_name) == [0.0, 1.0] + [0.0 for _ in range(nb_el - 2)]
        )

        # nan action: should be discarded
        act = self.helper_action()
        setattr(act, prop_name, (1, np.NaN))
        assert np.all(getattr(act, prop_name) == [0.0 for _ in range(nb_el)])

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (3.0, 1.0))
        assert np.all(
            getattr(act, prop_name) == this_zero
        ), f"a {name_el} has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (False, 1.0))
        assert np.all(
            getattr(act, prop_name) == this_zero
        ), f"a {name_el} has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, ("toto", 1.0))
        assert np.all(
            getattr(act, prop_name) == this_zero
        ), f"a {name_el} has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (1, "toto"))
        assert np.all(
            getattr(act, prop_name) == this_zero
        ), f"a {name_el} has been modified by an illegal action"

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (1, False))
        assert np.all(
            getattr(act, prop_name) == this_zero
        ), f"a {name_el} has been modified by an illegal action"

        # id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (nb_el + 1, 1.0))
        assert np.all(
            getattr(act, prop_name) == this_zero
        ), f"a {name_el} has been modified by an illegal action"

        # id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (-1, 1.0))
        assert np.all(
            getattr(act, prop_name) == this_zero
        ), f"a {name_el} has been modified by an illegal action"

        # tuple wrong size
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (1,))
        assert np.all(
            getattr(act, prop_name) == this_zero
        ), f"a {name_el} has been modified by an illegal action"

        # tuple wrong size
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (1, 1.0, 1))
        assert np.all(
            getattr(act, prop_name) == this_zero
        ), f"a {name_el} has been modified by an illegal action"

        # test correct canceling
        act = self.helper_action()
        setattr(act, prop_name, (1, 1.0))
        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, (1, 1.0, 1))
        assert np.all(
            getattr(act, prop_name) == [0.0, 1.0] + [0.0 for _ in range(nb_el - 2)]
        )

    def test_redisp_tuple(self):
        self._aux_change_val_tuple("redisp", self.helper_action.n_gen, "redispatch")

    def test_storage_power_tuple(self):
        self._aux_change_val_tuple("storage", self.helper_action.n_storage, "storage_p")

    def _aux_set_val_array(self, name_el, nb_el, prop_name):
        li_orig = [1.0, -1.0] + [0.0 for _ in range(nb_el - 2)]
        tmp = np.array(li_orig)
        tmp_dt_float = np.array(li_orig).astype(dt_float)
        tmp_np_float = np.array(li_orig).astype(float)

        # first set of tests, with numpy array
        act = self.helper_action()
        setattr(act, prop_name, tmp)  # ok
        assert np.all(getattr(act, prop_name) == li_orig)

        act = self.helper_action()
        setattr(act, prop_name, tmp_dt_float)  # ok
        assert np.all(getattr(act, prop_name) == li_orig)

        act = self.helper_action()
        setattr(act, prop_name, tmp_np_float)  # ok
        assert np.all(getattr(act, prop_name) == li_orig)

        # array too short
        act = self.helper_action()

        with self.assertRaises(IllegalAction):
            setattr(act, prop_name, tmp[0])
        assert np.all(getattr(act, prop_name) == 0)

        # array too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp2 = np.concatenate((tmp, (1,)))
            setattr(act, prop_name, tmp2)
        assert np.all(getattr(act, prop_name) == 0)

        # bool vect
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = np.array(li_orig).astype(dt_bool)
            setattr(act, prop_name, tmp3)
        assert np.all(getattr(act, prop_name) == 0)

        # int vect
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = np.array(li_orig).astype(dt_int)
            setattr(act, prop_name, tmp4)
        assert np.all(getattr(act, prop_name) == 0)

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = np.array(li_orig).astype(str)
            tmp6[1] = "toto"
            setattr(act, prop_name, tmp6)
        assert np.all(getattr(act, prop_name) == 0)

        # test reset ok
        act = self.helper_action()
        setattr(act, prop_name, tmp)  # ok
        with self.assertRaises(IllegalAction):
            tmp6 = np.array(li_orig).astype(str)
            tmp6[1] = "toto"
            setattr(act, prop_name, tmp6)
        assert np.all(getattr(act, prop_name) == li_orig)

    def test_redisp_array(self):
        self._aux_set_val_array("redisp", self.helper_action.n_gen, "redispatch")

    def test_storage_power_array(self):
        self._aux_set_val_array("storage", self.helper_action.n_storage, "storage_p")

    def _aux_set_val_list_asarray(self, name_el, nb_el, prop_name):
        """test the set attribute when list are given (list convertible to array)"""
        li_orig = [1.0, -1.0] + [0 for _ in range(nb_el - 2)]
        tmp = np.array(li_orig)

        # ok
        act = self.helper_action()
        setattr(act, prop_name, li_orig)
        assert np.all(getattr(act, prop_name) == tmp)

        # list too short
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp0 = copy.deepcopy(li_orig)
            tmp0.pop(0)
            setattr(act, prop_name, tmp0)
        assert np.all(getattr(act, prop_name) == 0)

        # list too big
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp1 = copy.deepcopy(li_orig)
            tmp1.append(1.0)
            setattr(act, prop_name, tmp1)
        assert np.all(getattr(act, prop_name) == 0)

        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [int(el) for el in li_orig]
            setattr(act, prop_name, tmp3)
        assert np.all(getattr(act, prop_name) == 0)

        # wrong type
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = [str(el) for el in li_orig]
            tmp6[1] = "toto"
            setattr(act, prop_name, tmp6)
        assert np.all(getattr(act, prop_name) == 0)

        # reset ok
        act = self.helper_action()
        setattr(act, prop_name, li_orig)
        with self.assertRaises(IllegalAction):
            tmp3 = [int(el) for el in li_orig]
            setattr(act, prop_name, tmp3)
        assert np.all(getattr(act, prop_name) == tmp)

    def test_redisp_list_asarray(self):
        self._aux_set_val_list_asarray("redisp", self.helper_action.n_gen, "redispatch")

    def test_storage_power_list_asarray(self):
        self._aux_set_val_list_asarray(
            "storage", self.helper_action.n_storage, "storage_p"
        )

    def _aux_set_val_list_oftuple(self, name_el, nb_el, prop_name):
        """test the set attribute when list are given (list of tuple)"""
        li_orig = [(0, 1.0), (1, -1.0)]
        # ok
        act = self.helper_action()
        setattr(act, prop_name, li_orig)
        assert np.all(
            getattr(act, prop_name) == [1.0, -1.0] + [0.0 for _ in range(nb_el - 2)]
        )

        # list of float (for the el_id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = [(float(id_), new_bus) for id_, new_bus in li_orig]
            setattr(act, prop_name, tmp3)
        assert np.all(getattr(act, prop_name) == 0)

        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(li_orig)
            tmp6[1] = ("toto", 1)
            setattr(act, prop_name, tmp6)
        assert np.all(getattr(act, prop_name) == 0)
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(li_orig)
            tmp7[1] = (3, "toto")
            setattr(act, prop_name, tmp7)
        assert np.all(getattr(act, prop_name) == 0)
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(li_orig)
            tmp8.append((21, 1))
            setattr(act, prop_name, tmp8)
        assert np.all(getattr(act, prop_name) == 0)
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(li_orig)
            tmp9.append((-1, 1))
            setattr(act, prop_name, tmp9)
        assert np.all(getattr(act, prop_name) == 0)

        # last test, when we give a list of tuple of exactly the right size
        act = self.helper_action()
        setattr(act, prop_name, [(el, 1) for el in range(nb_el)])
        assert np.all(getattr(act, prop_name) == 1)

    def test_redisp_list_oftuple(self):
        self._aux_set_val_list_oftuple("redisp", self.helper_action.n_gen, "redispatch")

    def test_storage_power_list_oftuple(self):
        self._aux_set_val_list_oftuple(
            "storage", self.helper_action.n_storage, "storage_p"
        )

    def todo_line_set_status_dict_with_id(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {0: 1}
        # ok
        act = self.helper_action()
        act.line_set_status = dict_orig
        assert np.all(act.line_set_status == [1, 0] + [0 for _ in range(18)])

        # list of float
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp3 = {float(id_): new_bus for id_, new_bus in dict_orig.items()}
            act.line_set_status = tmp3
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # one of the bus value too small
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp4 = copy.deepcopy(dict_orig)
            tmp4[1] = -2
            act.line_set_status = tmp4
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # one of the bus value too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp5 = copy.deepcopy(dict_orig)
            tmp5[1] = 3
            act.line_set_status = tmp5
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # wrong type (element id)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1
            act.line_set_status = tmp6
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # wrong type (bus value)
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp7 = copy.deepcopy(dict_orig)
            tmp7[1] = "tata"
            act.line_set_status = tmp7
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # el_id too large
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp8 = copy.deepcopy(dict_orig)
            tmp8[21] = 1
            act.line_set_status = tmp8
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"
        # el_id too low
        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp9 = copy.deepcopy(dict_orig)
            tmp9[-1] = 1
            act.line_set_status = tmp9
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"

    def todo_line_set_status_dict_with_name(self):
        """test the set attribute when list are given (list of tuple)"""
        dict_orig = {"line_0": 1}
        # ok
        act = self.helper_action()
        act.line_set_status = dict_orig
        assert np.all(act.line_set_status == [1, 0] + [0 for _ in range(18)])

        act = self.helper_action()
        with self.assertRaises(IllegalAction):
            tmp6 = copy.deepcopy(dict_orig)
            tmp6["toto"] = 1  # unknown load
            act.line_set_status = tmp6
        assert np.all(
            act.line_set_status == 0
        ), "a line status has been modified by an illegal action"


if __name__ == "__main__":
    unittest.main()
