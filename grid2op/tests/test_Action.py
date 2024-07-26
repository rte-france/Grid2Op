# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import json
import re
import warnings
import unittest
import numpy as np
from abc import ABC, abstractmethod

import grid2op
from grid2op.tests.helper_path_test import *

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import *
from grid2op.Action import *
from grid2op.Rules import RulesChecker, DefaultRules
from grid2op.Space import GridObjects
from grid2op.Space.space_utils import save_to_dict

# TODO check that if i set the element of a powerline to -1, then it's working as intended (disconnect both ends)


def _get_action_grid_class():
    GridObjects._clear_class_attribute()
    GridObjects.env_name = "test_action_env"
    GridObjects.n_busbar_per_sub = 2
    GridObjects.n_gen = 5
    GridObjects.name_gen = np.array(["gen_{}".format(i) for i in range(5)])
    GridObjects.n_load = 11
    GridObjects.name_load = np.array(["load_{}".format(i) for i in range(11)])
    GridObjects.n_line = 20
    GridObjects.name_line = np.array(["line_{}".format(i) for i in range(20)])
    GridObjects.n_sub = 14
    GridObjects.name_sub = np.array(["sub_{}".format(i) for i in range(14)])
    GridObjects.sub_info = np.array(
        [3, 7, 5, 6, 5, 6, 3, 2, 5, 3, 3, 3, 4, 3], dtype=dt_int
    )
    GridObjects.load_to_subid = np.array([1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
    GridObjects.gen_to_subid = np.array([0, 1, 2, 5, 7])
    GridObjects.line_or_to_subid = np.array(
        [0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 8, 8, 9, 11, 12]
    )
    GridObjects.line_ex_to_subid = np.array(
        [1, 4, 2, 3, 4, 3, 4, 6, 8, 5, 10, 11, 12, 7, 8, 9, 13, 10, 12, 13]
    )
    GridObjects.load_to_sub_pos = np.array([4, 2, 5, 4, 4, 4, 1, 1, 1, 2, 1])
    GridObjects.gen_to_sub_pos = np.array([2, 5, 3, 5, 1])
    GridObjects.line_or_to_sub_pos = np.array(
        [0, 1, 1, 2, 3, 1, 2, 3, 4, 3, 1, 2, 3, 1, 2, 2, 3, 0, 0, 1]
    )
    GridObjects.line_ex_to_sub_pos = np.array(
        [0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 3, 0, 1, 2, 2, 0, 0, 0]
    )
    GridObjects.load_pos_topo_vect = np.array(
        [7, 12, 20, 25, 30, 41, 43, 46, 49, 53, 56]
    )
    GridObjects.gen_pos_topo_vect = np.array([2, 8, 13, 31, 36])
    GridObjects.line_or_pos_topo_vect = np.array(
        [0, 1, 4, 5, 6, 11, 17, 18, 19, 24, 27, 28, 29, 33, 34, 39, 40, 42, 48, 52]
    )
    GridObjects.line_ex_pos_topo_vect = np.array(
        [3, 21, 10, 15, 22, 16, 23, 32, 37, 26, 47, 50, 54, 35, 38, 44, 57, 45, 51, 55]
    )

    GridObjects.redispatching_unit_commitment_available = True
    GridObjects.gen_type = np.array(["thermal"] * 5)
    GridObjects.gen_pmin = np.array([0.0] * 5)
    GridObjects.gen_pmax = np.array([100.0] * 5)
    GridObjects.gen_min_uptime = np.array([0] * 5)
    GridObjects.gen_min_downtime = np.array([0] * 5)
    GridObjects.gen_cost_per_MW = np.array([70.0] * 5)
    GridObjects.gen_startup_cost = np.array([0.0] * 5)
    GridObjects.gen_shutdown_cost = np.array([0.0] * 5)
    GridObjects.gen_redispatchable = np.array([True, False, False, True, False])
    GridObjects.gen_max_ramp_up = np.array([10.0, 5.0, 15.0, 7.0, 8.0])
    GridObjects.gen_max_ramp_down = np.array([11.0, 6.0, 16.0, 8.0, 9.0])
    GridObjects.gen_renewable = ~GridObjects.gen_redispatchable

    GridObjects.n_storage = 2
    GridObjects.name_storage = np.array(["storage_0", "storage_1"])
    GridObjects.storage_to_subid = np.array([1, 2])
    GridObjects.storage_to_sub_pos = np.array([6, 4])
    GridObjects.storage_pos_topo_vect = np.array([9, 14])
    GridObjects.storage_type = np.array(["battery"] * 2)
    GridObjects.storage_Emax = np.array([100.0, 100.0])
    GridObjects.storage_Emin = np.array([0.0, 0.0])
    GridObjects.storage_max_p_prod = np.array([10.0, 10.0])
    GridObjects.storage_max_p_absorb = np.array([15.0, 15.0])
    GridObjects.storage_marginal_cost = np.array([0.0, 0.0])
    GridObjects.storage_loss = np.array([0.0, 0.0])
    GridObjects.storage_discharging_efficiency = np.array([1.0, 1.0])
    GridObjects.storage_charging_efficiency = np.array([1.0, 1.0])

    GridObjects._topo_vect_to_sub = np.repeat(
        np.arange(GridObjects.n_sub), repeats=GridObjects.sub_info
    )
    GridObjects.glop_version = grid2op.__version__
    GridObjects._PATH_GRID_CLASSES = None

    json_ = {
        "glop_version": grid2op.__version__,
        "n_busbar_per_sub": "2",
        "name_gen": ["gen_0", "gen_1", "gen_2", "gen_3", "gen_4"],
        "name_load": [
            "load_0",
            "load_1",
            "load_2",
            "load_3",
            "load_4",
            "load_5",
            "load_6",
            "load_7",
            "load_8",
            "load_9",
            "load_10",
        ],
        "name_line": [
            "line_0",
            "line_1",
            "line_2",
            "line_3",
            "line_4",
            "line_5",
            "line_6",
            "line_7",
            "line_8",
            "line_9",
            "line_10",
            "line_11",
            "line_12",
            "line_13",
            "line_14",
            "line_15",
            "line_16",
            "line_17",
            "line_18",
            "line_19",
        ],
        "name_sub": [
            "sub_0",
            "sub_1",
            "sub_2",
            "sub_3",
            "sub_4",
            "sub_5",
            "sub_6",
            "sub_7",
            "sub_8",
            "sub_9",
            "sub_10",
            "sub_11",
            "sub_12",
            "sub_13",
        ],
        "name_storage": ["storage_0", "storage_1"],
        "env_name": "test_action_env",
        "sub_info": [3, 7, 5, 6, 5, 6, 3, 2, 5, 3, 3, 3, 4, 3],
        "load_to_subid": [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13],
        "gen_to_subid": [0, 1, 2, 5, 7],
        "line_or_to_subid": [
            0,
            0,
            1,
            1,
            1,
            2,
            3,
            3,
            3,
            4,
            5,
            5,
            5,
            6,
            6,
            8,
            8,
            9,
            11,
            12,
        ],
        "line_ex_to_subid": [
            1,
            4,
            2,
            3,
            4,
            3,
            4,
            6,
            8,
            5,
            10,
            11,
            12,
            7,
            8,
            9,
            13,
            10,
            12,
            13,
        ],
        "storage_to_subid": [1, 2],
        "load_to_sub_pos": [4, 2, 5, 4, 4, 4, 1, 1, 1, 2, 1],
        "gen_to_sub_pos": [2, 5, 3, 5, 1],
        "line_or_to_sub_pos": [
            0,
            1,
            1,
            2,
            3,
            1,
            2,
            3,
            4,
            3,
            1,
            2,
            3,
            1,
            2,
            2,
            3,
            0,
            0,
            1,
        ],
        "line_ex_to_sub_pos": [
            0,
            0,
            0,
            0,
            1,
            1,
            2,
            0,
            0,
            0,
            2,
            2,
            3,
            0,
            1,
            2,
            2,
            0,
            0,
            0,
        ],
        "storage_to_sub_pos": [6, 4],
        "load_pos_topo_vect": [7, 12, 20, 25, 30, 41, 43, 46, 49, 53, 56],
        "gen_pos_topo_vect": [2, 8, 13, 31, 36],
        "line_or_pos_topo_vect": [
            0,
            1,
            4,
            5,
            6,
            11,
            17,
            18,
            19,
            24,
            27,
            28,
            29,
            33,
            34,
            39,
            40,
            42,
            48,
            52,
        ],
        "line_ex_pos_topo_vect": [
            3,
            21,
            10,
            15,
            22,
            16,
            23,
            32,
            37,
            26,
            47,
            50,
            54,
            35,
            38,
            44,
            57,
            45,
            51,
            55,
        ],
        "storage_pos_topo_vect": [9, 14],
        "gen_type": ["thermal"] * 5,
        "gen_pmin": [0.0] * 5,
        "gen_pmax": [100.0] * 5,
        "gen_redispatchable": [True, False, False, True, False],
        "gen_renewable": [False, True, True, False, True],
        "gen_max_ramp_up": [10.0, 5.0, 15.0, 7.0, 8.0],
        "gen_max_ramp_down": [11.0, 6.0, 16.0, 8.0, 9.0],
        "gen_min_uptime": [0] * 5,
        "gen_min_downtime": [0] * 5,
        "gen_cost_per_MW": [70.0] * 5,
        "gen_startup_cost": [0.0] * 5,
        "gen_shutdown_cost": [0.0] * 5,
        "load_size":[0.0]*11,
        "load_flexible":[False]*11,
        "load_max_ramp_up": [0.0]*11,
        "load_max_ramp_down": [0.0]*11,
        "load_min_uptime": [0.0]*11,
        "load_min_downtime": [0.0]*11,
        "load_cost_per_MW": [0.0]*11,
        "storage_type": ["battery"] * 2,
        "storage_Emax": [100.0, 100.0],
        "storage_Emin": [0.0, 0.0],
        "storage_max_p_prod": [10.0, 10.0],
        "storage_max_p_absorb": [15.0, 15.0],
        "storage_marginal_cost": [0.0, 0.0],
        "storage_loss": [0.0, 0.0],
        "storage_charging_efficiency": [1.0, 1.0],
        "storage_discharging_efficiency": [1.0, 1.0],
        "grid_layout": None,
        "shunt_to_subid": None,
        "name_shunt": None,
        "dim_alarms": 0,
        "alarms_area_names": [],
        "alarms_lines_area": {},
        "alarms_area_lines": [],
        "dim_alerts": 0,
        "alertable_line_names": [],
        "alertable_line_ids": [],
        "_PATH_GRID_CLASSES": None,
        "assistant_warning_type": None
    }
    GridObjects.shunts_data_available = False
    my_cls = GridObjects.init_grid(GridObjects, force=True)
    GridObjects._clear_class_attribute()
    return my_cls, json_


class TestActionBase(ABC):
    @abstractmethod
    def _action_setup(self):
        pass

    def _skipMissingKey(self, key):
        if key not in self.authorized_keys:
            unittest.TestCase.skipTest(self, f"Skipped: Missing authorized_key {key}")

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

        self.ActionSpaceClass = ActionSpace.init_grid(GridObjects_cls)
        act_cls = self._action_setup()
        self.helper_action = self.ActionSpaceClass(
            GridObjects_cls,
            legal_action=self.game_rules.legal_action,
            actionClass=act_cls,
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

    def tearDown(self):
        self.authorized_keys = {}
        type(self.gridobj)._clear_class_attribute()
        GridObjects._clear_class_attribute()

    def test_reset_modified_flags(self):
        act = self.helper_action.sample()
        act._reset_modified_flags()
        assert not act._modif_inj
        assert not act._modif_set_bus
        assert not act._modif_change_bus
        assert not act._modif_set_status
        assert not act._modif_change_status
        assert not act._modif_redispatch
        assert not act._modif_storage
        assert not act._modif_curtailment

    def test_get_array_from_attr_name(self):
        act = self.helper_action.sample()
        with self.assertRaises(Grid2OpException):
            act._get_array_from_attr_name("toto")

    def test_assign_attr_from_name(self):
        key = "set_line_status"
        if key in self.authorized_keys:
            unittest.TestCase.skipTest(
                self, f'Skipped: key "{key}" is present, this test supposes it\'s not'
            )
        act = self.helper_action.sample()
        with self.assertRaises(AmbiguousAction):
            act._assign_attr_from_name(
                "_set_line_status", np.zeros(self.helper_action.n_line)
            )

    def test_eq_none(self):
        act = self.helper_action.sample()
        assert not (act == None)

    def test_eq_diff_grid(self):
        act = self.helper_action.sample()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_ref = grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__)
            act_ref = env_ref.action_space()
        assert not (act == act_ref)

    def test_from_vect_nan(self):
        """test of the issue https://github.com/rte-france/Grid2Op/issues/173"""
        vect = np.full(self.helper_action.n, fill_value=np.NaN, dtype=float)
        if self.helper_action.n > 0:
            with self.assertRaises(NonFiniteElement):
                act = self.helper_action.from_vect(vect)
        else:
            act = self.helper_action.from_vect(vect)
            dn_act = self.helper_action()
            assert act == dn_act

    def compare_vect(self, pred, true):
        return np.max(np.abs(pred - true)) <= self.tolvect

    def test_call(self):
        action = self.helper_action()
        (
            dict_injection,
            set_status,
            switch_status,
            set_topo_vect,
            switcth_topo_vect,
            redispatching,
            flexibility,
            storage,
            shunts,
        ) = action()

    def test_compare(self):
        action = self.helper_action()
        action2 = self.helper_action()
        assert action == action2

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
        assert action.size() == self.size_act

    def test_size_action_space(self):
        assert self.helper_action.size() == self.size_act

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
        self._skipMissingKey("injection")

        new_vect = np.random.randn(self.helper_action.n_load).astype(dt_float)
        action = self.helper_action({"injection": {"load_p": new_vect}})
        self.compare_vect(action._dict_inj["load_p"], new_vect)
        for i in range(self.helper_action.n_load):
            assert action.effect_on(load_id=i)["new_p"] == new_vect[i]

    def test_change_v(self):
        """

        :return:
        """
        self._skipMissingKey("injection")

        new_vect = np.random.randn(self.helper_action.n_gen).astype(dt_float)
        action = self.helper_action({"injection": {"prod_v": new_vect}})
        self.compare_vect(action._dict_inj["prod_v"], new_vect)
        for i in range(self.helper_action.n_gen):
            assert action.effect_on(gen_id=i)["new_v"] == new_vect[i]

    def test_change_p_q(self):
        """

        :return:
        """
        self._skipMissingKey("injection")

        new_vect = np.random.randn(self.helper_action.n_load).astype(dt_float)
        new_vect2 = np.random.randn(self.helper_action.n_load).astype(dt_float)
        action = self.helper_action(
            {"injection": {"load_p": new_vect, "load_q": new_vect2}}
        )
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
        self._skipMissingKey("set_line_status")

        for i in range(self.helper_action.n_line):
            disco = np.full(shape=self.helper_action.n_line, fill_value=0, dtype=dt_int)
            disco[i] = 1
            action = self.helper_action({"set_line_status": disco})
            for j in range(self.helper_action.n_line):
                assert (
                    action.effect_on(line_id=j)["set_line_status"] == disco[j]
                ), "problem with line {} if line {} is disconnected".format(j, i)
                assert action.effect_on(line_id=j)["change_line_status"] == False

    def test_update_disconnection_m1(self):
        """
        Test if the disconnection is working properly
        :return:
        """
        self._skipMissingKey("set_line_status")

        for i in range(self.helper_action.n_line):
            disco = np.full(shape=self.helper_action.n_line, fill_value=0, dtype=dt_int)
            disco[i] = -1
            action = self.helper_action({"set_line_status": disco})
            for j in range(self.helper_action.n_line):
                assert (
                    action.effect_on(line_id=j)["set_line_status"] == disco[j]
                ), "problem with line {} if line {} is disconnected".format(j, i)
                assert action.effect_on(line_id=j)["change_line_status"] == False

    def test_update_hazard(self):
        """
        Same test as above, but with hazard
        :return:
        """
        self._skipMissingKey("hazards")

        for i in range(self.helper_action.n_line):
            disco = np.full(
                shape=self.helper_action.n_line, fill_value=False, dtype=dt_bool
            )
            disco[i] = True
            action = self.helper_action({"hazards": disco})
            for j in range(self.helper_action.n_line):
                expected_res = -1 if j == i else 0
                assert (
                    action.effect_on(line_id=j)["set_line_status"] == expected_res
                ), "problem with line {} if line {} is disconnected".format(j, i)
                assert action.effect_on(line_id=j)["change_line_status"] == False

    def test_update_status(self):
        self._skipMissingKey("change_line_status")

        for i in range(self.helper_action.n_line):
            disco = np.full(
                shape=self.helper_action.n_line, fill_value=False, dtype=dt_bool
            )
            disco[i] = True
            action = self.helper_action({"change_line_status": disco})
            for j in range(self.helper_action.n_line):
                expected_res = j == i
                assert action.effect_on(line_id=j)["set_line_status"] == 0
                assert action.effect_on(line_id=j)["change_line_status"] == expected_res

    def test_update_set_topo_by_dict_obj(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("change_bus")

        action = self.helper_action({"set_bus": {"loads_id": [(1, 2)]}})
        assert action.effect_on(load_id=1)["set_bus"] == 2
        assert action.effect_on(load_id=1)["change_bus"] == False
        assert action.effect_on(load_id=0)["set_bus"] == 0
        assert action.effect_on(load_id=0)["change_bus"] == False

    def test_update_set_topo_by_dict_sub(self):
        self._skipMissingKey("set_bus")

        arr = np.array([1, 1, 1, 2, 2, 2, 2], dtype=dt_int)
        action = self.helper_action({"set_bus": {"substations_id": [(1, arr)]}})
        assert action.effect_on(line_id=2)["set_bus_or"] == 1
        assert action.effect_on(line_id=3)["set_bus_or"] == 1
        assert action.effect_on(line_id=4)["set_bus_or"] == 2
        assert action.effect_on(line_id=0)["set_bus_ex"] == 1
        assert action.effect_on(load_id=0)["set_bus"] == 2
        assert action.effect_on(gen_id=1)["set_bus"] == 2
        assert action.effect_on(storage_id=0)["set_bus"] == 2

        assert action.effect_on(load_id=1)["set_bus"] == 0
        assert action.effect_on(gen_id=0)["set_bus"] == 0

    def test_update_set_topo_by_dict_sub2(self):
        self._skipMissingKey("set_bus")

        arr = np.array([1, 1, 1, 2, 2, 2, 2], dtype=dt_int)
        arr3 = np.array([1, 2, 1, 2, 1, 2], dtype=dt_int)
        action = self.helper_action(
            {"set_bus": {"substations_id": [(3, arr3), (1, arr)]}}
        )
        assert action.effect_on(line_id=2)["set_bus_or"] == 1
        assert action.effect_on(line_id=3)["set_bus_or"] == 1
        assert action.effect_on(line_id=4)["set_bus_or"] == 2
        assert action.effect_on(line_id=0)["set_bus_ex"] == 1
        assert action.effect_on(load_id=0)["set_bus"] == 2
        assert action.effect_on(gen_id=1)["set_bus"] == 2

        assert action.effect_on(load_id=1)["set_bus"] == 0
        assert action.effect_on(gen_id=0)["set_bus"] == 0

    def test_update_undo_change_bus(self):
        self._skipMissingKey("change_bus")
        self._skipMissingKey("set_bus")

        # Create dummy change_bus action
        action = self.helper_action({"change_bus": {"loads_id": [1]}})
        # Check it is valid
        assert action.effect_on(load_id=0)["set_bus"] == 0
        assert action.effect_on(load_id=0)["change_bus"] == False
        assert action.effect_on(load_id=1)["set_bus"] == 0
        assert action.effect_on(load_id=1)["change_bus"] == True
        # Save a copy
        action_copy = copy.deepcopy(action)

        # Update it
        action.update({"change_bus": {"loads_id": [1]}})
        # Check it's updated
        assert action.effect_on(load_id=0)["set_bus"] == 0
        assert action.effect_on(load_id=0)["change_bus"] == False
        assert action.effect_on(load_id=1)["set_bus"] == 0
        assert action.effect_on(load_id=1)["change_bus"] == False

        # Update back to original
        action.update({"change_bus": {"loads_id": [1]}})
        # Check it's updated
        assert action.effect_on(load_id=0)["set_bus"] == 0
        assert action.effect_on(load_id=0)["change_bus"] == False
        assert action.effect_on(load_id=1)["set_bus"] == 0
        assert action.effect_on(load_id=1)["change_bus"] == True

        # Check it's equal to original
        assert action == action_copy

    def test_update_change_bus_by_dict_obj(self):
        self._skipMissingKey("change_bus")
        self._skipMissingKey("set_bus")

        action = self.helper_action({"change_bus": {"loads_id": [1]}})
        assert action.effect_on(load_id=1)["set_bus"] == 0
        assert action.effect_on(load_id=1)["change_bus"] == True
        assert action.effect_on(load_id=0)["set_bus"] == 0
        assert action.effect_on(load_id=0)["change_bus"] == False

    def test_update_change_bus_by_dict_sub(self):
        self._skipMissingKey("change_bus")

        arr = np.array([True, True, True, False, False, False, False], dtype=dt_bool)
        action = self.helper_action({"change_bus": {"substations_id": [(1, arr)]}})
        assert action.effect_on(line_id=2)["change_bus_or"] == True
        assert action.effect_on(line_id=3)["change_bus_or"] == True
        assert action.effect_on(line_id=4)["change_bus_or"] == False
        assert action.effect_on(line_id=0)["change_bus_ex"] == True
        assert action.effect_on(load_id=0)["change_bus"] == False
        assert action.effect_on(gen_id=1)["change_bus"] == False
        assert action.effect_on(storage_id=0)["change_bus"] == False

        assert action.effect_on(load_id=1)["change_bus"] == False
        assert action.effect_on(gen_id=0)["change_bus"] == False

    def test_update_change_bus_by_dict_sub2(self):
        self._skipMissingKey("change_bus")

        arr = np.array([True, True, True, False, False, False, False], dtype=dt_bool)
        arr3 = np.array([True, False, True, False, True, False], dtype=dt_bool)
        action = self.helper_action(
            {"change_bus": {"substations_id": [(3, arr3), (1, arr)]}}
        )
        assert action.effect_on(line_id=2)["change_bus_or"] == True
        assert action.effect_on(line_id=3)["change_bus_or"] == True
        assert action.effect_on(line_id=4)["change_bus_or"] == False
        assert action.effect_on(line_id=0)["change_bus_ex"] == True
        assert action.effect_on(load_id=0)["change_bus"] == False
        assert action.effect_on(gen_id=1)["change_bus"] == False

        assert action.effect_on(load_id=1)["change_bus"] == False
        assert action.effect_on(gen_id=0)["change_bus"] == False

    def test_ambiguity_topo(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("change_bus")

        # i switch the bus of the origin of powerline 1
        action = self.helper_action({"change_bus": {"lines_or_id": [1]}})
        action.update(
            {"set_bus": {"lines_or_id": [(1, 1)]}}
        )  # i set the origin of powerline 1 to bus 1
        try:
            action()
            raise RuntimeError("This should hav thrown an InvalidBusStatus error")
        except InvalidBusStatus as e:
            pass

    def test_ambiguity_line_status_when_set_and_change(self):
        self._skipMissingKey("set_line_status")
        self._skipMissingKey("change_line_status")

        arr = np.zeros(self.helper_action.n_line, dtype=dt_int)
        arr[1] = -1

        # i switch set the status of powerline 1 to "disconnected"
        action = self.helper_action({"set_line_status": arr})
        action.update({"change_line_status": [1]})  # i asked to change this status
        try:
            action()
            raise RuntimeError("This should hav thrown an InvalidBusStatus error")
        except InvalidLineStatus as e:
            pass

    def test_ambiguity_line_reconnected_without_bus(self):
        self.skipTest("deprecated with backend action")
        self._skipMissingKey("set_line_status")
        arr = np.zeros(self.helper_action.n_line)
        arr[1] = 1
        action = self.helper_action(
            {"set_line_status": arr}
        )  # i switch set the status of powerline 1 to "connected"
        # and i don't say on which bus to connect it

        try:
            action()
            raise RuntimeError(
                "This should have thrown an InvalidBusStatus error for {}"
                "".format(self.helper_action.actionClass)
            )
        except InvalidLineStatus as e:
            pass

    def test_set_status_and_setbus_isambiguous(self):
        """

        :return:
        """
        self._skipMissingKey("set_bus")
        self._skipMissingKey("set_line_status")

        arr = np.array([1, 1, 1, 2, 2, 2, 0], dtype=dt_int)
        id_ = 2
        action = self.helper_action({"set_bus": {"substations_id": [(1, arr)]}})
        arr2 = np.zeros(self.helper_action.n_line, dtype=dt_int)
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
        self._skipMissingKey("set_bus")
        self._skipMissingKey("hazards")

        arr = np.array([1, 1, 1, 2, 2, 2, 2], dtype=dt_int)
        id_ = 2
        action = self.helper_action({"set_bus": {"substations_id": [(1, arr)]}})
        assert action.effect_on(line_id=id_)["set_bus_or"] == 1, "fail for {}".format(
            self.helper_action.actionClass
        )
        action.update({"hazards": [id_]})
        assert action.effect_on(line_id=id_)["set_bus_or"] == 0, "fail for {}".format(
            self.helper_action.actionClass
        )
        assert (
            action.effect_on(line_id=id_)["set_line_status"] == -1
        ), "fail for {}".format(self.helper_action.actionClass)
        assert action.effect_on(line_id=id_)["set_bus_ex"] == 0, "fail for {}".format(
            self.helper_action.actionClass
        )

    def test_action_str(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("change_bus")

        arr1 = np.array([False, False, False, True, True, True, True], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        id_1 = 1
        id_2 = 12
        action = self.helper_action(
            {
                "change_bus": {"substations_id": [(id_1, arr1)]},
                "set_bus": {"substations_id": [(id_2, arr2)]},
            }
        )
        res = action.__str__()
        act_str = (
            "This action will:\n\t - NOT change anything to the injections"
            "\n\t - NOT perform any redispatching action\n"
            "\t - NOT modify any storage capacity\n"
            "\t - NOT perform any curtailment"
            "\n\t - NOT force any line status\n"
            "\t - NOT switch any line status"
            "\n\t - Change the bus of the following element(s):"
            "\n\t \t - Switch bus of line (origin) id 4 [on substation 1]"
            "\n\t \t - Switch bus of load id 0 [on substation 1]"
            "\n\t \t - Switch bus of generator id 1 [on substation 1]"
            "\n\t \t - Switch bus of storage id 0 [on substation 1]"
            "\n\t - Set the bus of the following element(s):"
            "\n\t \t - Assign bus 1 to line (extremity) id 18 [on substation 12]"
            "\n\t \t - Assign bus 1 to line (origin) id 19 [on substation 12]"
            "\n\t \t - Assign bus 2 to load id 9 [on substation 12]"
            "\n\t \t - Assign bus 2 to line (extremity) id 12 [on substation 12]"
        )
        assert res == act_str

    def test_to_vect(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("change_bus")

        arr1 = np.array([False, False, False, True, True, True, False], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        id_1 = 1
        id_2 = 12

        action = self.helper_action(
            {
                "change_bus": {"substations_id": [(id_1, arr1)]},
                "set_bus": {"substations_id": [(id_2, arr2)]},
            }
        )

        res = action.to_vect()
        tmp = np.zeros(self.size_act)
        if "curtail" in action.authorized_keys:
            # for curtailment, at the end, and by default its -1
            tmp[-action.n_gen :] = -1

        # compute the "set_bus" vect
        id_set = np.nonzero(np.array(type(action).attr_list_vect) == "_set_topo_vect")[0][0]
        size_before = 0
        for el in type(action).attr_list_vect[:id_set]:
            arr_ = action._get_array_from_attr_name(el)
            size_before += arr_.shape[0]
        tmp[size_before : (size_before + action.dim_topo)] = np.array(
            [
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
                1,
                1,
                2,
                2,
                0,
                0,
                0,
            ]
        )
        id_change = np.nonzero(np.array(type(action).attr_list_vect) == "_change_bus_vect")[0][
            0
        ]
        size_before = 0
        for el in type(action).attr_list_vect[:id_change]:
            arr_ = action._get_array_from_attr_name(el)
            size_before += arr_.shape[0]
        tmp[size_before : (size_before + action.dim_topo)] = 1.0 * np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )
        assert np.all(res[np.isfinite(tmp)] == tmp[np.isfinite(tmp)])
        assert np.all(np.isfinite(res) == np.isfinite(tmp))

    def test__eq__(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("change_bus")

        arr1 = np.array([False, False, False, True, True, True, True], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        id_1 = 1
        id_2 = 12
        action1 = self.helper_action(
            {
                "change_bus": {"substations_id": [(id_1, arr1)]},
                "set_bus": {"substations_id": [(id_2, arr2)]},
            }
        )
        action2 = self.helper_action(
            {
                "change_bus": {"substations_id": [(id_1, arr1)]},
                "set_bus": {"substations_id": [(id_2, arr2)]},
            }
        )
        action3 = self.helper_action()
        test = type(action1).assert_grid_correct_cls()
        assert action1 == action2
        assert action1 != action3

    def test_from_vect_dn(self):
        action1 = self.helper_action({})
        action2 = self.helper_action({})

        vect_act1 = action1.to_vect()
        action2.from_vect(vect_act1)
        # if i load an action with from_vect it's equal to the original one
        assert action1 == action2

        vect_act2 = action2.to_vect()
        assert np.all(
            vect_act1[np.isfinite(vect_act2)] == vect_act2[np.isfinite(vect_act2)]
        )
        assert np.all(np.isfinite(vect_act1) == np.isfinite(vect_act2))

    def test_from_vect_change_bus(self):
        arr1 = np.array([False, False, False, True, True, True, False], dtype=dt_bool)
        id_1 = 1

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            action1 = self.helper_action(
                {"change_bus": {"substations_id": [(id_1, arr1)]}}
            )
        action2 = self.helper_action({})

        vect_act1 = action1.to_vect()
        action2.from_vect(vect_act1)
        # if i load an action with from_vect it's equal to the original one
        assert action1 == action2

        vect_act2 = action2.to_vect()
        # if i convert it back to a vector, it's equal to the original converted vector
        assert np.all(
            vect_act1[np.isfinite(vect_act2)] == vect_act2[np.isfinite(vect_act2)]
        )
        assert np.all(np.isfinite(vect_act1) == np.isfinite(vect_act2))

    def test_from_vect_set_bus(self):
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        id_2 = 12
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            action1 = self.helper_action(
                {"set_bus": {"substations_id": [(id_2, arr2)]}}
            )
        action2 = self.helper_action({})

        vect_act1 = action1.to_vect()
        action2.from_vect(vect_act1)
        # if i load an action with from_vect it's equal to the original one
        assert action1 == action2

        vect_act2 = action2.to_vect()
        # if i convert it back to a vector, it's equal to the original converted vector
        assert np.all(
            vect_act1[np.isfinite(vect_act2)] == vect_act2[np.isfinite(vect_act2)]
        )
        assert np.all(np.isfinite(vect_act1) == np.isfinite(vect_act2))

    def test_from_vect_set_line_status(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            action1 = self.helper_action({"set_line_status": [(1, -1)]})
        action2 = self.helper_action({})

        vect_act1 = action1.to_vect()
        action2.from_vect(vect_act1)
        # if i load an action with from_vect it's equal to the original one
        assert action1 == action2

        vect_act2 = action2.to_vect()
        # if i convert it back to a vector, it's equal to the original converted vector
        assert np.all(
            vect_act1[np.isfinite(vect_act2)] == vect_act2[np.isfinite(vect_act2)]
        )
        assert np.all(np.isfinite(vect_act1) == np.isfinite(vect_act2))

    def test_from_vect_change_line_status(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            action1 = self.helper_action({"change_line_status": [2]})
        action2 = self.helper_action({})

        vect_act1 = action1.to_vect()
        action2.from_vect(vect_act1)
        # if i load an action with from_vect it's equal to the original one
        assert action1 == action2

        vect_act2 = action2.to_vect()
        # if i convert it back to a vector, it's equal to the original converted vector
        assert np.all(
            vect_act1[np.isfinite(vect_act2)] == vect_act2[np.isfinite(vect_act2)]
        )
        assert np.all(np.isfinite(vect_act1) == np.isfinite(vect_act2))

    def test_from_vect_redisp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            action1 = self.helper_action({"redispatch": [(0, -7.42)]})
        action2 = self.helper_action({})

        vect_act1 = action1.to_vect()
        action2.from_vect(vect_act1)
        # if i load an action with from_vect it's equal to the original one
        assert action1 == action2

        vect_act2 = action2.to_vect()
        # if i convert it back to a vector, it's equal to the original converted vector
        assert np.all(
            vect_act1[np.isfinite(vect_act2)] == vect_act2[np.isfinite(vect_act2)]
        )
        assert np.all(np.isfinite(vect_act1) == np.isfinite(vect_act2))

    def test_from_vect_storage(self):
        """test from vect also work with storage action"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            action1 = self.helper_action({"set_storage": [(0, -7.42)]})
        action2 = self.helper_action({})

        vect_act1 = action1.to_vect()
        action2.from_vect(vect_act1)
        # if i load an action with from_vect it's equal to the original one
        assert action1 == action2

        vect_act2 = action2.to_vect()
        # if i convert it back to a vector, it's equal to the original converted vector
        assert np.all(
            vect_act1[np.isfinite(vect_act2)] == vect_act2[np.isfinite(vect_act2)]
        )
        assert np.all(np.isfinite(vect_act1) == np.isfinite(vect_act2))

    def test_from_vect(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("change_bus")

        arr1 = np.array([False, False, False, True, True, True, False], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        id_1 = 1
        id_2 = 12
        action1 = self.helper_action(
            {
                "change_bus": {"substations_id": [(id_1, arr1)]},
                "set_bus": {"substations_id": [(id_2, arr2)]},
            }
        )
        action2 = self.helper_action({})

        vect_act1 = action1.to_vect()
        action2.from_vect(vect_act1)
        # if i load an action with from_vect it's equal to the original one
        assert action1 == action2
        vect_act2 = action2.to_vect()

        # if i convert it back to a vector, it's equal to the original converted vector
        assert np.all(
            vect_act1[np.isfinite(vect_act2)] == vect_act2[np.isfinite(vect_act2)]
        )
        assert np.all(np.isfinite(vect_act1) == np.isfinite(vect_act2))

    def test_call_change_set(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("change_bus")
        self._skipMissingKey("set_line_status")
        self._skipMissingKey("change_line_status")
        self._skipMissingKey("injection")

        arr1 = np.array([False, False, False, True, True, True, True], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        id_1 = 1
        id_2 = 12
        new_vect = np.random.randn(self.helper_action.n_load).astype(dt_int)
        new_vect2 = np.random.randn(self.helper_action.n_load).astype(dt_int)

        change_status_orig = np.random.randint(0, 2, self.helper_action.n_line).astype(
            dt_bool
        )
        set_status_orig = np.random.randint(-1, 2, self.helper_action.n_line)
        set_status_orig[change_status_orig] = 0

        change_topo_vect_orig = np.random.randint(
            0, 2, self.helper_action.dim_topo
        ).astype(dt_bool)
        # powerline that are set to be reconnected, can't be moved to another bus
        change_topo_vect_orig[
            self.helper_action.line_or_pos_topo_vect[set_status_orig == 1]
        ] = False
        change_topo_vect_orig[
            self.helper_action.line_ex_pos_topo_vect[set_status_orig == 1]
        ] = False
        # powerline that are disconnected, can't be moved to the other bus
        change_topo_vect_orig[
            self.helper_action.line_or_pos_topo_vect[set_status_orig == -1]
        ] = False
        change_topo_vect_orig[
            self.helper_action.line_ex_pos_topo_vect[set_status_orig == -1]
        ] = False

        set_topo_vect_orig = np.random.randint(0, 3, self.helper_action.dim_topo)
        set_topo_vect_orig[change_topo_vect_orig] = 0  # don't both change and set
        # I need to make sure powerlines that are reconnected are indeed reconnected to a bus
        set_topo_vect_orig[
            self.helper_action.line_or_pos_topo_vect[set_status_orig == 1]
        ] = 1
        set_topo_vect_orig[
            self.helper_action.line_ex_pos_topo_vect[set_status_orig == 1]
        ] = 1
        # I need to make sure powerlines that are disconnected are not assigned to a bus
        set_topo_vect_orig[
            self.helper_action.line_or_pos_topo_vect[set_status_orig == -1]
        ] = 0
        set_topo_vect_orig[
            self.helper_action.line_ex_pos_topo_vect[set_status_orig == -1]
        ] = 0

        action = self.helper_action(
            {
                "change_bus": change_topo_vect_orig,
                "set_bus": set_topo_vect_orig,
                "injection": {"load_p": new_vect, "load_q": new_vect2},
                "change_line_status": change_status_orig,
                "set_line_status": set_status_orig,
            }
        )
        (
            dict_injection,
            set_status,
            change_status,
            set_topo_vect,
            switcth_topo_vect,
            redispatching,
            storage,
            shunts,
        ) = action()

        assert "load_p" in dict_injection
        assert np.all(dict_injection["load_p"] == new_vect)
        assert "load_q" in dict_injection
        assert np.all(dict_injection["load_q"] == new_vect2)

        assert np.all(set_status == set_status_orig)
        assert np.all(change_status == change_status_orig)
        assert np.all(set_topo_vect == set_topo_vect_orig)
        assert np.all(switcth_topo_vect == change_topo_vect_orig)

    def test_get_topological_impact(self):
        self._skipMissingKey("set_bus")
        self._skipMissingKey("change_bus")
        self._skipMissingKey("set_line_status")
        self._skipMissingKey("change_line_status")

        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True, True], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=dt_bool)
        arr_line1[id_line] = True
        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=dt_int)
        arr_line2[id_line2] = 1

        do_nothing = self.helper_action({})
        aff_lines, aff_subs = do_nothing.get_topological_impact()
        assert np.sum(aff_lines) == 0
        assert np.sum(aff_subs) == 0

        act_sub1 = self.helper_action(
            {"change_bus": {"substations_id": [(id_1, arr1)]}}
        )
        aff_lines, aff_subs = act_sub1.get_topological_impact()
        assert np.sum(aff_lines) == 0
        assert np.sum(aff_subs) == 1
        assert aff_subs[id_1]

        act_sub1_sub12 = self.helper_action(
            {
                "change_bus": {"substations_id": [(id_1, arr1)]},
                "set_bus": {"substations_id": [(id_2, arr2)]},
            }
        )
        aff_lines, aff_subs = act_sub1_sub12.get_topological_impact()
        assert np.sum(aff_lines) == 0
        assert np.sum(aff_subs) == 2
        assert aff_subs[id_1]
        assert aff_subs[id_2]

        act_sub1_sub12_line1 = self.helper_action(
            {
                "change_bus": {"substations_id": [(id_1, arr1)]},
                "set_bus": {"substations_id": [(id_2, arr2)]},
                "change_line_status": arr_line1,
            }
        )
        aff_lines, aff_subs = act_sub1_sub12_line1.get_topological_impact()
        assert np.sum(aff_lines) == 1
        assert aff_lines[id_line] == 1
        assert np.sum(aff_subs) == 2
        assert aff_subs[id_1]
        assert aff_subs[id_2]

        act_sub1_sub12_line1_line2 = self.helper_action(
            {
                "change_bus": {"substations_id": [(id_1, arr1)]},
                "set_bus": {"substations_id": [(id_2, arr2)]},
                "change_line_status": arr_line1,
                "set_line_status": arr_line2,
            }
        )
        aff_lines, aff_subs = act_sub1_sub12_line1_line2.get_topological_impact()
        assert np.sum(aff_lines) == 2
        assert aff_lines[id_line] == 1
        assert aff_lines[id_line2] == 1
        assert np.sum(aff_subs) == 2
        assert aff_subs[id_1]
        assert aff_subs[id_2]

    def test_to_dict(self):
        dict_ = self.helper_action.cls_to_dict()
        for el in dict_:
            assert el in self.res, f"missing key {el} in self.res"
        for el in self.res:
            assert el in dict_, f"missing key {el} in dict_"
            
        for el in self.res:
            val = dict_[el]
            val_res = self.res[el]
            if val is None and val_res is not None:
                raise AssertionError(f"val is None and val_res is not None: val_res: {val_res}")
            if val is not None and val_res is None:
                raise AssertionError(f"val is not None and val_res is None: val {val}")
            if val is None and val_res is None:
                continue
            
            ok_ = np.array_equal(val, val_res)
            assert ok_, (f"values different for {el}: "
                         f"{dict_[el]}"
                         f"{self.res[el]}")

    def test_from_dict(self):
        res = ActionSpace.from_dict(self.res)
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
        assert np.all(
            res.line_or_pos_topo_vect == self.helper_action.line_or_pos_topo_vect
        )
        assert np.all(
            res.line_ex_pos_topo_vect == self.helper_action.line_ex_pos_topo_vect
        )
        # pdb.set_trace()
        assert issubclass(res.actionClass, self.helper_action._init_subtype)

    def test_json_serializable(self):
        dict_ = self.helper_action.cls_to_dict()
        res = json.dumps(obj=dict_, indent=4, sort_keys=True)

    def test_json_loadable(self):
        dict_ = self.helper_action.cls_to_dict()
        tmp = json.dumps(obj=dict_, indent=4, sort_keys=True)
        res = ActionSpace.from_dict(json.loads(tmp))

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
        assert np.all(
            res.line_or_pos_topo_vect == self.helper_action.line_or_pos_topo_vect
        )
        assert np.all(
            res.line_ex_pos_topo_vect == self.helper_action.line_ex_pos_topo_vect
        )
        assert issubclass(res.actionClass, self.helper_action._init_subtype)

    def test_as_dict(self):
        act = self.helper_action({})
        dict_ = act.as_dict()
        assert dict_ == {}

    def test_to_from_vect_action(self):
        act = self.helper_action({})
        vect_ = act.to_vect()
        act2 = self.helper_action.from_vect(vect_)
        assert act == act2

    def test_sum_shape_equal_size(self):
        act = self.helper_action({})
        assert act.size() == np.sum(act.shapes())

    def test_shape_correct(self):
        act = self.helper_action({})
        assert act.shapes().shape == act.dtypes().shape

    def test_redispatching(self):
        self._skipMissingKey("redispatch")

        act = self.helper_action({"redispatch": (1, 10)})
        act = self.helper_action({"redispatch": [(1, 10), (2, 100)]})
        act = self.helper_action(
            {"redispatch": np.array([10.0, 20.0, 30.0, 40.0, 50.0])}
        )

    def test_possibility_reconnect_powerlines(self):
        self._skipMissingKey("set_line_status")
        self._skipMissingKey("set_bus")
        self.helper_action.legal_action = DefaultRules()

        act = self.helper_action.reconnect_powerline(line_id=1, bus_or=1, bus_ex=1)
        line_impact, sub_impact = act.get_topological_impact()
        assert np.sum(line_impact) == 1
        assert np.sum(sub_impact) == 0

        act = self.helper_action.reconnect_powerline(line_id=1, bus_or=1, bus_ex=1)
        act.update({"set_bus": {"generators_id": [(1, 2)]}})
        line_impact, sub_impact = act.get_topological_impact()
        assert np.sum(line_impact) == 1
        assert np.all(sub_impact == [False, True] + [False for _ in range(12)])

        act = self.helper_action.reconnect_powerline(line_id=1, bus_or=1, bus_ex=1)
        act.update({"set_bus": {"generators_id": [(0, 2)]}})
        line_impact, sub_impact = act.get_topological_impact()
        assert np.sum(line_impact) == 1
        assert np.all(sub_impact == [True] + [False for _ in range(13)])

        # there were a bug that occurred when updating an action, some vectors were not reset
        act = self.helper_action.reconnect_powerline(line_id=1, bus_or=1, bus_ex=1)
        line_impact, sub_impact = act.get_topological_impact()
        assert np.sum(line_impact) == 1
        assert np.sum(sub_impact) == 0
        act.update({"set_bus": {"generators_id": [(1, 2)]}})
        line_impact, sub_impact = act.get_topological_impact()
        assert np.sum(line_impact) == 1
        assert np.all(sub_impact == [False, True] + [False for _ in range(12)])

    def test_extract_from_vect(self):
        self._skipMissingKey("set_line_status")
        act = self.helper_action()
        vect = act.to_vect()
        res = self.helper_action.extract_from_vect(vect, "_set_line_status")
        assert np.all(res == act._set_line_status)

    def test_sample(self):
        try:
            for i in range(30):
                act = self.helper_action.sample()
        except Exception as exc_:
            assert False, f"sample() raised error {exc_}"


class TestAction(TestActionBase, unittest.TestCase):
    """
    Test suite using the BaseAction class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=BaseAction)
        return BaseAction


class TestTopologyAction(TestActionBase, unittest.TestCase):
    """
    Test suite using the TopologyAction class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=TopologyAction)
        return TopologyAction


class TestDispatchAction(TestActionBase, unittest.TestCase):
    """
    Test suite using the DispatchAction class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=DispatchAction)
        return DispatchAction


class TestTopologyAndDispatchAction(TestActionBase, unittest.TestCase):
    """
    Test suite using the TopologyAndDispatchAction class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=TopologyAndDispatchAction)
        return TopologyAndDispatchAction


class TestTopologySetAction(TestActionBase, unittest.TestCase):
    """
    Test suite using the TopologySetAction class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=TopologySetAction)
        return TopologySetAction


class TestTopologySetAndDispatchAction(TestActionBase, unittest.TestCase):
    """
    Test suite using the TopologySetAndDispatchAction class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=TopologySetAndDispatchAction)
        return TopologySetAndDispatchAction


class TestTopologyChangeAction(TestActionBase, unittest.TestCase):
    """
    Test suite using the TopologySetAction class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=TopologyChangeAction)
        return TopologyChangeAction


class TestTopologyChangeAndDispatchAction(TestActionBase, unittest.TestCase):
    """
    Test suite using the TopologyChangeAndDispatchAction class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=TopologyChangeAndDispatchAction)
        return TopologyChangeAndDispatchAction


class TestPowerlineSetAction(TestActionBase, unittest.TestCase):
    """
    Test suite using the PowerlineSetAction class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=PowerlineSetAction)
        return PowerlineSetAction


class TestPowerlineChangeAction(TestActionBase, unittest.TestCase):
    """
    Test suite using the PowerlineChangeAction class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=PowerlineChangeAction)
        return PowerlineChangeAction


class TestPowerlineSetAndDispatchAction(TestActionBase, unittest.TestCase):
    """
    Test suite using the PowerlineSetAction class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=PowerlineSetAndDispatchAction)
        return PowerlineSetAndDispatchAction


class TestPowerlineChangeAndDispatchAction(TestActionBase, unittest.TestCase):
    """
    Test suite using the PowerlineChangeAction class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=PowerlineChangeAndDispatchAction)
        return PowerlineChangeAndDispatchAction


class TestDontAct(TestActionBase, unittest.TestCase):
    """
    Test suite using the DontAct class
    """

    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=DontAct)
        return DontAct


class TestIADD:
    """Test that act += act2 equals what we want and don't use data it is not allowed to"""

    def _skipMissingKey(self, key, act):
        if key not in act.attr_list_set:
            unittest.TestCase.skipTest(self, "Skipped: Missing authorized_key {key}")

    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.game_rules = RulesChecker()
        self.n_line = 20
        GridObjects_cls, self.res = _get_action_grid_class()
        self.gridobj = GridObjects_cls()
        self.n_line = self.gridobj.n_line

        self.ActionSpaceClass = ActionSpace.init_grid(self.gridobj)
        # self.size_act = 229
        self.action_space_1 = self.get_action_space_1()
        self.action_space_2 = self.get_action_space_2()
        np.random.seed(42)

    def aux_get_act(self, helper_action):
        template_act = helper_action()
        dict_act = {}
        tmp_inj = {}
        if "load_p" in template_act.attr_list_set:
            tmp_inj["load_p"] = np.random.randn(helper_action.n_load).astype(dt_float)
        if "load_q" in template_act.attr_list_set:
            tmp_inj["load_q"] = np.random.randn(helper_action.n_load).astype(dt_float)
        if "prod_p" in template_act.attr_list_set:
            tmp_inj["prod_p"] = np.random.randn(helper_action.n_gen).astype(dt_float)
        if "prod_v" in template_act.attr_list_set:
            tmp_inj["prod_v"] = np.random.randn(helper_action.n_gen).astype(dt_float)
        if dict_act:
            dict_act["injection"] = tmp_inj

        if "_hazards" in template_act.attr_list_set:
            dict_act["hazards"] = np.random.choice(
                [True, False], helper_action.n_line
            ).astype(dt_bool)
        if "_maintenance" in template_act.attr_list_set:
            dict_act["maintenance"] = np.random.choice(
                [True, False], helper_action.n_line
            ).astype(dt_bool)
        if "_redispatch" in template_act.attr_list_set:
            dict_act["redispatch"] = np.random.randn(helper_action.n_gen).astype(
                dt_float
            )
            dict_act["redispatch"] -= np.mean(dict_act["redispatch"])
        if "_set_line_status" in template_act.attr_list_set:
            # i dont test the line reconnection...
            dict_act["set_line_status"] = np.random.choice(
                [-1, 0], helper_action.n_line
            ).astype(dt_int)
        if "_switch_line_status" in template_act.attr_list_set:
            # i dont test the line reconnection...
            dict_act["change_line_status"] = np.random.choice(
                [True, False], helper_action.n_line
            ).astype(dt_bool)
        if "_set_topo_vect" in template_act.attr_list_set:
            # i dont test the line reconnection...
            dict_act["set_bus"] = np.random.choice(
                [1, 2], helper_action.dim_topo
            ).astype(dt_int)
        if "_change_bus_vect" in template_act.attr_list_set:
            # i dont test the line reconnection...
            dict_act["change_bus"] = np.random.choice(
                [True, False], helper_action.dim_topo
            ).astype(dt_bool)
        return helper_action(dict_act)

    def test_iadd_modify(self):

        act1_init = self.aux_get_act(self.action_space_1)
        act1 = copy.deepcopy(act1_init)
        act2 = self.aux_get_act(self.action_space_2)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            if act2.attr_list_set - act1.attr_list_set:
                # it should raise a warning if i attempt to set an attribute it's not supposed to
                with self.assertWarns(UserWarning):
                    act1 += act2
            else:
                # i can add it i check it's properly added, without warnings
                act1 += act2
                # i now test all attributes have been modified for attributes in both
                for attr_nm in act1.attr_list_set & act2.attr_list_set:
                    assert np.any(
                        act1.__dict__[attr_nm] != act1_init.__dict__[attr_nm]
                    ), "error, attr {} has not been updated".format(attr_nm)

                # for all in act1 not in act2, nothing should have changed
                for attr_nm in act1.attr_list_set - act2.attr_list_set:
                    if attr_nm == "_set_line_status" or attr_nm == "_set_topo_vect":
                        # these vector can be changed if act2 allows for "change_line_status" or "change_bus"
                        # TODO improve these tests
                        continue
                    if (
                        attr_nm == "_switch_line_status"
                        or attr_nm == "_change_bus_vect"
                    ):
                        # these vector can be changed if act2 allows for "set_line_status" or "set_bus"
                        # TODO improve these tests
                        continue
                    assert np.all(
                        act1.__dict__[attr_nm] == act1_init.__dict__[attr_nm]
                    ), "error, attr {} has been updated".format(attr_nm)

    def test_iadd_change_set_status(self):
        self._skipMissingKey("change_line_status", self.action_space_1)
        self._skipMissingKey("set_line_status", self.action_space_2)

        act1_init = self.aux_get_act(self.action_space_1)
        act1 = copy.deepcopy(act1_init)
        act2 = self.aux_get_act(self.action_space_2)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # proper warnings are handled above
            act1 += act2
        assert np.sum(act1._switch_line_status[act2._set_line_status != 0]) == 0

    def test_iadd_set_change_status(self):
        self._skipMissingKey("set_line_status", self.action_space_1)
        self._skipMissingKey("change_line_status", self.action_space_2)

        act1_init = self.aux_get_act(self.action_space_1)
        act1 = copy.deepcopy(act1_init)
        act2 = self.aux_get_act(self.action_space_2)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # proper warnings are handled above
            act1 += act2
        # if act1 set a line but act2 change it, then it's equivalent to act1 set to the other value
        indx_change = (act1._set_line_status != 0) & (act2._switch_line_status)
        assert np.all(
            act1._set_line_status[indx_change]
            != act1_init._set_line_status[indx_change]
        )
        # if act1 does not set, but act2 change, then act1 is not affected (for set)
        indx_same = (act1._set_line_status == 0) & (act2._switch_line_status)
        assert np.all(
            act1._set_line_status[indx_same] == act1_init._set_line_status[indx_same]
        )

    def test_iadd_change_set_bus(self):
        self._skipMissingKey("change_bus", self.action_space_1)
        self._skipMissingKey("set_bus", self.action_space_2)

        act1_init = self.aux_get_act(self.action_space_1)
        act1 = copy.deepcopy(act1_init)
        act2 = self.aux_get_act(self.action_space_2)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # proper warnings are handled above
            act1 += act2
        assert np.sum(act1._change_bus_vect[act2._set_topo_vect != 0]) == 0

    def test_iadd_set_change_bus(self):
        self._skipMissingKey("set_bus", self.action_space_1)
        self._skipMissingKey("change_bus", self.action_space_2)

        act1_init = self.aux_get_act(self.action_space_1)
        act1 = copy.deepcopy(act1_init)
        act2 = self.aux_get_act(self.action_space_2)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # proper warnings are handled above
            act1 += act2
        # if act1 set a line but act2 change it, then it's equivalent to act1 set to the other value
        indx_change = (act1._set_topo_vect != 0) & (act2._change_bus_vect)
        assert np.all(
            act1._set_topo_vect[indx_change] != act1_init._set_topo_vect[indx_change]
        )
        # if act1 does not set, but act2 change, then act1 is not affected (for set)
        indx_same = (act1._set_topo_vect == 0) & (act2._change_bus_vect)
        assert np.all(
            act1._set_topo_vect[indx_same] == act1_init._set_topo_vect[indx_same]
        )

    def test_iadd_empty_change_bus(self):
        self._skipMissingKey("change_bus", self.action_space_1)

        act1 = self.action_space_1({})
        act2 = self.action_space_1({"change_bus": {"substations_id": [(0, [0, 1])]}})

        # Iadd change
        act1 += act2

        assert act2._change_bus_vect[0] == True
        assert act2._change_bus_vect[1] == True
        assert act1._change_bus_vect[0] == True
        assert act1._change_bus_vect[1] == True
        assert np.any(act1._set_topo_vect != 0) == False

    def test_iadd_change_change_bus(self):
        self._skipMissingKey("change_bus", self.action_space_1)

        act1 = self.action_space_1({"change_bus": {"substations_id": [(0, [0, 1])]}})

        act2 = self.action_space_1({"change_bus": {"substations_id": [(0, [0, 1])]}})

        # Iadd change
        act1 += act2

        assert act2._change_bus_vect[0] == True
        assert act2._change_bus_vect[1] == True
        assert act1._change_bus_vect[0] == False
        assert act1._change_bus_vect[1] == False
        assert np.any(act1._set_topo_vect != 0) == False

    def test_add_modify(self):

        act1_init = self.aux_get_act(self.action_space_1)
        act1 = copy.deepcopy(act1_init)
        act2 = self.aux_get_act(self.action_space_2)
        list_2 = copy.deepcopy(act2.attr_list_set)
        list_1 = copy.deepcopy(act1.attr_list_set)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            if len(list_2 - list_1):
                # it should raise a warning if i attempt to set an attribute it's not supposed to
                with self.assertWarns(UserWarning):
                    res = act1 + act2
            else:
                # i can add it i check it's properly added, without warnings
                res = act1 + act2
                # i now test all attributes have been modified for attributes in both
                for attr_nm in res.attr_list_set & act2.attr_list_set:
                    assert np.any(
                        getattr(res, attr_nm) != getattr(act1_init, attr_nm)
                    ), "error, attr {} has not been updated (case 0)".format(attr_nm)

                # for all in act1 not in act2, nothing should have changed
                for attr_nm in res.attr_list_set - act2.attr_list_set:
                    if attr_nm == "_set_line_status" or attr_nm == "_set_topo_vect":
                        # these vector can be changed if act2 allows for "change_line_status" or "change_bus"
                        # TODO improve these tests
                        continue
                    if (
                        attr_nm == "_switch_line_status"
                        or attr_nm == "_change_bus_vect"
                    ):
                        # these vector can be changed if act2 allows for "set_line_status" or "set_bus"
                        # TODO improve these tests
                        continue
                    assert np.all(
                        getattr(res, attr_nm) == getattr(act1_init, attr_nm)
                    ), "error, attr {} has been updated (case 1)".format(attr_nm)

    def test_add_change_set_status(self):
        self._skipMissingKey("change_line_status", self.action_space_1)
        self._skipMissingKey("set_line_status", self.action_space_2)

        act1_init = self.aux_get_act(self.action_space_1)
        act1 = copy.deepcopy(act1_init)
        act2 = self.aux_get_act(self.action_space_2)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # proper warnings are handled above
            res = act1 + act2
        assert np.sum(res._switch_line_status[act2._set_line_status != 0]) == 0

    def test_add_set_change_status(self):
        self._skipMissingKey("set_line_status", self.action_space_1)
        self._skipMissingKey("change_line_status", self.action_space_2)

        act1_init = self.aux_get_act(self.action_space_1)
        act1 = copy.deepcopy(act1_init)
        act2 = self.aux_get_act(self.action_space_2)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # proper warnings are handled above
            res = act1 + act2
        # if act1 set a line but act2 change it, then it's equivalent to act1 set to the other value
        indx_change = (res._set_line_status != 0) & (act2._switch_line_status)
        assert np.all(
            res._set_line_status[indx_change] != act1_init._set_line_status[indx_change]
        )
        # if act1 does not set, but act2 change, then act1 is not affected (for set)
        indx_same = (res._set_line_status == 0) & (act2._switch_line_status)
        assert np.all(
            res._set_line_status[indx_same] == act1_init._set_line_status[indx_same]
        )

    def test_add_change_set_bus(self):
        self._skipMissingKey("change_bus", self.action_space_1)
        self._skipMissingKey("set_bus", self.action_space_2)

        act1_init = self.aux_get_act(self.action_space_1)
        act1 = copy.deepcopy(act1_init)
        act2 = self.aux_get_act(self.action_space_2)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # proper warnings are handled above
            res = act1 + act2
        assert np.sum(res._change_bus_vect[act2._set_topo_vect != 0]) == 0

    def test_add_set_change_bus(self):
        self._skipMissingKey("set_bus", self.action_space_1)
        self._skipMissingKey("change_bus", self.action_space_2)

        act1_init = self.aux_get_act(self.action_space_1)
        act1 = copy.deepcopy(act1_init)
        act2 = self.aux_get_act(self.action_space_2)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # proper warnings are handled above
            res = act1 + act2

        # if act1 set a line but act2 change it, then it's equivalent to act1 set to the other value
        indx_change = (res._set_topo_vect != 0) & (act2._change_bus_vect)
        assert np.all(
            res._set_topo_vect[indx_change] != act1_init._set_topo_vect[indx_change]
        )
        # if act1 does not set, but act2 change, then act1 is not affected (for set)
        indx_same = (res._set_topo_vect == 0) & (act2._change_bus_vect)
        assert np.all(
            res._set_topo_vect[indx_same] == act1_init._set_topo_vect[indx_same]
        )

    def test_add_empty_change_bus(self):
        self._skipMissingKey("change_bus", self.action_space_1)

        act1 = self.action_space_1({})
        act2 = self.action_space_1({"change_bus": {"substations_id": [(0, [0, 1])]}})

        # add change
        res = act1 + act2

        assert act2._change_bus_vect[0] == True
        assert act2._change_bus_vect[1] == True
        assert res._change_bus_vect[0] == True
        assert res._change_bus_vect[1] == True
        assert np.any(res._set_topo_vect != 0) == False

    def test_add_change_change_bus(self):
        self._skipMissingKey("change_bus", self.action_space_1)

        act1 = self.action_space_1({"change_bus": {"substations_id": [(0, [0, 1])]}})

        act2 = self.action_space_1({"change_bus": {"substations_id": [(0, [0, 1])]}})

        # add change
        res = act1 + act2

        assert act2._change_bus_vect[0] == True
        assert act2._change_bus_vect[1] == True
        assert res._change_bus_vect[0] == False
        assert res._change_bus_vect[1] == False
        assert np.any(res._set_topo_vect != 0) == False


# TODO a generic method to build them all maybe ?
class TestDontAct_PowerlineChangeAndDispatchAction(TestIADD, unittest.TestCase):
    """
    Test suite using the DontAct class
    """

    def get_action_space_1(self):
        return self.ActionSpaceClass(
            self.gridobj, legal_action=self.game_rules.legal_action, actionClass=DontAct
        )

    def get_action_space_2(self):
        return self.ActionSpaceClass(
            self.gridobj,
            legal_action=self.game_rules.legal_action,
            actionClass=PowerlineChangeAndDispatchAction,
        )


class TestPowerlineChangeAndDispatchAction_PowerlineChangeAndDispatchAction(
    TestIADD, unittest.TestCase
):
    """
    Test suite using the DontAct class
    """

    def get_action_space_1(self):
        return self.ActionSpaceClass(
            self.gridobj,
            legal_action=self.game_rules.legal_action,
            actionClass=PowerlineChangeAndDispatchAction,
        )

    def get_action_space_2(self):
        return self.ActionSpaceClass(
            self.gridobj,
            legal_action=self.game_rules.legal_action,
            actionClass=PowerlineChangeAndDispatchAction,
        )


class TestTopologyAndDispatchAction_PowerlineChangeAndDispatchAction(
    TestIADD, unittest.TestCase
):
    """
    Test suite using the DontAct class
    """

    def get_action_space_1(self):
        return self.ActionSpaceClass(
            self.gridobj,
            legal_action=self.game_rules.legal_action,
            actionClass=TopologyAndDispatchAction,
        )

    def get_action_space_2(self):
        return self.ActionSpaceClass(
            self.gridobj,
            legal_action=self.game_rules.legal_action,
            actionClass=PowerlineChangeAndDispatchAction,
        )


class TestTopologicalImpact(unittest.TestCase):
    # weird stuf to avoid copy paste and inheritance
    def _action_setup(self):
        return TopologyAndDispatchAction

    def setUp(self):
        TestActionBase.setUp(self)

    def test_no_inj_possible(self):
        donothing = self.helper_action()
        donothing._dict_inj["prod_p"] = 0.0
        with self.assertRaises(AmbiguousAction):
            _ = donothing()

    def test_get_topo_imp_dn(self):
        donothing = self.helper_action()
        lines_impacted, subs_impacted = donothing.get_topological_impact()
        assert np.sum(lines_impacted) == 0
        assert np.sum(subs_impacted) == 0

    def test_get_topo_imp_changebus(self):
        changelor = self.helper_action({"change_bus": {"lines_or_id": [0]}})
        lines_impacted, subs_impacted = changelor.get_topological_impact()
        assert np.sum(lines_impacted) == 0
        assert np.sum(subs_impacted) == 1
        assert subs_impacted[0]
        changelex = self.helper_action({"change_bus": {"lines_ex_id": [0]}})
        lines_impacted, subs_impacted = changelex.get_topological_impact()
        assert np.sum(lines_impacted) == 0
        assert np.sum(subs_impacted) == 1
        assert subs_impacted[1]

        changeload = self.helper_action({"change_bus": {"loads_id": [0]}})
        lines_impacted, subs_impacted = changeload.get_topological_impact()
        assert np.sum(lines_impacted) == 0
        assert np.sum(subs_impacted) == 1
        assert subs_impacted[1]

        changegen = self.helper_action({"change_bus": {"generators_id": [0]}})
        lines_impacted, subs_impacted = changegen.get_topological_impact()
        assert np.sum(lines_impacted) == 0
        assert np.sum(subs_impacted) == 1
        assert subs_impacted[0]

    def test_get_topo_imp_setbus(self):
        changelor = self.helper_action({"set_bus": {"lines_or_id": [(0, 2)]}})
        lines_impacted, subs_impacted = changelor.get_topological_impact()
        assert np.sum(lines_impacted) == 0
        assert np.sum(subs_impacted) == 1
        assert subs_impacted[0]
        changelex = self.helper_action({"set_bus": {"lines_ex_id": [(0, 2)]}})
        lines_impacted, subs_impacted = changelex.get_topological_impact()
        assert np.sum(lines_impacted) == 0
        assert np.sum(subs_impacted) == 1
        assert subs_impacted[1]

        changeload = self.helper_action({"set_bus": {"loads_id": [(0, 2)]}})
        lines_impacted, subs_impacted = changeload.get_topological_impact()
        assert np.sum(lines_impacted) == 0
        assert np.sum(subs_impacted) == 1
        assert subs_impacted[1]

        changegen = self.helper_action({"set_bus": {"generators_id": [(0, 2)]}})
        lines_impacted, subs_impacted = changegen.get_topological_impact()
        assert np.sum(lines_impacted) == 0
        assert np.sum(subs_impacted) == 1
        assert subs_impacted[0]

    def test_get_topo_imp_changestatus(self):
        changelor = self.helper_action({"change_line_status": [7]})
        lines_impacted, subs_impacted = changelor.get_topological_impact()
        assert np.sum(lines_impacted) == 1
        assert np.sum(subs_impacted) == 0
        assert lines_impacted[7]

    def test_get_topo_imp_setstatus_down(self):
        changelor = self.helper_action({"set_line_status": [(9, -1)]})
        lines_impacted, subs_impacted = changelor.get_topological_impact()
        assert np.sum(lines_impacted) == 1
        assert np.sum(subs_impacted) == 0
        assert lines_impacted[9]

    def test_get_topo_imp_setstatus_up(self):
        changelor = self.helper_action(
            {
                "set_line_status": [(9, 1)],
                "set_bus": {"lines_or_id": [(9, 2)], "lines_ex_id": [(9, 2)]},
            }
        )
        lines_impacted, subs_impacted = changelor.get_topological_impact()
        assert np.sum(lines_impacted) == 1
        assert np.sum(subs_impacted) == 0
        assert lines_impacted[9]

    def test_get_topo_imp_setstatus_up_2(self):
        # i set a status (no substation impacted, one line impacted)
        # plus I change the topology of one element (one sub impacted, no line impacted)
        l_id = 0
        changelor = self.helper_action(
            {
                "set_line_status": [(l_id, 1)],
                "set_bus": {"lines_or_id": [(l_id, 2)], "lines_ex_id": [(l_id, 2)]},
            }
        )
        changelor.update({"set_bus": {"generators_id": [(0, 2)]}})
        lines_impacted, subs_impacted = changelor.get_topological_impact()
        assert np.sum(lines_impacted) == 1
        assert np.sum(subs_impacted) == 1
        assert lines_impacted[l_id]
        assert subs_impacted[0]

    def test_get_topo_imp_setstatus_up_alreadyup(self):
        l_id = 15
        powerline_status = np.full(self.n_line, fill_value=True, dtype=dt_bool)
        changelor = self.helper_action(
            {
                "set_line_status": [(l_id, 1)],
                "set_bus": {"lines_or_id": [(l_id, 1)], "lines_ex_id": [(l_id, 1)]},
            }
        )
        # powerline is already connected
        # i fake connect it, so it's like changing both its ends
        lines_impacted, subs_impacted = changelor.get_topological_impact(
            powerline_status
        )
        assert np.sum(lines_impacted) == 1
        assert np.sum(subs_impacted) == 2
        assert lines_impacted[l_id]
        assert subs_impacted[self.res["line_or_to_subid"][l_id]]
        assert subs_impacted[self.res["line_ex_to_subid"][l_id]]

    def test_get_topo_imp_setstatus_up_alreadyup2(self):
        # change a line that is already reconnected (2 subs) + 1 line
        # and i change the object on same substation, should count as 0
        l_id = 0
        powerline_status = np.full(self.n_line, fill_value=True, dtype=dt_bool)
        changelor = self.helper_action(
            {
                "set_line_status": [(l_id, 1)],
                "set_bus": {"lines_or_id": [(l_id, 1)], "lines_ex_id": [(l_id, 1)]},
            }
        )
        changelor.update({"set_bus": {"generators_id": [(0, 2)]}})
        # powerline is already connected
        # i fake connect it, so it's like changing both its ends
        lines_impacted, subs_impacted = changelor.get_topological_impact(
            powerline_status
        )
        assert np.sum(lines_impacted) == 1
        assert np.sum(subs_impacted) == 2
        assert lines_impacted[l_id]
        assert subs_impacted[self.res["line_or_to_subid"][l_id]]
        assert subs_impacted[self.res["line_ex_to_subid"][l_id]]
        assert subs_impacted[0]

    def test_get_topo_imp_setstatus_up_isdown(self):
        l_id = 15
        powerline_status = np.full(self.n_line, fill_value=True, dtype=dt_bool)
        powerline_status[l_id] = False
        changelor = self.helper_action(
            {
                "set_line_status": [(l_id, 1)],
                "set_bus": {"lines_or_id": [(l_id, 1)], "lines_ex_id": [(l_id, 1)]},
            }
        )
        # this is a real reconnection, is concerns only the powerline
        lines_impacted, subs_impacted = changelor.get_topological_impact(
            powerline_status
        )
        assert np.sum(lines_impacted) == 1
        assert np.sum(subs_impacted) == 0
        assert lines_impacted[l_id]

    def test_get_topo_imp_setstatus_down_alreadydown(self):
        self.skipTest("does not pass but should imho")
        l_id = 12
        powerline_status = np.full(self.n_line, fill_value=True, dtype=dt_bool)
        powerline_status[l_id] = False
        changelor = self.helper_action({"set_line_status": [(l_id, -1)]})
        # powerline is already disconnected
        lines_impacted, subs_impacted = changelor.get_topological_impact(
            powerline_status
        )
        assert np.sum(lines_impacted) == 0
        assert np.sum(subs_impacted) == 0

    def test_get_topo_imp_setstatus_down_isup(self):
        l_id = 3
        powerline_status = np.full(self.n_line, fill_value=True, dtype=dt_bool)
        changelor = self.helper_action({"set_line_status": [(l_id, -1)]})
        # this is a real reconnection, is concerns only the powerline
        lines_impacted, subs_impacted = changelor.get_topological_impact(
            powerline_status
        )
        assert np.sum(lines_impacted) == 1
        assert np.sum(subs_impacted) == 0
        assert lines_impacted[l_id]


class TestDeepCopy(unittest.TestCase):
    def test_alarm(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("l2rpn_icaps_2021", test=True, _add_to_name=type(self).__name__) as env:
                act = env.action_space()
                act.raise_alarm = [0]

                cpy = copy.deepcopy(act)
                assert cpy == act
                assert np.all(cpy.raise_alarm == [True, False, False])

    def test_redisp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("l2rpn_icaps_2021", test=True, _add_to_name=type(self).__name__) as env:
                act = env.action_space()
                act.redispatch = [(0, -1.0)]

                cpy = copy.deepcopy(act)
                assert cpy == act
                assert cpy.redispatch[0] == -1.0

    def test_storage(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("educ_case14_storage", test=True, _add_to_name=type(self).__name__) as env:
                act = env.action_space()
                act.storage_p = [(0, -1.0)]

                cpy = copy.deepcopy(act)
                assert cpy == act
                assert cpy.storage_p[0] == -1.0

    def test_topo(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__) as env:
                # set line status
                act = env.action_space()
                act.set_line_status = [(0, -1)]
                cpy = copy.deepcopy(act)
                assert cpy == act
                assert cpy.set_line_status[0] == -1

                # change line status
                act = env.action_space()
                act.change_line_status = [0]
                cpy = copy.deepcopy(act)
                assert cpy == act
                assert cpy.change_line_status[0]

                # set_bus
                act = env.action_space()
                act.set_bus = [(0, 1), (1, 2), (2, 1)]
                cpy = copy.deepcopy(act)
                assert cpy == act
                assert np.all(cpy.set_bus[:3] == [1, 2, 1])

                # change_bus
                act = env.action_space()
                act.change_bus = [0, 1, 2]
                cpy = copy.deepcopy(act)
                assert cpy == act
                assert np.all(cpy.change_bus[:3] == [True, True, True])


if __name__ == "__main__":
    unittest.main()
