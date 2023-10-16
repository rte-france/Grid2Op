# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import numpy as np
import warnings
import grid2op

from grid2op.dtypes import dt_int
from grid2op.Exceptions import *
from grid2op.Action import BaseAction, ActionSpace, PlayableAction, BaseAction
from grid2op.Rules import RulesChecker
from grid2op.Space import GridObjects
import json
import tempfile


import pdb


def _get_action_grid_class():
    GridObjects.env_name = "test_action_serial_dict"
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

    GridObjects.redispatching_unit_commitment_availble = True
    GridObjects.gen_type = np.array(["thermal"] * 3 + ["wind"] * 2)
    GridObjects.gen_pmin = np.array([0.0] * 5)
    GridObjects.gen_pmax = np.array([100.0] * 5)
    GridObjects.gen_min_uptime = np.array([0] * 5)
    GridObjects.gen_min_downtime = np.array([0] * 5)
    GridObjects.gen_cost_per_MW = np.array([70.0] * 5)
    GridObjects.gen_startup_cost = np.array([0.0] * 5)
    GridObjects.gen_shutdown_cost = np.array([0.0] * 5)
    GridObjects.gen_redispatchable = np.array([True, True, True, False, False])
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
    GridObjects._PATH_ENV = None

    GridObjects.shunts_data_available = True
    GridObjects.n_shunt = 2
    GridObjects.shunt_to_subid = np.array([0, 1])
    GridObjects.name_shunt = np.array(["shunt_1", "shunt_2"])

    GridObjects.alarms_area_lines = [[el for el in GridObjects.name_line]]
    GridObjects.alarms_area_names = ["all"]
    GridObjects.alarms_lines_area = {el: ["all"] for el in GridObjects.name_line}
    GridObjects.dim_alarms = 1
    my_cls = GridObjects.init_grid(GridObjects, force=True)
    return my_cls


class TestActionSerialDict(unittest.TestCase):
    def _action_setup(self):
        # return self.ActionSpaceClass(self.gridobj, legal_action=self.game_rules.legal_action, actionClass=BaseAction)
        return BaseAction

    def tearDown(self):
        self.authorized_keys = {}
        self.gridobj._clear_class_attribute()
        ActionSpace._clear_class_attribute()

    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.game_rules = RulesChecker()

        GridObjects_cls = _get_action_grid_class()
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
        self.authorized_keys = self.helper_action().authorized_keys
        self.size_act = self.helper_action.size()

    def test_set_line_status(self):
        act = self.helper_action(
            {
                "set_line_status": [
                    (l_id, status) for l_id, status in zip([2, 4, 5], [1, -1, 1])
                ]
            }
        )
        dict_ = act.as_serializable_dict()
        act2 = self.helper_action(dict_)
        assert act == act2
        dict_2 = act.as_serializable_dict()
        assert dict_ == dict_2
        with tempfile.TemporaryFile(mode="w") as f:
            json.dump(fp=f, obj=dict_)

    def test_change_status(self):
        act = self.helper_action({"change_line_status": [l_id for l_id in [2, 4, 5]]})
        dict_ = act.as_serializable_dict()
        act2 = self.helper_action(dict_)
        assert act == act2
        dict_2 = act.as_serializable_dict()
        assert dict_ == dict_2
        with tempfile.TemporaryFile(mode="w") as f:
            json.dump(fp=f, obj=dict_)

    def test_set_bus(self):
        act = self.helper_action(
            {
                "set_bus": [
                    (el_id, status)
                    for el_id, status in zip([2, 4, 5, 8, 9, 10], [1, -1, 1, 2, 2, 1])
                ]
            }
        )
        dict_ = act.as_serializable_dict()
        act2 = self.helper_action(dict_)
        assert act == act2
        dict_2 = act.as_serializable_dict()
        assert dict_ == dict_2
        with tempfile.TemporaryFile(mode="w") as f:
            json.dump(fp=f, obj=dict_)

    def test_change_bus(self):
        act = self.helper_action(
            {"change_bus": [el_id for el_id in [2, 4, 5, 8, 9, 10]]}
        )
        dict_ = act.as_serializable_dict()
        act2 = self.helper_action(dict_)
        assert act == act2
        dict_2 = act.as_serializable_dict()
        assert dict_ == dict_2
        with tempfile.TemporaryFile(mode="w") as f:
            json.dump(fp=f, obj=dict_)

    def test_redispatch(self):
        act = self.helper_action(
            {
                "redispatch": [
                    (el_id, amount) for el_id, amount in zip([0, 2], [-3.0, 28.9])
                ]
            }
        )
        dict_ = act.as_serializable_dict()
        act2 = self.helper_action(dict_)
        assert act == act2
        dict_2 = act.as_serializable_dict()
        assert dict_ == dict_2
        with tempfile.TemporaryFile(mode="w") as f:
            json.dump(fp=f, obj=dict_)

    def test_curtail(self):
        act = self.helper_action(
            {"curtail": [(el_id, amount) for el_id, amount in zip([3, 4], [0.5, 0.7])]}
        )
        dict_ = act.as_serializable_dict()
        act2 = self.helper_action(dict_)
        assert act == act2
        dict_2 = act.as_serializable_dict()
        assert dict_ == dict_2
        with tempfile.TemporaryFile(mode="w") as f:
            json.dump(fp=f, obj=dict_)

    def test_set_storage(self):
        act = self.helper_action(
            {
                "set_storage": [
                    (el_id, amount) for el_id, amount in zip([0, 1], [-0.5, 0.7])
                ]
            }
        )
        dict_ = act.as_serializable_dict()
        act2 = self.helper_action(dict_)
        assert act == act2
        dict_2 = act.as_serializable_dict()
        assert dict_ == dict_2
        with tempfile.TemporaryFile(mode="w") as f:
            json.dump(fp=f, obj=dict_)

    def test_raise_alarm(self):
        act = self.helper_action({"raise_alarm": [0]})
        dict_ = act.as_serializable_dict()
        act2 = self.helper_action(dict_)
        assert act == act2
        dict_2 = act.as_serializable_dict()
        assert dict_ == dict_2
        with tempfile.TemporaryFile(mode="w") as f:
            json.dump(fp=f, obj=dict_)

    def test_injection(self):
        np.random.seed(0)
        act = self.helper_action(
            {
                "injection": {
                    "prod_p": np.random.uniform(size=self.helper_action.n_gen),
                    "prod_v": np.random.normal(size=self.helper_action.n_gen),
                    "load_p": np.random.lognormal(size=self.helper_action.n_load),
                    "load_q": np.random.logistic(size=self.helper_action.n_load),
                }
            }
        )
        dict_ = act.as_serializable_dict()
        act2 = self.helper_action(dict_)
        assert act == act2
        dict_2 = act.as_serializable_dict()
        assert dict_ == dict_2
        with tempfile.TemporaryFile(mode="w") as f:
            json.dump(fp=f, obj=dict_)

    def test_shunt(self):
        np.random.seed(0)
        act = self.helper_action(
            {
                "shunt": {
                    "shunt_p": np.random.uniform(size=self.helper_action.n_shunt),
                    "shunt_q": np.random.normal(size=self.helper_action.n_shunt),
                    "shunt_bus": [(0, 1), (1, 2)],
                }
            }
        )
        dict_ = act.as_serializable_dict()
        act2 = self.helper_action(dict_)
        assert act == act2
        dict_2 = act.as_serializable_dict()
        assert dict_ == dict_2
        with tempfile.TemporaryFile(mode="w") as f:
            json.dump(fp=f, obj=dict_)

    def test_iadd(self):
        """I add a bug when += a change_bus after a set bus"""
        act = self.helper_action(
            {
                "set_bus": [
                    (el_id, status)
                    for el_id, status in zip([2, 4, 5, 8, 9, 10], [1, -1, 1, 2, 2, 1])
                ]
            }
        )
        act += self.helper_action(
            {"change_bus": [el_id for el_id in [2, 4, 5, 8, 9, 10]]}
        )
        assert np.all(act._set_topo_vect <= 4)

    def test_all_at_once(self):
        np.random.seed(1)
        act = self.helper_action(
            {
                "set_line_status": [
                    (l_id, status) for l_id, status in zip([2, 4, 5], [1, -1, 1])
                ]
            }
        )
        act += self.helper_action({"change_line_status": [l_id for l_id in [6, 7, 8]]})
        act += self.helper_action(
            {
                "set_bus": [
                    (el_id, status)
                    for el_id, status in zip([2, 4, 5, 8, 9, 10], [1, -1, 1, 2, 2, 1])
                ]
            }
        )
        act += self.helper_action(
            {"change_bus": [el_id for el_id in [2, 3, 5, 11, 12, 15]]}
        )
        act += self.helper_action(
            {
                "redispatch": [
                    (el_id, amount) for el_id, amount in zip([0, 2], [-3.0, 28.9])
                ]
            }
        )
        act += self.helper_action(
            {"curtail": [(el_id, amount) for el_id, amount in zip([3, 4], [0.5, 0.7])]}
        )
        act += self.helper_action(
            {
                "set_storage": [
                    (el_id, amount) for el_id, amount in zip([0, 1], [-0.5, 0.7])
                ]
            }
        )
        act += self.helper_action({"raise_alarm": [0]})
        act += self.helper_action(
            {
                "injection": {
                    "prod_p": np.random.uniform(size=self.helper_action.n_gen),
                    "prod_v": np.random.normal(size=self.helper_action.n_gen),
                    "load_p": np.random.lognormal(size=self.helper_action.n_load),
                    "load_q": np.random.logistic(size=self.helper_action.n_load),
                }
            }
        )
        act += self.helper_action(
            {
                "shunt": {
                    "shunt_p": np.random.uniform(size=self.helper_action.n_shunt),
                    "shunt_q": np.random.normal(size=self.helper_action.n_shunt),
                    "shunt_bus": [(0, 1), (1, 2)],
                }
            }
        )
        dict_ = act.as_serializable_dict()
        act2 = self.helper_action(dict_)
        assert act == act2
        dict_2 = act.as_serializable_dict()
        assert dict_ == dict_2
        with tempfile.TemporaryFile(mode="w") as f:
            json.dump(fp=f, obj=dict_)


class TestMultiGrid(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env1 = grid2op.make("l2rpn_case14_sandbox",
                                     test=True,
                                     action_class=PlayableAction,
                                     _add_to_name=type(self).__name__+"env1")
            self.env2 = grid2op.make("educ_case14_storage",
                                     test=True,
                                     action_class=PlayableAction,
                                     _add_to_name=type(self).__name__+"env2")
        return super().setUp()
    def tearDown(self) -> None:
        self.env1.close()
        self.env2.close()
        return super().tearDown()

    def test_can_make_lineor(self):
        act : BaseAction = self.env1.action_space({"set_bus": {"lines_or_id": [(0, 2), (5, 1), (15, 2)]}})
        dict_ = act.as_serializable_dict()
        assert dict_ == {'set_bus': {'lines_or_id': [(0, 2), (5, 1), (15, 2)]}}
        act2 = self.env2.action_space(dict_)
        dict_2 = act2.as_serializable_dict()
        assert dict_ == dict_2
        
        act : BaseAction = self.env1.action_space({"change_bus": {"lines_or_id": [0, 5, 15]}})
        dict_ = act.as_serializable_dict()
        assert dict_ == {'change_bus': {'lines_or_id': [0, 5, 15]}}
        act2 = self.env2.action_space(dict_)
        dict_2 = act2.as_serializable_dict()
        assert dict_ == dict_2

    def test_can_make_lineex(self):
        act : BaseAction = self.env1.action_space({"set_bus": {"lines_ex_id": [(0, 2), (5, 1), (15, 2)]}})
        dict_ = act.as_serializable_dict()
        assert dict_ == {'set_bus': {'lines_ex_id': [(0, 2), (5, 1), (15, 2)]}}
        act2 = self.env2.action_space(dict_)
        dict_2 = act2.as_serializable_dict()
        assert dict_ == dict_2
        
        act : BaseAction = self.env1.action_space({"change_bus": {"lines_ex_id": [0, 5, 15]}})
        dict_ = act.as_serializable_dict()
        assert dict_ == {'change_bus': {'lines_ex_id': [0, 5, 15]}}
        act2 = self.env2.action_space(dict_)
        dict_2 = act2.as_serializable_dict()
        assert dict_ == dict_2

    def test_can_make_gen(self):
        act : BaseAction = self.env1.action_space({"set_bus": {"generators_id": [(0, 2), (5, 1)]}})
        dict_ = act.as_serializable_dict()
        assert dict_ == {'set_bus': {'generators_id': [(0, 2), (5, 1)]}}
        act2 = self.env2.action_space(dict_)
        dict_2 = act2.as_serializable_dict()
        assert dict_ == dict_2
        
        act : BaseAction = self.env1.action_space({"change_bus": {"generators_id": [0, 5]}})
        dict_ = act.as_serializable_dict()
        assert dict_ == {'change_bus': {'generators_id': [0, 5]}}
        act2 = self.env2.action_space(dict_)
        dict_2 = act2.as_serializable_dict()
        assert dict_ == dict_2

    def test_can_make_load(self):
        act : BaseAction = self.env1.action_space({"set_bus": {"loads_id": [(0, 2), (5, 1)]}})
        dict_ = act.as_serializable_dict()
        assert dict_ == {'set_bus': {'loads_id': [(0, 2), (5, 1)]}}
        act2 = self.env2.action_space(dict_)
        dict_2 = act2.as_serializable_dict()
        assert dict_ == dict_2
        
        act : BaseAction = self.env1.action_space({"change_bus": {"loads_id": [0, 5]}})
        dict_ = act.as_serializable_dict()
        assert dict_ == {'change_bus': {'loads_id': [0, 5]}}
        act2 = self.env2.action_space(dict_)
        dict_2 = act2.as_serializable_dict()
        assert dict_ == dict_2

    def test_with_gen_load_lineor_lineex(self):
        act : BaseAction = self.env1.action_space({"set_bus": {"loads_id": [(0, 2), (5, 1)],
                                                               "generators_id": [(0, 2), (5, 1)],
                                                               "lines_ex_id": [(0, 2), (5, 1), (15, 2)],
                                                               "lines_or_id": [(0, 2), (5, 1), (15, 2)]
                                                               }})
        dict_ = act.as_serializable_dict()
        assert dict_ == {'set_bus': {'loads_id': [(0, 2), (5, 1)],
                                     'generators_id': [(0, 2), (5, 1)],
                                     'lines_ex_id': [(0, 2), (5, 1), (15, 2)],
                                     'lines_or_id': [(0, 2), (5, 1), (15, 2)]
                                     }}
        act2 = self.env2.action_space(dict_)
        dict_2 = act2.as_serializable_dict()
        assert dict_ == dict_2
        
        act : BaseAction = self.env1.action_space({"change_bus": {'loads_id': [0, 5],
                                                                  'generators_id': [0, 5],
                                                                  'lines_ex_id': [0, 5, 15],
                                                                  'lines_or_id': [0, 5, 15]
                                                                  }})
        dict_ = act.as_serializable_dict()
        assert dict_ == {'change_bus': {'loads_id': [0, 5],
                                        'generators_id': [0, 5],
                                        'lines_ex_id': [0, 5, 15],
                                        'lines_or_id': [0, 5, 15]
                                        }}
        act2 = self.env2.action_space(dict_)
        dict_2 = act2.as_serializable_dict()
        assert dict_ == dict_2
        
        
if __name__ == "__main__":
    unittest.main()
