# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import os
import json
import unittest
from grid2op.Action import BaseAction, PlayableAction
from grid2op.tests.helper_path_test import *

from grid2op.MakeEnv import make
from grid2op.Parameters import Parameters
from grid2op.Converter import ConnectivityConverter, IdToAct
import tempfile
import pdb

import warnings


class TestConnectivityConverter(HelperTests, unittest.TestCase):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        param = Parameters()
        param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make(
                "educ_case14_storage",
                test=True,
                param=param,
                action_class=PlayableAction,
            )
        np.random.seed(0)

    def tearDown(self):
        self.env.close()

    def test_ConnectivityConverter(self):
        converter = ConnectivityConverter(self.env.action_space)
        converter.seed(0)
        converter.init_converter()

        res = np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                12,
                12,
                12,
                12,
                12,
                12,
            ]
        )

        assert np.array_equal(converter.subs_ids, res)
        assert len(converter.obj_type) == converter.n
        assert len(set(converter.obj_type)) == converter.n
        assert converter.pos_topo.shape[0] == converter.n
        assert len(set([tuple(sorted(el)) for el in converter.pos_topo])) == converter.n

        n_trial = 100
        preds = np.zeros(n_trial)
        for i in range(n_trial):
            coded_act = np.random.rand(converter.n) * 2.0 - 1.0
            preds[i] = converter._compute_disagreement(coded_act, np.ones(converter.n))
            # check the formula is properly implemented in case of "everything connected together"
            assert np.abs(preds[i] - 0.5 * (1 - np.mean(coded_act))) <= self.tol_one

        # check that on average over "a lot" of stuff the distance to the "everything connected" is 0.5
        assert np.abs(np.mean(preds) - 0.5) <= 1 / np.sqrt(n_trial)

        # and not test i can produce an action that can be implemented
        act = converter.convert_act(encoded_act=coded_act)
        obs, reward, done, info = self.env.step(act)

        # test sample
        obs = self.env.reset()
        act = converter.sample()
        obs, reward, done, info = self.env.step(act)

    def test_max_sub_changed(self):
        for ms_sub in [1, 2, 3]:
            converter = ConnectivityConverter(self.env.action_space)
            converter.init_converter(max_sub_changed=ms_sub)
            converter.seed(0)

            coded_act = np.random.rand(converter.n)

            # and not test i can produce an action that can be implemented
            act = converter.convert_act(encoded_act=coded_act)
            lines_impacted, subs_impacted = act.get_topological_impact()
            assert (
                np.sum(subs_impacted) == ms_sub
            ), "wrong number of substations affected. It should be {}".format(ms_sub)
            obs, reward, done, info = self.env.step(act)

            # test sample
            obs = self.env.reset()
            act = converter.sample()
            lines_impacted, subs_impacted = act.get_topological_impact()
            assert (
                np.sum(subs_impacted) == ms_sub
            ), "wrong number of substations affected. It should be {}".format(ms_sub)
            obs, reward, done, info = self.env.step(act)

    def test_can_make_action(self):
        converter = ConnectivityConverter(self.env.action_space)
        converter.init_converter(max_sub_changed=self.env.parameters.MAX_SUB_CHANGED)
        converter.seed(0)
        with self.assertRaises(Exception):
            # one too shot
            tmp = np.zeros(converter.n - 1)
            converter.convert_act(tmp)
        with self.assertRaises(Exception):
            # one too long
            tmp = np.zeros(converter.n + 1)
            converter.convert_act(tmp)
        with self.assertRaises(Exception):
            # one too low
            tmp = np.zeros(converter.n)
            tmp[3] = -1.1
            converter.convert_act(tmp)
        with self.assertRaises(Exception):
            # one too high
            tmp = np.zeros(converter.n)
            tmp[3] = 1.1
            converter.convert_act(tmp)

        size_ = converter.n
        # test do nothing gives do nothing indeed
        dn_enc = converter.do_nothing_encoded_act()
        glop_dn_act = converter.convert_act(dn_enc)
        assert glop_dn_act == self.env.action_space()

        # encode the topology [1,1 ,2,2] at sub 12, meaning: line_ex 9 / line_ex 13 together,
        # and line or 14 / load 9 together
        complex_act = 1.0 * dn_enc
        complex_act[84] = 1.0  # line ex 9 and line ex 13 together
        complex_act[85] = -1.0  # line ex 9 and line or 14 not together
        complex_act[86] = -1.0  # line ex 9 and load 9 not together

        glop_act = converter.convert_act(complex_act)
        aff_line, aff_sub = glop_act.get_topological_impact()
        assert np.sum(aff_line) == 0
        assert np.sum(aff_sub) == 1
        assert aff_sub[12]
        assert glop_act.line_ex_set_bus[9] != 0
        assert glop_act.line_ex_set_bus[13] != 0
        assert glop_act.line_or_set_bus[14] != 0
        assert glop_act.load_set_bus[9] != 0
        assert glop_act.line_ex_set_bus[9] == glop_act.line_ex_set_bus[13]
        assert glop_act.line_ex_set_bus[9] != glop_act.line_or_set_bus[14]
        assert glop_act.line_ex_set_bus[9] != glop_act.load_set_bus[9]
        assert glop_act.line_or_set_bus[14] == glop_act.load_set_bus[9]

        # encode the topology [1,1 ,2,2] at sub 12, meaning: line_ex 9 / line_ex 13 together,
        # and line or 14 / load 9 together but in a "soft" manner
        complex_act = 1.0 * dn_enc
        complex_act[84] = 0.8  # line ex 9 and line ex 13 together
        complex_act[85] = -0.9  # line ex 9 and line or 14 not together
        complex_act[86] = -0.9  # line ex 9 and load 9 not together
        glop_act2 = converter.convert_act(complex_act)
        assert (
            abs(converter.last_disagreement - 0.5 * (0.2 + 0.1 + 0.1) / size_)
            <= self.tol_one
        )
        assert glop_act == glop_act2

        # now tricky stuff, such that the greedy do not work and i need to explore a bit
        # line_ex 9 / line_ex 13 together and line or 14 / load 9 together
        complex_act = 1.0 * dn_enc
        complex_act[84] = 0.6  # line ex 9 and line ex 13 together
        complex_act[85] = -0.9  # line ex 9 and line or 14 not together
        complex_act[86] = -0.9  # line ex 9 and load 9 not together
        complex_act[87] = 0.61  # "line_ex id 13" and the "line_or id 14" together
        complex_act[88] = -0.2  # "line_ex id 13" and the "load id 9" together
        complex_act[89] = 0.0  # "line_or id 14" and the "load id 9" no preferences
        glop_act3 = converter.convert_act(complex_act)
        # this gives : [ 1     2     2      2   ], which is sub optimal
        #               Lex9 Lex13  lor14  loa9
        frst_disag = converter.last_disagreement
        assert (
            abs(frst_disag - 0.5 * (1.6 + 0.1 + 0.1 + 0.39 + 1.2) / size_)
            <= self.tol_one
        )
        assert glop_act != glop_act3  # but glop_act should be better !

        glop_act4 = converter.convert_act(complex_act, explore=3)
        this_disag = converter.last_disagreement
        assert this_disag < frst_disag, "this disagreement should always be lower !"
        assert converter.indx_sel != 0, "the first has been selected, it should not !"
        assert glop_act == glop_act4, "the same action first action should be optimal"
        assert (
            abs(this_disag - 0.5 * (0.4 + 0.1 + 0.1 + 1.61 + 0.8) / size_)
            <= self.tol_one
        )

    def test_bug_in_doc(self):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("rte_case14_realistic", test=True)
        converter = ConnectivityConverter(env.action_space)
        # it's a good practice to seed the element that can be, for reproducibility
        converter.seed(0)
        # to avoid creating illegal actions affecting more than the allowed number of parameters
        converter.init_converter(max_sub_changed=env.parameters.MAX_SUB_CHANGED)

        encoded_act = np.zeros(converter.n)
        encoded_act[0] = 1  # i want to connect  "line_ex id 0" and the "line_or id 2"
        encoded_act[
            1
        ] = -1  # i don't want to connect "line_ex id 0" and the "line_or id 3"
        encoded_act[
            2
        ] = -1  # i don't want to connect "line_ex id 0" and the "line_or id 4"
        encoded_act[3] = -1  # i don't want to connect "line_ex id 0" and the "gen id 0"
        encoded_act[4] = 1  # i want to connect "line_ex id 0" and the "load id 0"
        # and now retrieve the corresponding grid2op action:
        grid2op_act = converter.convert_act(encoded_act)
        assert converter.last_disagreement == 0.0
        assert np.array_equal(grid2op_act.set_bus[3:9], [1, 1, 2, 2, 2, 1])

        # second way to express the same action
        encoded_act2 = np.zeros(converter.n)
        encoded_act2[0] = 1  # i want to connect  "line_ex id 0" and the "line_or id 2"
        encoded_act2[4] = 1  # i want to connect "line_ex id 0" and the "load id 0"

        encoded_act2[9] = 1  # i want to connect "line_or id 3" and the "line_or id 4"
        encoded_act2[10] = 1  # i want to connect "line_or id 3" and the "gen id 0"

        encoded_act2[14] = -1  # i don't want to connect "gen id 0" and the "load id 0"

        # and now retrieve the corresponding grid2op action:
        grid2op_act2 = converter.convert_act(encoded_act2)
        assert converter.last_disagreement == 0.0
        assert np.array_equal(grid2op_act2.set_bus[3:9], [1, 1, 2, 2, 2, 1])

        # trick it: i don't specified enough constraints (used to be infinite loop)
        encoded_act3 = np.zeros(converter.n)
        encoded_act3[0] = 1  # i want to connect  "line_ex id 0" and the "line_or id 2"
        encoded_act3[4] = 1  # i want to connect "line_ex id 0" and the "load id 0"

        encoded_act3[9] = 1  # i want to connect "line_or id 3" and the "line_or id 4"
        encoded_act3[10] = 1  # i want to connect "line_or id 3" and the "gen id 0"
        grid2op_act3 = converter.convert_act(encoded_act3)
        assert converter.last_disagreement == 0.0
        assert np.array_equal(grid2op_act3.set_bus[3:9], [1, 1, 1, 1, 1, 1])

        size_ = converter.n

        # trick (the compute_disagreement function) in another way: not all components are set
        missing = np.ones(converter.n)
        missing[3] = 0  # encodes for line_ex id O
        disag0 = converter._compute_disagreement(encoded_act3, missing)
        # 2 constraints not met because the line_ex id O is not set in the action
        assert abs(disag0 - 2.0 / size_) <= self.tol_one

        missing2 = np.ones(converter.n)
        missing2[8] = 0  # encodes for load id 0
        disag2 = converter._compute_disagreement(encoded_act3, missing2)
        # 1 constraints not met because the load id 0 O is not set in the action
        assert abs(disag2 - 1.0 / size_) <= self.tol_one

        missing3 = np.ones(converter.n)
        missing3[3:9] = 0  # all constraints not met
        disag3 = converter._compute_disagreement(encoded_act3, missing3)
        # one component not set in the "action candidate" among 4 constraints
        assert abs(disag3 - 4.0 / size_) <= self.tol_one


class TestIdToAct(HelperTests, unittest.TestCase):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        param = Parameters()
        param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make(
                "educ_case14_storage",
                test=True,
                param=param,
                action_class=PlayableAction,
            )
        np.random.seed(0)
        self.filenamedict = "test_action_json_educ_case14_storage.json"

    def tearDown(self):
        self.env.close()

    def test_save_reload(self):
        path_ = tempfile.mkdtemp()
        converter = IdToAct(self.env.action_space)
        converter.init_converter(set_line_status=False, change_bus_vect=False)
        converter.save(path_, "tmp_convert.npy")
        init_size = converter.size()
        array = np.load(os.path.join(path_, "tmp_convert.npy"))
        act = converter.convert_act(27)
        act_ = converter.convert_act(-1)
        assert array.shape[1] == self.env.action_space.size()
        converter2 = IdToAct(self.env.action_space)
        converter2.init_converter(all_actions=os.path.join(path_, "tmp_convert.npy"))
        assert init_size == converter2.size()
        act2 = converter2.convert_act(27)
        act2_ = converter2.convert_act(-1)
        assert act == act2
        assert act_ == act2_

    def test_specific_attr(self):
        dict_orig = {
            "set_line_status": False,
            "change_line_status": False,
            "set_topo_vect": False,
            "change_bus_vect": False,
            "redispatch": False,
            "curtail": False,
            "storage": False,
        }

        dims = {
            "set_line_status": 101,
            "change_line_status": 21,
            "set_topo_vect": 235,
            "change_bus_vect": 255,
            "redispatch": 25,
            "curtail": 31,
            "storage": 17,
        }

        for attr in dict_orig.keys():
            kwargs = dict_orig.copy()
            kwargs[attr] = True
            converter = IdToAct(self.env.action_space)
            converter.init_converter(**kwargs)
            assert converter.n == dims[attr], (
                f'dim for "{attr}" should be {dims[attr]} but is ' f"{converter.n}"
            )

    def test_init_from_list_of_dict(self):
        path_input = os.path.join(PATH_DATA_TEST, self.filenamedict)
        with open(path_input, "r") as f:
            list_act = json.load(f)
        converter = IdToAct(self.env.action_space)
        converter.init_converter(all_actions=list_act)
        assert converter.n == 255
        assert isinstance(converter.all_actions[-1], BaseAction)
        assert isinstance(converter.all_actions[0], BaseAction)


if __name__ == "__main__":
    unittest.main()
