# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import sys
import unittest
import numpy as np
import copy
import pdb
import warnings

from grid2op.tests.helper_path_test import HelperTests, PATH_DATA_TEST_PP

from grid2op.Action import ActionSpace, CompleteAction
from grid2op.Backend import PandaPowerBackend
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChronicsHandler, ChangeNothing
from grid2op.Environment import Environment
from grid2op.Exceptions import *
from grid2op.Rules import RulesChecker
from grid2op.MakeEnv import make
from grid2op.Rules import AlwaysLegal
from grid2op.Space import GridObjects

PATH_DATA_TEST = PATH_DATA_TEST_PP
import pandapower as pppp


class TestLoadingCase(unittest.TestCase):
    def setUp(self):
        self.tolvect = 1e-2
        self.tol_one = 1e-5

    def test_load_file(self):
        backend = PandaPowerBackend()
        path_matpower = PATH_DATA_TEST
        case_file = "test_case14.json"
        backend.load_grid(path_matpower, case_file)

        assert backend.n_line == 20
        assert backend.n_gen == 5
        assert backend.n_load == 11
        assert backend.n_sub == 14

        name_line = ['0_1_0', '0_4_1',  '8_9_2', '8_13_3', '9_10_4', '11_12_5', '12_13_6', '1_2_7',
                      '1_3_8', '1_4_9', '2_3_10', '3_4_11', '5_10_12', '5_11_13', '5_12_14',  '3_6_15',
                      '3_8_16', '4_5_17', '6_7_18', '6_8_19']
        name_line = np.array(name_line)
        assert np.all(sorted(backend.name_line) == sorted(name_line))

        name_sub = ['sub_0', 'sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6', 'sub_7', 'sub_8', 'sub_9', 'sub_10',
                     'sub_11', 'sub_12', 'sub_13']
        name_sub = np.array(name_sub)
        assert np.all(sorted(backend.name_sub) == sorted(name_sub))

        name_gen = ['gen_0_4', 'gen_1_0', 'gen_2_1', 'gen_5_2', 'gen_7_3']
        name_gen = np.array(name_gen)
        assert np.all(sorted(backend.name_gen) == sorted(name_gen))

        name_load = ['load_1_0', 'load_2_1', 'load_13_2', 'load_3_3', 'load_4_4', 'load_5_5', 'load_8_6',
                      'load_9_7', 'load_10_8', 'load_11_9', 'load_12_10']
        name_load = np.array(name_load)
        assert np.all(sorted(backend.name_load) == sorted(name_load))

        assert np.all(backend.get_topo_vect() == np.ones(np.sum(backend.sub_info)))

        backend.runpf()
        try:
            p_subs, q_subs, p_bus, q_bus = backend.check_kirchoff()
            assert np.max(np.abs(p_subs)) <= self.tolvect
            assert np.max(np.abs(q_subs)) <= self.tolvect
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect

        except Grid2OpException:
            pass

    def test_assert_grid_correct(self):
        backend = PandaPowerBackend()
        path_matpower = PATH_DATA_TEST
        case_file = "test_case14.json"
        backend.load_grid(path_matpower, case_file)
        backend.assert_grid_correct()
        backend.runpf()
        backend.assert_grid_correct_after_powerflow()


class TestLoadingBackendFunc(unittest.TestCase):
    # Cette méthode sera appelée avant chaque test.
    def setUp(self):
        self.backend = PandaPowerBackend()
        self.path_matpower = PATH_DATA_TEST
        self.case_file = "test_case14.json"
        self.backend.load_grid(self.path_matpower, self.case_file)
        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.game_rules = RulesChecker()
        self.action_env = ActionSpace(gridobj=self.backend, legal_action=self.game_rules.legal_action)

    # Cette méthode sera appelée après chaque test.
    def tearDown(self):
        pass

    def compare_vect(self, pred, true):
        return np.max(np.abs(pred- true)) <= self.tolvect

    def test_runpf(self):
        conv = self.backend.runpf(is_dc=True)
        assert conv

        true_values_ac = np.array([ 1.56882891e+02,  7.55103818e+01,  5.22755247e+00,  9.42638103e+00,
                                   -3.78532238e+00,  1.61425777e+00,  5.64385098e+00,  7.32375792e+01,
                                    5.61314959e+01,  4.15162150e+01, -2.32856901e+01, -6.11582304e+01,
                                    7.35327698e+00,  7.78606702e+00,  1.77479769e+01,  2.80741759e+01,
                                    1.60797576e+01,  4.40873209e+01, -1.11022302e-14,  2.80741759e+01])
        true_values_dc = np.array([147.83859556,  71.16140444,   5.7716542 ,   9.64132512,
                                    -3.2283458 ,   1.50735814,   5.25867488,  70.01463596,
                                    55.1518527 ,  40.9721069 , -24.18536404, -61.74649065,
                                     6.7283458 ,   7.60735814,  17.25131674,  28.36115279,
                                    16.55182652,  42.78702069,   0.        ,  28.36115279])

        p_or, *_ = self.backend.lines_or_info()
        assert self.compare_vect(p_or, true_values_dc)

        conv = self.backend.runpf(is_dc=False)
        assert conv
        p_or, *_ = self.backend.lines_or_info()
        assert self.compare_vect(p_or, true_values_ac)

    def test_voltage_convert_powerlines(self):
        # i have the correct voltages in powerlines if the formula to link mw, mvar, kv and amps is correct
        conv = self.backend.runpf(is_dc=False)
        assert conv

        p_or, q_or, v_or, a_or = self.backend.lines_or_info()
        a_th = np.sqrt(p_or ** 2 + q_or ** 2) * 1e3 / (np.sqrt(3) * v_or)
        assert self.compare_vect(a_th, a_or)

        p_ex, q_ex, v_ex, a_ex = self.backend.lines_ex_info()
        a_th = np.sqrt(p_ex ** 2 + q_ex ** 2) * 1e3 / (np.sqrt(3) * v_ex)
        assert self.compare_vect(a_th, a_ex)

    def test_voltages_correct_load_gen(self):
        # i have the right voltages to generators and load, if it's the same as the voltage (correct from the above test)
        # of the powerline connected to it.

        conv = self.backend.runpf(is_dc=False)
        load_p, load_q, load_v = self.backend.loads_info()
        gen_p, gen__q, gen_v = self.backend.generators_info()
        p_or, q_or, v_or, a_or = self.backend.lines_or_info()
        p_ex, q_ex, v_ex, a_ex = self.backend.lines_ex_info()

        for c_id, sub_id in enumerate(self.backend.load_to_subid):
            l_id = np.where(self.backend.line_or_to_subid == sub_id)[0]
            if len(l_id):
                l_id = l_id[0]
                assert np.abs(v_or[l_id] - load_v[c_id]) <= self.tol_one, "problem for load {}".format(c_id)
                continue

            l_id = np.where(self.backend.line_ex_to_subid == sub_id)[0]
            if len(l_id):
                l_id = l_id[0]
                assert np.abs(v_ex[l_id] - load_v[c_id]) <= self.tol_one, "problem for load {}".format(c_id)
                continue
            assert False, "load {} has not been checked".format(c_id)

        for g_id, sub_id in enumerate(self.backend.gen_to_subid):
            l_id = np.where(self.backend.line_or_to_subid == sub_id)[0]
            if len(l_id):
                l_id = l_id[0]
                assert np.abs(v_or[l_id] - gen_v[g_id]) <= self.tol_one, "problem for generator {}".format(g_id)
                continue

            l_id = np.where(self.backend.line_ex_to_subid == sub_id)[0]
            if len(l_id):
                l_id = l_id[0]
                assert np.abs(v_ex[l_id] - gen_v[g_id]) <= self.tol_one, "problem for generator {}".format(g_id)
                continue
            assert False, "generator {} has not been checked".format(g_id)

    def test_copy(self):
        conv = self.backend.runpf(is_dc=False)
        p_or_orig, *_ = self.backend.lines_or_info()
        adn_backend_cpy = self.backend.copy()

        self.backend._disconnect_line(3)
        conv = self.backend.runpf(is_dc=False)
        assert conv
        conv2 = adn_backend_cpy.runpf(is_dc=False)
        assert conv2
        p_or_ref, *_ = self.backend.lines_or_info()
        p_or, *_ = adn_backend_cpy.lines_or_info()
        assert np.abs(p_or_ref[3]) <= self.tol_one
        assert self.compare_vect(p_or_orig, p_or)

    def test_get_line_status(self):
        assert np.all(self.backend.get_line_status())
        self.backend._disconnect_line(3)
        assert np.sum(~self.backend.get_line_status()) == 1
        assert not self.backend.get_line_status()[3]

    def test_get_line_flow(self):
        self.backend.runpf(is_dc=False)
        true_values_ac = np.array([-20.40429168,   3.85499114,   4.2191378 ,   3.61000624,
                                    -1.61506292,   0.75395917,   1.74717378,   3.56020295,
                                    -1.5503504 ,   1.17099786,   4.47311562,  15.82364194,
                                     3.56047297,   2.50341424,   7.21657539,  -9.68106571,
                                    -0.42761118,  12.47067981, -17.16297051,   5.77869057])
        p_or_orig, q_or_orig, *_ = self.backend.lines_or_info()
        assert self.compare_vect(q_or_orig, true_values_ac)

        self.backend._disconnect_line(3)
        a = self.backend.runpf(is_dc=False)
        true_values_ac = np.array([-20.40028207,   3.65600775,   3.77916284,   0.        ,
                                    -2.10761554,   1.34025308,   5.86505081,   3.58514625,
                                    -2.28717836,   0.81979017,   3.72328838,  17.09556423,
                                     3.9548798 ,   3.18389804,  11.24144925, -11.09660174,
                                    -1.70423701,  13.14347167, -14.82917601,   2.276297  ])
        p_or_orig, q_or_orig, *_ = self.backend.lines_or_info()
        assert self.compare_vect(q_or_orig, true_values_ac)

    def test_pf_ac_dc(self):
        true_values_ac = np.array([-20.40429168,   3.85499114,   4.2191378 ,   3.61000624,
                                    -1.61506292,   0.75395917,   1.74717378,   3.56020295,
                                    -1.5503504 ,   1.17099786,   4.47311562,  15.82364194,
                                     3.56047297,   2.50341424,   7.21657539,  -9.68106571,
                                    -0.42761118,  12.47067981, -17.16297051,   5.77869057])
        conv = self.backend.runpf(is_dc=True)
        assert conv
        p_or_orig, q_or_orig, *_ = self.backend.lines_or_info()
        assert np.all(q_or_orig == 0.)
        conv = self.backend.runpf(is_dc=False)
        assert conv
        p_or_orig, q_or_orig, *_ = self.backend.lines_or_info()
        assert self.compare_vect(q_or_orig, true_values_ac)


    def test_get_thermal_limit(self):
        res = self.backend.get_thermal_limit()
        true_values_ac = np.array([   42339.01974057,    42339.01974057, 27479652.23546777,
                                   27479652.23546777, 27479652.23546777, 27479652.23546777,
                                   27479652.23546777,    42339.01974057,    42339.01974057,
                                      42339.01974057,    42339.01974057,    42339.01974057,
                                   27479652.23546777, 27479652.23546777, 27479652.23546777,
                                      42339.01974057,    42339.01974057,    42339.01974057,
                                     408269.11892695,   408269.11892695])
        assert self.compare_vect(res, true_values_ac)

    def test_disconnect_line(self):
        for i in range(self.backend.n_line):
            if i == 18:
                # powerflow diverge if line 1 is removed, unfortunately
                continue
            backend_cpy = self.backend.copy()
            backend_cpy._disconnect_line(i)
            conv = backend_cpy.runpf()
            assert conv, "Power flow computation does not converge if line {} is removed".format(i)
            flows = backend_cpy.get_line_status()
            assert not flows[i]
            assert np.sum(~flows) == 1

    def test_donothing_action(self):
        conv = self.backend.runpf()
        init_flow = self.backend.get_line_flow()
        init_lp, *_ = self.backend.loads_info()
        init_gp, *_ = self.backend.generators_info()
        init_ls = self.backend.get_line_status()

        action = self.action_env({})  # update the action
        self.backend.apply_action(action)
        after_lp, *_ = self.backend.loads_info()
        after_gp, *_ = self.backend.generators_info()
        after_ls = self.backend.get_line_status()
        assert self.compare_vect(init_lp, after_lp)  # check i didn't modify the loads
        # assert self.compare_vect(init_gp, after_gp)  # check i didn't modify the generators  # TODO here !!! problem with steady state P=C+L
        assert np.all(init_ls == after_ls)  # check i didn't disconnect any powerlines

        conv = self.backend.runpf()
        assert conv, "Cannot perform a powerflow after doing nothing"
        after_flow = self.backend.get_line_flow()
        assert self.compare_vect(init_flow, after_flow)

    def test_apply_action_active_value(self):
        # test that i can modify only the load / prod active values of the powergrid
        # to do that i modify the productions and load all of a factor 0.5 and compare that the DC flows are
        # also multiply by 2
        conv = self.backend.runpf(is_dc=True)
        init_flow, *_ = self.backend.lines_or_info()
        init_lp, init_l_q, *_ = self.backend.loads_info()
        init_gp, *_ = self.backend.generators_info()
        init_ls = self.backend.get_line_status()

        ratio = 0.5
        action = self.action_env({"injection": {"load_p": ratio*init_lp,
                                                "prod_p": ratio*init_gp*np.sum(init_lp)/np.sum(init_gp)}})  # update the action

        self.backend.apply_action(action)
        conv = self.backend.runpf(is_dc=True)
        assert conv, "Cannot perform a powerflow after doing nothing"

        after_lp, after_lq, *_ = self.backend.loads_info()
        after_gp, *_ = self.backend.generators_info()
        after_ls = self.backend.get_line_status()
        assert self.compare_vect(ratio*init_lp, after_lp)  # check i didn't modify the loads
        try:
            p_subs, q_subs, p_bus, q_bus = self.backend.check_kirchoff()
            # i'm in DC mode, i can't check for reactive values...
            assert np.max(np.abs(p_subs)) <= self.tolvect, "problem with active values, at substation"
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect, "problem with active values, at a bus"
        except Grid2OpException:
            pass

        assert self.compare_vect(ratio*init_gp, after_gp)  # check i didn't modify the generators
        assert np.all(init_ls == after_ls)  # check i didn't disconnect any powerlines

        after_flow, *_ = self.backend.lines_or_info()
        assert self.compare_vect(ratio*init_flow, after_flow) # probably an error with the DC approx

    def test_apply_action_prod_v(self):
        conv = self.backend.runpf(is_dc=False)
        prod_p_init, prod_q_init, prod_v_init = self.backend.generators_info()
        ratio = 1.05
        action = self.action_env({"injection": {"prod_v": ratio*prod_v_init}})  # update the action
        self.backend.apply_action(action)
        conv = self.backend.runpf(is_dc=False)
        assert conv, "Cannot perform a powerflow aftermodifying the powergrid"

        prod_p_after, prod_q_after, prod_v_after = self.backend.generators_info()
        assert self.compare_vect(ratio*prod_v_init, prod_v_after)  # check i didn't modify the generators
        
    def test_apply_action_maintenance(self):
        # retrieve some initial data to be sure only a subpart of the _grid is modified
        conv = self.backend.runpf()
        init_lp, *_ = self.backend.loads_info()
        init_gp, *_ = self.backend.generators_info()

        # check that maintenance vector is properly taken into account
        maintenance = np.full((self.backend.n_line,), fill_value=False, dtype=np.bool)
        maintenance[19] = True
        action = self.action_env({"maintenance": maintenance})  # update the action

        # apply the action here
        self.backend.apply_action(action)

        # compute a load flow an performs more tests
        conv = self.backend.runpf()
        assert conv, "Power does not converge if line {} is removed".format(19)

        # performs basic check
        after_lp, *_ = self.backend.loads_info()
        after_gp, *_ = self.backend.generators_info()
        after_ls = self.backend.get_line_status()
        assert self.compare_vect(init_lp, after_lp)  # check i didn't modify the loads
        # assert self.compare_vect(init_gp, after_gp)  # check i didn't modify the generators  # TODO here problem with steady state P=C+L
        assert np.all(~maintenance == after_ls)  # check i didn't disconnect any powerlines beside the correct one

        flows = self.backend.get_line_status()
        assert np.sum(~flows) == 1
        assert not flows[19]

    def test_apply_action_hazard(self):
        conv = self.backend.runpf()
        init_lp, *_ = self.backend.loads_info()
        init_gp, *_ = self.backend.generators_info()

        # check that maintenance vector is properly taken into account
        maintenance = np.full((self.backend.n_line,), fill_value=False, dtype=np.bool)
        maintenance[17] = True
        action = self.action_env({"hazards": maintenance})  # update the action

        # apply the action here
        self.backend.apply_action(action)

        # compute a load flow an performs more tests
        conv = self.backend.runpf()
        assert conv, "Power does not converge if line {} is removed".format(19)

        # performs basic check
        after_lp, *_ = self.backend.loads_info()
        after_gp, *_ = self.backend.generators_info()
        after_ls = self.backend.get_line_status()
        assert self.compare_vect(init_lp, after_lp)  # check i didn't modify the loads
        # assert self.compare_vect(init_gp, after_gp)  # check i didn't modify the generators  # TODO here problem with steady state P=C+L
        assert np.all(maintenance == ~after_ls)  # check i didn't disconnect any powerlines beside the correct one

    #
    def test_apply_action_disconnection(self):
        # retrieve some initial data to be sure only a subpart of the _grid is modified
        conv = self.backend.runpf()
        init_lp, *_ = self.backend.loads_info()
        init_gp, *_ = self.backend.generators_info()

        # check that maintenance vector is properly taken into account
        maintenance = np.full((self.backend.n_line,), fill_value=False, dtype=np.bool)
        maintenance[19] = True

        disc = np.full((self.backend.n_line,), fill_value=False, dtype=np.bool)
        disc[17] = True

        action = self.action_env({"hazards": disc, "maintenance": maintenance})  # update the action
        # apply the action here
        self.backend.apply_action(action)

        # compute a load flow an performs more tests
        conv = self.backend.runpf()
        assert conv, "Power does not converge if lines {} and {} are removed".format(17, 19)

        # performs basic check
        after_lp, *_ = self.backend.loads_info()
        after_gp, *_ = self.backend.generators_info()
        after_ls = self.backend.get_line_status()
        assert self.compare_vect(init_lp, after_lp)  # check i didn't modify the loads
        # assert self.compare_vect(init_gp, after_gp)  # check i didn't modify the generators # TODO here problem with steady state, P=C+L
        assert np.all(disc | maintenance == ~after_ls)  # check i didn't disconnect any powerlines beside the correct one

        flows = self.backend.get_line_status()
        assert np.sum(~flows) == 2
        assert not flows[19]
        assert not flows[17]


class TestTopoAction(unittest.TestCase):
    # Cette méthode sera appelée avant chaque test.
    def setUp(self):
        self.backend = PandaPowerBackend()
        self.path_matpower = PATH_DATA_TEST
        self.case_file = "test_case14.json"
        self.backend.load_grid(self.path_matpower, self.case_file)
        self.tolvect = 1e-2
        self.tol_one = 1e-5

        self.game_rules = RulesChecker()
        self.helper_action = ActionSpace(gridobj=self.backend, legal_action=self.game_rules.legal_action)

    # Cette méthode sera appelée après chaque test.
    def tearDown(self):
        pass

    def compare_vect(self, pred, true):
        return np.max(np.abs(pred- true)) <= self.tolvect

    def test_get_topo_vect_speed(self):
        # retrieve some initial data to be sure only a subpart of the _grid is modified
        conv = self.backend.runpf()
        init_amps_flow = self.backend.get_line_flow()

        # check that maintenance vector is properly taken into account
        arr = np.array([1, 1, 1, 2, 2, 2], dtype=np.int)
        id_=1
        action = self.helper_action({"set_bus": {"substations_id": [(id_, arr)]}})

        # apply the action here
        self.backend.apply_action(action)
        conv = self.backend.runpf()
        assert conv
        after_amps_flow = self.backend.get_line_flow()

        topo_vect = self.backend.get_topo_vect()
        topo_vect_old = self.backend._get_topo_vect_old()
        assert self.compare_vect(topo_vect, topo_vect_old) == True

    def test_topo_set1sub(self):
        # retrieve some initial data to be sure only a subpart of the _grid is modified
        conv = self.backend.runpf()
        init_amps_flow = self.backend.get_line_flow()

        # check that maintenance vector is properly taken into account
        arr = np.array([1, 1, 1, 2, 2, 2], dtype=np.int)
        id_=1
        action = self.helper_action({"set_bus": {"substations_id": [(id_, arr)]}})

        # apply the action here
        self.backend.apply_action(action)
        conv = self.backend.runpf()
        assert conv
        after_amps_flow = self.backend.get_line_flow()

        topo_vect = self.backend.get_topo_vect()
        assert np.min(topo_vect) == 1, "all buses have been changed"
        assert np.max(topo_vect) == 2, "no buses have been changed"

        # check that the objects have been properly moved
        load_ids = np.where(self.backend.load_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.load_pos_topo_vect[load_ids]] == arr[self.backend.load_to_sub_pos[load_ids]])
        lor_ids = np.where(self.backend.line_or_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.line_or_pos_topo_vect[lor_ids]] == arr[self.backend.line_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.line_ex_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.line_ex_pos_topo_vect[lex_ids]] == arr[self.backend.line_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == arr[self.backend.gen_to_sub_pos[gen_ids]])

        after_amps_flow_th = np.array([6.38865247e+02, 3.81726828e+02, 1.78001287e+04, 2.70742428e+04,
                                       1.06755055e+04, 4.71160165e+03, 1.52265925e+04, 3.37755751e+02,
                                       3.00535519e+02, 5.01164454e-13, 7.01900962e+01, 1.73874580e+02,
                                       2.08904697e+04, 2.11757439e+04, 4.93863382e+04, 1.31935835e+02,
                                       6.99779475e+01, 1.85068609e+02, 7.47283039e+02, 1.14125596e+03])

        after_amps_flow_th = np.array([  596.58386539,   342.31364678, 18142.87789987, 27084.37162086,
                                       10155.86483194,  4625.93022957, 15064.92626615,   322.59381855,
                                         273.6977149 ,    82.21908229,    80.91290202,   206.04740125,
                                       20480.81970337, 21126.22533095, 49275.71520428,   128.04429617,
                                          69.00661266,   188.44754187,   688.1371226 ,  1132.42521887])
        assert self.compare_vect(after_amps_flow, after_amps_flow_th)

        try:
            p_subs, q_subs, p_bus, q_bus = self.backend.check_kirchoff()
            assert np.max(np.abs(p_subs)) <= self.tolvect, "problem with active values, at substation"
            assert np.max(np.abs(q_subs)) <= self.tolvect, "problem with reactive values, at substation"
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect, "problem with active values, at a bus"
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect, "problem with reaactive values, at a load"

        except Grid2OpException:
            pass

    def test_topo_change1sub(self):
        # check that switching the bus of 3 object is equivalent to set them to bus 2 (as above)
        conv = self.backend.runpf()
        init_amps_flow = self.backend.get_line_flow()

        # check that maintenance vector is properly taken into account
        arr = np.array([False, False, False, True, True, True], dtype=np.bool)
        id_ = 1
        action = self.helper_action({"change_bus": {"substations_id": [(id_, arr)]}})

        # apply the action here
        self.backend.apply_action(action)

        # run the powerflow
        conv = self.backend.runpf()
        assert conv
        after_amps_flow = self.backend.get_line_flow()

        topo_vect = self.backend.get_topo_vect()
        assert np.min(topo_vect) == 1, "all buses have been changed"
        assert np.max(topo_vect) == 2, "no buses have been changed"

        # check that the objects have been properly moved
        load_ids = np.where(self.backend.load_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.load_pos_topo_vect[load_ids]] == 1+arr[self.backend.load_to_sub_pos[load_ids]])
        lor_ids = np.where(self.backend.line_or_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.line_or_pos_topo_vect[lor_ids]] == 1+arr[self.backend.line_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.line_ex_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.line_ex_pos_topo_vect[lex_ids]] == 1+arr[self.backend.line_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == 1+arr[self.backend.gen_to_sub_pos[gen_ids]])

        after_amps_flow_th = np.array([  596.58386539,   342.31364678, 18142.87789987, 27084.37162086,
                                       10155.86483194,  4625.93022957, 15064.92626615,   322.59381855,
                                         273.6977149 ,    82.21908229,    80.91290202,   206.04740125,
                                       20480.81970337, 21126.22533095, 49275.71520428,   128.04429617,
                                          69.00661266,   188.44754187,   688.1371226 ,  1132.42521887])
        assert self.compare_vect(after_amps_flow, after_amps_flow_th)

        try:
            p_subs, q_subs, p_bus, q_bus = self.backend.check_kirchoff()
            assert np.max(np.abs(p_subs)) <= self.tolvect, "problem with active values, at substation"
            assert np.max(np.abs(q_subs)) <= self.tolvect, "problem with reactive values, at substation"
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect, "problem with active values, at a bus"
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect, "problem with reaactive values, at a load"

        except Grid2OpException:
            pass

    def test_topo_change_1sub_twice(self):
        # check that switching the bus of 3 object is equivalent to set them to bus 2 (as above)
        # and that setting it again is equivalent to doing nothing
        conv = self.backend.runpf()
        init_amps_flow = np.array([el for el in self.backend.get_line_flow()])

        # check that maintenance vector is properly taken into account
        arr = np.array([False, False, False, True, True, True], dtype=np.bool)
        id_ = 1
        action = self.helper_action({"change_bus": {"substations_id": [(id_, arr)]}})

        # apply the action here
        self.backend.apply_action(action)
        conv = self.backend.runpf()
        assert conv
        after_amps_flow = self.backend.get_line_flow()

        topo_vect = self.backend.get_topo_vect()
        assert np.min(topo_vect) == 1, "all buses have been changed"
        assert np.max(topo_vect) == 2, "no buses have been changed"

        # check that the objects have been properly moved
        load_ids = np.where(self.backend.load_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.load_pos_topo_vect[load_ids]] == 1+arr[self.backend.load_to_sub_pos[load_ids]])
        lor_ids = np.where(self.backend.line_or_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.line_or_pos_topo_vect[lor_ids]] == 1+arr[self.backend.line_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.line_ex_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.line_ex_pos_topo_vect[lex_ids]] == 1+arr[self.backend.line_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid == id_)[0]
        assert np.all(topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == 1+arr[self.backend.gen_to_sub_pos[gen_ids]])

        after_amps_flow_th = np.array([  596.58386539,   342.31364678, 18142.87789987, 27084.37162086,
                                       10155.86483194,  4625.93022957, 15064.92626615,   322.59381855,
                                         273.6977149 ,    82.21908229,    80.91290202,   206.04740125,
                                       20480.81970337, 21126.22533095, 49275.71520428,   128.04429617,
                                          69.00661266,   188.44754187,   688.1371226 ,  1132.42521887])
        assert self.compare_vect(after_amps_flow, after_amps_flow_th)
        try:
            p_subs, q_subs, p_bus, q_bus = self.backend.check_kirchoff()
            assert np.max(np.abs(p_subs)) <= self.tolvect, "problem with active values, at substation"
            assert np.max(np.abs(q_subs)) <= self.tolvect, "problem with reactive values, at substation"
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect, "problem with active values, at a bus"
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect, "problem with reaactive values, at a load"

        except Grid2OpException:
            pass

        action = self.helper_action({"change_bus": {"substations_id": [(id_, arr)]}})

        # apply the action here
        self.backend.apply_action(action)
        conv = self.backend.runpf()
        assert conv

        after_amps_flow = self.backend.get_line_flow()
        assert self.compare_vect(after_amps_flow, init_amps_flow)
        topo_vect = self.backend.get_topo_vect()
        assert np.min(topo_vect) == 1
        assert np.max(topo_vect) == 1
        try:
            p_subs, q_subs, p_bus, q_bus = self.backend.check_kirchoff()
            assert np.max(np.abs(p_subs)) <= self.tolvect, "problem with active values, at substation"
            assert np.max(np.abs(q_subs)) <= self.tolvect, "problem with reactive values, at substation"
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect, "problem with active values, at a bus"
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect, "problem with reaactive values, at a load"

        except Grid2OpException:
            pass


    def test_topo_change_2sub(self):
        # check that maintenance vector is properly taken into account
        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        id_1 = 1
        id_2 = 12
        action = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                     "set_bus": {"substations_id": [(id_2, arr2)]}})
        # apply the action here
        self.backend.apply_action(action)
        conv = self.backend.runpf()
        assert conv

        # check the _grid is correct
        topo_vect = self.backend.get_topo_vect()
        assert np.min(topo_vect) == 1, "all buses have been changed"
        assert np.max(topo_vect) == 2, "no buses have been changed"

        # check that the objects have been properly moved
        load_ids = np.where(self.backend.load_to_subid == id_1)[0]
        assert np.all(
            topo_vect[self.backend.load_pos_topo_vect[load_ids]] == 1+arr1[self.backend.load_to_sub_pos[load_ids]])
        lor_ids = np.where(self.backend.line_or_to_subid == id_1)[0]
        assert np.all(
            topo_vect[self.backend.line_or_pos_topo_vect[lor_ids]] == 1+arr1[self.backend.line_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.line_ex_to_subid == id_1)[0]
        assert np.all(
            topo_vect[self.backend.line_ex_pos_topo_vect[lex_ids]] == 1+arr1[self.backend.line_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid == id_1)[0]
        assert np.all(topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == 1+arr1[self.backend.gen_to_sub_pos[gen_ids]])


        load_ids = np.where(self.backend.load_to_subid == id_2)[0]
        assert np.all(
            topo_vect[self.backend.load_pos_topo_vect[load_ids]] == arr2[self.backend.load_to_sub_pos[load_ids]])
        lor_ids = np.where(self.backend.line_or_to_subid == id_2)[0]
        assert np.all(
            topo_vect[self.backend.line_or_pos_topo_vect[lor_ids]] == arr2[self.backend.line_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.line_ex_to_subid == id_2)[0]
        assert np.all(
            topo_vect[self.backend.line_ex_pos_topo_vect[lex_ids]] == arr2[self.backend.line_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid == id_2)[0]
        assert np.all(topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == arr2[self.backend.gen_to_sub_pos[gen_ids]])

        after_amps_flow = self.backend.get_line_flow()
        after_amps_flow_th = np.array([  596.97014348,   342.10559579, 16615.11815357, 31328.50690716,
                                       11832.77202397, 11043.10650167, 11043.10650167,   322.79533908,
                                         273.86501458,    82.34066647,    80.89289074,   208.42396413,
                                       22178.16766548, 27690.51322075, 38684.31540646,   129.44842477,
                                          70.02629553,   185.67687123,   706.77680037,  1155.45754617])
        assert self.compare_vect(after_amps_flow, after_amps_flow_th)

        try:
            p_subs, q_subs, p_bus, q_bus = self.backend.check_kirchoff()
            assert np.max(np.abs(p_subs)) <= self.tolvect, "problem with active values, at substation"
            assert np.max(np.abs(q_subs)) <= self.tolvect, "problem with reactive values, at substation"
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect, "problem with active values, at a bus"
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect, "problem with reaactive values, at a load"

        except Grid2OpException:
            pass


class TestEnvPerformsCorrectCascadingFailures(unittest.TestCase):
    """
    Test the "next_grid_state" method of the back-end
    """
    def setUp(self):
        self.backend = PandaPowerBackend(detailed_infos_for_cascading_failures=True)
        self.path_matpower = PATH_DATA_TEST
        self.case_file = "test_case14.json"
        self.backend.load_grid(self.path_matpower, self.case_file)
        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.game_rules = RulesChecker()
        self.action_env = ActionSpace(gridobj=self.backend, legal_action=self.game_rules.legal_action)

        self.lines_flows_init = np.array([  638.28966637,   305.05042301, 17658.9674809 , 26534.04334098,
                                           10869.23856329,  4686.71726729, 15612.65903298,   300.07915572,
                                             229.8060832 ,   169.97292682,   100.40192958,   265.47505664,
                                           21193.86923911, 21216.44452327, 49701.1565287 ,   124.79684388,
                                              67.59759985,   192.19424706,   666.76961936,  1113.52773632])
        # _parameters for the environment
        self.env_params = Parameters()

        # used for init an env too
        self.chronics_handler = ChronicsHandler()
        self.id_first_line_disco = 8  # due to hard overflow
        self.id_2nd_line_disco = 11  # due to soft overflow

    def next_grid_state_no_overflow(self):
        # first i test that, when there is no overflow, i dont do a cascading failure
        env = Environment(init_grid_path=os.path.join(self.path_matpower, self.case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=self.env_params)

        disco, infos = self.backend.next_grid_state(env, is_dc=False)
        assert not infos

    def test_next_grid_state_1overflow(self):
        # second i test that, when is one line on hard overflow it is disconnected
        case_file = self.case_file
        env_params = copy.deepcopy(self.env_params)
        env_params.HARD_OVERFLOW_THRESHOLD = 1.5
        env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=env_params)
        self.backend.load_grid(self.path_matpower, case_file)

        thermal_limit = 10*self.lines_flows_init
        thermal_limit[self.id_first_line_disco] = self.lines_flows_init[self.id_first_line_disco]/2
        self.backend.set_thermal_limit(thermal_limit)

        disco, infos = self.backend.next_grid_state(env, is_dc=False)
        assert len(infos) == 1  # check that i have only one overflow
        assert np.sum(disco) == 1

    def test_next_grid_state_1overflow_envNoCF(self):
        # third i test that, if a line is on hard overflow, but i'm on a "no cascading failure" mode,
        # i don't simulate a cascading failure
        self.env_params.NO_OVERFLOW_DISCONNECTION = True
        case_file = self.case_file
        env_params = copy.deepcopy(self.env_params)
        env_params.HARD_OVERFLOW_THRESHOLD = 1.5
        env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=self.env_params)
        self.backend.load_grid(self.path_matpower, case_file)

        thermal_limit = 10*self.lines_flows_init
        thermal_limit[self.id_first_line_disco] = self.lines_flows_init[self.id_first_line_disco]/2
        self.backend.set_thermal_limit(thermal_limit)

        disco, infos = self.backend.next_grid_state(env, is_dc=False)
        assert not infos  # check that don't simulate a cascading failure
        assert np.sum(disco) == 0

    def test_nb_timestep_overflow_disc0(self):
        # on this _grid, first line with id 5 is overheated,
        # it is disconnected
        # then powerline 16 have a relative flow of 1.5916318201096937
        # in this scenario i don't have a second line disconnection.
        case_file = self.case_file
        env_params = copy.deepcopy(self.env_params)
        env_params.HARD_OVERFLOW_THRESHOLD = 1.5
        env_params.NB_TIMESTEP_POWERFLOW_ALLOWED = 0
        env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=env_params)
        self.backend.load_grid(self.path_matpower, case_file)

        thermal_limit = 10*self.lines_flows_init
        thermal_limit[self.id_first_line_disco] = self.lines_flows_init[self.id_first_line_disco]/2
        thermal_limit[self.id_2nd_line_disco] = 400
        self.backend.set_thermal_limit(thermal_limit)

        disco, infos = self.backend.next_grid_state(env, is_dc=False)

        assert len(infos) == 2  # check that there is a cascading failure of length 2
        assert disco[self.id_first_line_disco]
        assert disco[self.id_2nd_line_disco]
        assert np.sum(disco) == 2

    def test_nb_timestep_overflow_nodisc(self):
        # on this _grid, first line with id 18 is overheated,
        # it is disconnected
        # then powerline 16 have a relative flow of 1.5916318201096937

        # in this scenario i don't have a second line disconnection because
        # the overflow is a soft overflow and  the powerline is presumably overflow since 0
        # timestep
        case_file = self.case_file
        env_params = copy.deepcopy(self.env_params)
        env_params.HARD_OVERFLOW_THRESHOLD = 1.5
        env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=env_params)
        self.backend.load_grid(self.path_matpower, case_file)

        env.timestep_overflow[self.id_2nd_line_disco] = 0
        thermal_limit = 10*self.lines_flows_init
        thermal_limit[self.id_first_line_disco] = self.lines_flows_init[self.id_first_line_disco]/2
        thermal_limit[self.id_2nd_line_disco] = 400
        self.backend.set_thermal_limit(thermal_limit)

        disco, infos = self.backend.next_grid_state(env, is_dc=False)
        assert len(infos) == 1  # check that don't simulate a cascading failure
        assert disco[self.id_first_line_disco]
        assert np.sum(disco) == 1

    def test_nb_timestep_overflow_nodisc_2(self):
        # on this _grid, first line with id 18 is overheated,
        # it is disconnected
        # then powerline 16 have a relative flow of 1.5916318201096937

        # in this scenario i don't have a second line disconnection because
        # the overflow is a soft overflow and  the powerline is presumably overflow since only 1
        # timestep
        case_file = self.case_file
        env_params = copy.deepcopy(self.env_params)
        env_params.HARD_OVERFLOW_THRESHOLD = 1.5
        env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=env_params)
        self.backend.load_grid(self.path_matpower, case_file)

        env.timestep_overflow[self.id_2nd_line_disco] = 1

        thermal_limit = 10*self.lines_flows_init
        thermal_limit[self.id_first_line_disco] = self.lines_flows_init[self.id_first_line_disco]/2
        thermal_limit[self.id_2nd_line_disco] = 400
        self.backend.set_thermal_limit(thermal_limit)

        disco, infos = self.backend.next_grid_state(env, is_dc=False)
        assert len(infos) == 1  # check that don't simulate a cascading failure
        assert disco[self.id_first_line_disco]
        assert np.sum(disco) == 1

    def test_nb_timestep_overflow_disc2(self):
        # on this _grid, first line with id 18 is overheated,
        # it is disconnected
        # then powerline 16 have a relative flow of 1.5916318201096937

        # in this scenario I have a second disconnection, because the powerline is allowed to be on overflow for 2
        # timestep and is still on overflow here.
        case_file = self.case_file
        env_params = copy.deepcopy(self.env_params)
        env_params.HARD_OVERFLOW_THRESHOLD = 1.5
        env_params.NB_TIMESTEP_POWERFLOW_ALLOWED = 2
        env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=env_params)
        self.backend.load_grid(self.path_matpower, case_file)

        env.timestep_overflow[self.id_2nd_line_disco] = 2

        thermal_limit = 10*self.lines_flows_init
        thermal_limit[self.id_first_line_disco] = self.lines_flows_init[self.id_first_line_disco]/2
        thermal_limit[self.id_2nd_line_disco] = 400
        self.backend.set_thermal_limit(thermal_limit)

        disco, infos = self.backend.next_grid_state(env, is_dc=False)
        assert len(infos) == 2  # check that there is a cascading failure of length 2
        assert disco[self.id_first_line_disco]
        assert disco[self.id_2nd_line_disco]
        assert np.sum(disco) == 2
        for i, grid_tmp in enumerate(infos):
            assert (not grid_tmp.get_line_status()[self.id_first_line_disco])
            if i == 1:
                assert (not grid_tmp.get_line_status()[self.id_2nd_line_disco])


class TestChangeBusAffectRightBus(unittest.TestCase):
    def test_set_bus(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make()
        env.reset()
        # action = env.helper_action_player({"change_bus": {"lines_or_id": [17]}})
        action = env.helper_action_player({"set_bus": {"lines_or_id": [(17, 2)]}})
        obs, reward, done, info = env.step(action)
        assert np.all(np.isfinite(obs.v_or))
        assert np.sum(env.backend._grid["bus"]["in_service"]) == 15

    def test_change_bus(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make()
        env.reset()
        action = env.helper_action_player({"change_bus": {"lines_or_id": [17]}})
        obs, reward, done, info = env.step(action)
        assert np.all(np.isfinite(obs.v_or))
        assert np.sum(env.backend._grid["bus"]["in_service"]) == 15

    def test_change_bustwice(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make()
        env.reset()
        action = env.helper_action_player({"change_bus": {"lines_or_id": [17]}})
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        assert np.all(np.isfinite(obs.v_or))
        assert np.sum(env.backend._grid["bus"]["in_service"]) == 14

    def test_isolate_load(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make()
        act = env.action_space({"set_bus": {"loads_id": [(0, 2)]}})
        obs, reward, done, info = env.step(act)
        assert done, "an isolated laod has not lead to a game over"

    def test_reco_disco_bus(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_case1 = make("case5_example", gamerules_class=AlwaysLegal)
        obs = env_case1.reset()  # reset is good
        act = env_case1.action_space.disconnect_powerline(line_id=5)  # I disconnect a powerline
        obs, reward, done, info = env_case1.step(act)  # do the action, it's valid
        act_case1 = env_case1.action_space.reconnect_powerline(line_id=5, bus_or=2, bus_ex=2)  # reconnect powerline on bus 2 both ends
        # this should lead to a game over a the powerline is out of the grid, 2 buses are, but without anything
        # this is a non connex grid
        obs_case1, reward_case1, done_case1, info_case1 = env_case1.step(act_case1)
        assert done_case1

    def test_reco_disco_bus2(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_case2 = make("case5_example", gamerules_class=AlwaysLegal)
        obs = env_case2.reset()  # reset is good
        obs, reward, done, info = env_case2.step(env_case2.action_space())  # do the action, it's valid
        act_case2 = env_case2.action_space.reconnect_powerline(line_id=5, bus_or=2, bus_ex=2)  # reconnect powerline on bus 2 both ends
        # this should lead to a game over a the powerline is out of the grid, 2 buses are, but without anything
        # this is a non connex grid
        obs_case2, reward_case2, done_case2, info_case2 = env_case2.step(act_case2)
        # this was illegal before, but test it is still illegal
        assert done_case2

    def test_reco_disco_bus3(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_case2 = make("case5_example", gamerules_class=AlwaysLegal)
        obs = env_case2.reset()  # reset is good
        obs, reward, done, info = env_case2.step(env_case2.action_space())  # do the action, it's valid
        act_case2 = env_case2.action_space.reconnect_powerline(line_id=5, bus_or=1, bus_ex=2)  # reconnect powerline on bus 2 both ends
        # this should not lead to a game over this time, the grid is connex!
        obs_case2, reward_case2, done_case2, info_case2 = env_case2.step(act_case2)
        assert done_case2 is False

    def test_reco_disco_bus4(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_case2 = make("case5_example", gamerules_class=AlwaysLegal)
        obs = env_case2.reset()  # reset is good
        obs, reward, done, info = env_case2.step(env_case2.action_space())  # do the action, it's valid
        act_case2 = env_case2.action_space.reconnect_powerline(line_id=5, bus_or=2, bus_ex=1)  # reconnect powerline on bus 2 both ends
        # this should not lead to a game over this time, the grid is connex!
        obs_case2, reward_case2, done_case2, info_case2 = env_case2.step(act_case2)
        assert done_case2 is False

    def test_reco_disco_bus5(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_case2 = make("case5_example", gamerules_class=AlwaysLegal)
        obs = env_case2.reset()  # reset is good
        act_case2 = env_case2.action_space({"set_bus": {"lines_or_id": [(5,2)], "lines_ex_id": [(5,2)]}})  # reconnect powerline on bus 2 both ends
        # this should not lead to a game over this time, the grid is connex!
        obs_case2, reward_case2, done_case2, info_case2 = env_case2.step(act_case2)
        assert done_case2


class TestShuntAction(HelperTests):
    def test_shunt_ambiguous_id_incorrect(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", gamerules_class=AlwaysLegal, action_class=CompleteAction) as env_case2:
                with self.assertRaises(AmbiguousAction):
                    act = env_case2.action_space({"shunt": {"set_bus": [(0, 2)]}})

    def test_shunt_effect(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_ref = make("case14_realistic", gamerules_class=AlwaysLegal, action_class=CompleteAction)
            env_change_q = make("case14_realistic", gamerules_class=AlwaysLegal, action_class=CompleteAction)

        obs_ref, *_ = env_ref.step(env_ref.action_space())
        obs_change_p, *_ = env_change_q.step(env_change_q.action_space({"shunt": {"shunt_q": [(0, -30)]}}))
        assert obs_ref.v_or[10] < obs_change_p.v_or[10]
        obs_change_p, *_ = env_change_q.step(env_change_q.action_space({"shunt": {"shunt_q": [(0, +30)]}}))
        obs_ref, *_ = env_ref.step(env_ref.action_space())
        assert obs_ref.v_or[10] > obs_change_p.v_or[10]
        obs_change_p, *_ = env_change_q.step(env_change_q.action_space({"shunt": {"set_bus": [(0, -1)]}}))
        env_ref.backend._grid.shunt["in_service"] = False  # force disconnection of shunt
        obs_ref, *_ = env_ref.step(env_ref.action_space())
        assert np.abs(obs_ref.v_or[10] - obs_change_p.v_or[10]) < self.tol_one

class TestResetEqualsLoadGrid(unittest.TestCase):
    def setUp(self):
        self.env1 = make("case5_example", backend=PandaPowerBackend())
        self.backend1 = self.env1.backend
        self.env2 = make("case5_example", backend=PandaPowerBackend())
        self.backend2 = self.env2.backend

    def tearDown(self):
        self.env1.close()
        self.env2.close()

    def test_reset_equals_reset(self):
        # Reset backend1 with reset
        self.env1.reset()
        # Reset backend2 with reset
        self.env2.reset()

        # Compare
        assert np.all(self.backend1.prod_pu_to_kv == self.backend2.prod_pu_to_kv)
        assert np.all(self.backend1.load_pu_to_kv == self.backend2.load_pu_to_kv)
        assert np.all(self.backend1.lines_or_pu_to_kv == self.backend2.lines_or_pu_to_kv)
        assert np.all(self.backend1.lines_ex_pu_to_kv == self.backend2.lines_ex_pu_to_kv)
        assert np.all(self.backend1.p_or == self.backend2.p_or)
        assert np.all(self.backend1.q_or == self.backend2.q_or)
        assert np.all(self.backend1.v_or == self.backend2.v_or)
        assert np.all(self.backend1.a_or == self.backend2.a_or)
        assert np.all(self.backend1.p_ex == self.backend2.p_ex)
        assert np.all(self.backend1.a_ex == self.backend2.a_ex)
        assert np.all(self.backend1.v_ex == self.backend2.v_ex)

    def test_reset_equals_load_grid(self):
        # Reset backend1 with reset
        self.env1.reset()
        # Reset backend2 with load_grid
        self.backend2.reset = self.backend2.load_grid
        self.env2.reset()

        # Compare
        assert np.all(self.backend1.prod_pu_to_kv == self.backend2.prod_pu_to_kv)
        assert np.all(self.backend1.load_pu_to_kv == self.backend2.load_pu_to_kv)
        assert np.all(self.backend1.lines_or_pu_to_kv == self.backend2.lines_or_pu_to_kv)
        assert np.all(self.backend1.lines_ex_pu_to_kv == self.backend2.lines_ex_pu_to_kv)
        assert np.all(self.backend1.p_or == self.backend2.p_or)
        assert np.all(self.backend1.q_or == self.backend2.q_or)
        assert np.all(self.backend1.v_or == self.backend2.v_or)
        assert np.all(self.backend1.a_or == self.backend2.a_or)
        assert np.all(self.backend1.p_ex == self.backend2.p_ex)
        assert np.all(self.backend1.a_ex == self.backend2.a_ex)
        assert np.all(self.backend1.v_ex == self.backend2.v_ex)

    def test_load_grid_equals_load_grid(self):
        # Reset backend1 with load_grid
        self.backend1.reset = self.backend1.load_grid
        self.env1.reset()
        # Reset backend2 with load_grid
        self.backend2.reset = self.backend2.load_grid
        self.env2.reset()

        # Compare
        assert np.all(self.backend1.prod_pu_to_kv == self.backend2.prod_pu_to_kv)
        assert np.all(self.backend1.load_pu_to_kv == self.backend2.load_pu_to_kv)
        assert np.all(self.backend1.lines_or_pu_to_kv == self.backend2.lines_or_pu_to_kv)
        assert np.all(self.backend1.lines_ex_pu_to_kv == self.backend2.lines_ex_pu_to_kv)
        assert np.all(self.backend1.p_or == self.backend2.p_or)
        assert np.all(self.backend1.q_or == self.backend2.q_or)
        assert np.all(self.backend1.v_or == self.backend2.v_or)
        assert np.all(self.backend1.a_or == self.backend2.a_or)
        assert np.all(self.backend1.p_ex == self.backend2.p_ex)
        assert np.all(self.backend1.a_ex == self.backend2.a_ex)
        assert np.all(self.backend1.v_ex == self.backend2.v_ex)

    def test_obs_from_same_chronic(self):
        # Store first observation
        obs1 = self.env1.current_obs
        obs2 = None
        for i in range(3):
            self.env1.step(self.env1.action_space({}))

        # Reset to first chronic
        self.env1.chronics_handler.tell_id(-1)
        self.env1.reset()

        # Store second observation
        obs2 = self.env1.current_obs

        # Compare
        assert np.allclose(obs1.prod_p, obs2.prod_p)
        assert np.allclose(obs1.prod_q, obs2.prod_q)
        assert np.allclose(obs1.prod_v, obs2.prod_v)
        assert np.allclose(obs1.load_p, obs2.load_p)
        assert np.allclose(obs1.load_q, obs2.load_q)
        assert np.allclose(obs1.load_v, obs2.load_v)
        assert np.allclose(obs1.p_or, obs2.p_or)
        assert np.allclose(obs1.q_or, obs2.q_or)
        assert np.allclose(obs1.v_or, obs2.v_or)
        assert np.allclose(obs1.a_or, obs2.a_or)
        assert np.allclose(obs1.p_ex, obs2.p_ex)
        assert np.allclose(obs1.q_ex, obs2.q_ex)
        assert np.allclose(obs1.v_ex, obs2.v_ex)
        assert np.allclose(obs1.a_ex, obs2.a_ex)
        assert np.allclose(obs1.rho, obs2.rho)
        assert np.all(obs1.line_status == obs2.line_status)
        assert np.all(obs1.topo_vect == obs2.topo_vect)
        assert np.all(obs1.timestep_overflow == obs2.timestep_overflow)
        assert np.all(obs1.time_before_cooldown_line == obs2.time_before_cooldown_line)
        assert np.all(obs1.time_before_cooldown_sub == obs2.time_before_cooldown_sub)
        assert np.all(obs1.time_before_line_reconnectable == obs2.time_before_line_reconnectable)
        assert np.all(obs1.time_next_maintenance == obs2.time_next_maintenance)
        assert np.all(obs1.duration_next_maintenance == obs2.duration_next_maintenance)
        assert np.all(obs1.target_dispatch == obs2.target_dispatch)
        assert np.all(obs1.actual_dispatch == obs2.actual_dispatch)
    
if __name__ == "__main__":
    unittest.main()
