# making some test that the backned is working as expected
import os  # load the python os default module
import sys
import unittest

import numpy as np
import copy
import pdb

from helper_path_test import PATH_DATA_TEST_PP, PATH_CHRONICS

from Action import HelperAction
from BackendPandaPower import PandaPowerBackend
from Parameters import Parameters
from ChronicsHandler import ChronicsHandler, ChangeNothing
from Environment import Environment
from Exceptions import *
from GameRules import GameRules

PATH_DATA_TEST = PATH_DATA_TEST_PP
import pandapower as pppp

class TestLoadingADN(unittest.TestCase):
    def setUp(self):
        self.tolvect = 1e-2
        self.tol_one = 1e-5

    def test_load_file(self):
        backend = PandaPowerBackend()
        path_matpower = PATH_DATA_TEST
        case_file = "test_case14.json"
        backend.load_grid(path_matpower, case_file)

        assert backend.n_lines == 20
        assert backend.n_generators == 5
        assert backend.n_loads == 11
        assert backend.n_substations == 14

        name_lines = ['0_1_0', '0_4_1',  '8_9_2', '8_13_3', '9_10_4', '11_12_5', '12_13_6', '1_2_7',
                      '1_3_8', '1_4_9', '2_3_10', '3_4_11', '5_10_12', '5_11_13', '5_12_14',  '3_6_15',
                      '3_8_16', '4_5_17', '6_7_18', '6_8_19']
        name_lines = np.array(name_lines)
        assert np.all(sorted(backend.name_lines) == sorted(name_lines))

        name_subs = ['sub_0', 'sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6', 'sub_7', 'sub_8', 'sub_9', 'sub_10',
                     'sub_11', 'sub_12', 'sub_13']
        name_subs = np.array(name_subs)
        assert np.all(sorted(backend.name_subs) == sorted(name_subs))

        name_prods = ['gen_0_4', 'gen_1_0', 'gen_2_1', 'gen_5_2', 'gen_7_3']
        name_prods = np.array(name_prods)
        assert np.all(sorted(backend.name_prods) == sorted(name_prods))

        name_loads = ['load_1_0', 'load_2_1', 'load_13_2', 'load_3_3', 'load_4_4', 'load_5_5', 'load_8_6',
                      'load_9_7', 'load_10_8', 'load_11_9', 'load_12_10']
        name_loads = np.array(name_loads)
        assert np.all(sorted(backend.name_loads) == sorted(name_loads))

        assert np.all(backend.get_topo_vect() == np.ones(np.sum(backend.subs_elements)))

        backend.runpf()
        try:
            p_subs, q_subs, p_bus, q_bus = backend.check_kirchoff()
            assert np.max(np.abs(p_subs)) <= self.tolvect
            assert np.max(np.abs(q_subs)) <= self.tolvect
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect

        except Grid4RLException:
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
        self.game_rules = GameRules()
        self.action_env = HelperAction(name_prod=self.backend.name_prods,
                                       name_load=self.backend.name_loads,
                                       name_line=self.backend.name_lines,
                                  subs_info=self.backend.subs_elements,
                                  load_to_subid=self.backend.load_to_subid,
                                  gen_to_subid=self.backend.gen_to_subid,
                                  lines_or_to_subid=self.backend.lines_or_to_subid,
                                  lines_ex_to_subid=self.backend.lines_ex_to_subid, #####
                                  load_to_sub_pos=self.backend.load_to_sub_pos,
                                  gen_to_sub_pos=self.backend.gen_to_sub_pos,
                                  lines_or_to_sub_pos=self.backend.lines_or_to_sub_pos,
                                  lines_ex_to_sub_pos=self.backend.lines_ex_to_sub_pos, #####
                                  load_pos_topo_vect=self.backend.load_pos_topo_vect,
                                  gen_pos_topo_vect=self.backend.gen_pos_topo_vect,
                                  lines_or_pos_topo_vect=self.backend.lines_or_pos_topo_vect,
                                  lines_ex_pos_topo_vect=self.backend.lines_ex_pos_topo_vect,
                                       game_rules=self.game_rules)

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
        for i in range(self.backend.n_lines):
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
        action = self.action_env({"_injection": {"load_p": ratio*init_lp,
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
        except Grid4RLException:
            pass

        assert self.compare_vect(ratio*init_gp, after_gp)  # check i didn't modify the generators
        assert np.all(init_ls == after_ls)  # check i didn't disconnect any powerlines



        after_flow, *_ = self.backend.lines_or_info()
        assert self.compare_vect(ratio*init_flow, after_flow) # probably an error with the DC approx

    def test_apply_action_maintenance(self):
        # retrieve some initial data to be sure only a subpart of the _grid is modified
        conv = self.backend.runpf()
        init_lp, *_ = self.backend.loads_info()
        init_gp, *_ = self.backend.generators_info()

        # check that _maintenance vector is properly taken into account
        maintenance = np.full((self.backend.n_lines,), fill_value=False, dtype=np.bool)
        maintenance[19] = True
        action = self.action_env({"_maintenance": maintenance})  # update the action

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

        # check that _maintenance vector is properly taken into account
        maintenance = np.full((self.backend.n_lines,), fill_value=False, dtype=np.bool)
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

        # check that _maintenance vector is properly taken into account
        maintenance = np.full((self.backend.n_lines,), fill_value=False, dtype=np.bool)
        maintenance[19] = True

        disc = np.full((self.backend.n_lines,), fill_value=False, dtype=np.bool)
        disc[17] = True

        action = self.action_env({"hazards": disc, "_maintenance": maintenance})  # update the action
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

        self.game_rules = GameRules()
        self.helper_action = HelperAction(name_prod=self.backend.name_prods,
                                       name_load=self.backend.name_loads,
                                       name_line=self.backend.name_lines,
                                  subs_info=self.backend.subs_elements,
                                  load_to_subid=self.backend.load_to_subid,
                                  gen_to_subid=self.backend.gen_to_subid,
                                  lines_or_to_subid=self.backend.lines_or_to_subid,
                                  lines_ex_to_subid=self.backend.lines_ex_to_subid, #####
                                  load_to_sub_pos=self.backend.load_to_sub_pos,
                                  gen_to_sub_pos=self.backend.gen_to_sub_pos,
                                  lines_or_to_sub_pos=self.backend.lines_or_to_sub_pos,
                                  lines_ex_to_sub_pos=self.backend.lines_ex_to_sub_pos, #####
                                  load_pos_topo_vect=self.backend.load_pos_topo_vect,
                                  gen_pos_topo_vect=self.backend.gen_pos_topo_vect,
                                  lines_or_pos_topo_vect=self.backend.lines_or_pos_topo_vect,
                                  lines_ex_pos_topo_vect=self.backend.lines_ex_pos_topo_vect,
                                       game_rules=self.game_rules)

    # Cette méthode sera appelée après chaque test.
    def tearDown(self):
        pass

    def compare_vect(self, pred, true):
        return np.max(np.abs(pred- true)) <= self.tolvect

    def test_topo_set1sub(self):
        # retrieve some initial data to be sure only a subpart of the _grid is modified
        conv = self.backend.runpf()
        init_amps_flow = self.backend.get_line_flow()

        # check that _maintenance vector is properly taken into account
        arr = np.array([1, 1, 1, 2, 2, 2], dtype=np.int)
        id_=1
        action = self.helper_action({"set_bus": {"substations": [(id_, arr)]}})

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
        lor_ids = np.where(self.backend.lines_or_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.lines_or_pos_topo_vect[lor_ids]] == arr[self.backend.lines_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.lines_ex_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.lines_ex_pos_topo_vect[lex_ids]] == arr[self.backend.lines_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == arr[self.backend.gen_to_sub_pos[gen_ids]])


        after_amps_flow_th = np.array([6.38865247e+02, 3.81726828e+02, 1.78001287e+04, 2.70742428e+04,
                                       1.06755055e+04, 4.71160165e+03, 1.52265925e+04, 3.37755751e+02,
                                       3.00535519e+02, 5.01164454e-13, 7.01900962e+01, 1.73874580e+02,
                                       2.08904697e+04, 2.11757439e+04, 4.93863382e+04, 1.31935835e+02,
                                       6.99779475e+01, 1.85068609e+02, 7.47283039e+02, 1.14125596e+03])
        assert self.compare_vect(after_amps_flow, after_amps_flow_th)

        try:
            p_subs, q_subs, p_bus, q_bus = self.backend.check_kirchoff()
            assert np.max(np.abs(p_subs)) <= self.tolvect, "problem with active values, at substation"
            assert np.max(np.abs(q_subs)) <= self.tolvect, "problem with reactive values, at substation"
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect, "problem with active values, at a bus"
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect, "problem with reaactive values, at a load"

        except Grid4RLException:
            pass

    def test_topo_change1sub(self):
        # check that switching the bus of 3 object is equivalent to set them to bus 2 (as above)
        conv = self.backend.runpf()
        init_amps_flow = self.backend.get_line_flow()

        # check that _maintenance vector is properly taken into account
        arr = np.array([False, False, False, True, True, True], dtype=np.bool)
        id_ = 1
        action = self.helper_action({"change_bus": {"substations": [(id_, arr)]}})

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
        lor_ids = np.where(self.backend.lines_or_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.lines_or_pos_topo_vect[lor_ids]] == 1+arr[self.backend.lines_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.lines_ex_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.lines_ex_pos_topo_vect[lex_ids]] == 1+arr[self.backend.lines_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid==id_)[0]
        assert np.all(topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == 1+arr[self.backend.gen_to_sub_pos[gen_ids]])


        after_amps_flow_th = np.array([6.38865247e+02, 3.81726828e+02, 1.78001287e+04, 2.70742428e+04,
                                       1.06755055e+04, 4.71160165e+03, 1.52265925e+04, 3.37755751e+02,
                                       3.00535519e+02, 5.01164454e-13, 7.01900962e+01, 1.73874580e+02,
                                       2.08904697e+04, 2.11757439e+04, 4.93863382e+04, 1.31935835e+02,
                                       6.99779475e+01, 1.85068609e+02, 7.47283039e+02, 1.14125596e+03])
        assert self.compare_vect(after_amps_flow, after_amps_flow_th)

        try:
            p_subs, q_subs, p_bus, q_bus = self.backend.check_kirchoff()
            assert np.max(np.abs(p_subs)) <= self.tolvect, "problem with active values, at substation"
            assert np.max(np.abs(q_subs)) <= self.tolvect, "problem with reactive values, at substation"
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect, "problem with active values, at a bus"
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect, "problem with reaactive values, at a load"

        except Grid4RLException:
            pass

    def test_topo_change_1sub_twice(self):
        # check that switching the bus of 3 object is equivalent to set them to bus 2 (as above)
        # and that setting it again is equivalent to doing nothing
        conv = self.backend.runpf()
        init_amps_flow = np.array([el for el in self.backend.get_line_flow()])

        # check that _maintenance vector is properly taken into account
        arr = np.array([False, False, False, True, True, True], dtype=np.bool)
        id_ = 1
        action = self.helper_action({"change_bus": {"substations": [(id_, arr)]}})

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
        lor_ids = np.where(self.backend.lines_or_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.lines_or_pos_topo_vect[lor_ids]] == 1+arr[self.backend.lines_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.lines_ex_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.lines_ex_pos_topo_vect[lex_ids]] == 1+arr[self.backend.lines_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid == id_)[0]
        assert np.all(topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == 1+arr[self.backend.gen_to_sub_pos[gen_ids]])

        after_amps_flow_th = np.array([6.38865247e+02, 3.81726828e+02, 1.78001287e+04, 2.70742428e+04,
                                       1.06755055e+04, 4.71160165e+03, 1.52265925e+04, 3.37755751e+02,
                                       3.00535519e+02, 5.01164454e-13, 7.01900962e+01, 1.73874580e+02,
                                       2.08904697e+04, 2.11757439e+04, 4.93863382e+04, 1.31935835e+02,
                                       6.99779475e+01, 1.85068609e+02, 7.47283039e+02, 1.14125596e+03])
        assert self.compare_vect(after_amps_flow, after_amps_flow_th)
        try:
            p_subs, q_subs, p_bus, q_bus = self.backend.check_kirchoff()
            assert np.max(np.abs(p_subs)) <= self.tolvect, "problem with active values, at substation"
            assert np.max(np.abs(q_subs)) <= self.tolvect, "problem with reactive values, at substation"
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect, "problem with active values, at a bus"
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect, "problem with reaactive values, at a load"

        except Grid4RLException:
            pass

        action = self.helper_action({"change_bus": {"substations": [(id_, arr)]}})

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

        except Grid4RLException:
            pass


    def test_topo_change_2sub(self):
        # check that _maintenance vector is properly taken into account
        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        id_1 = 1
        id_2 = 12
        action = self.helper_action({"change_bus": {"substations": [(id_1, arr1)]},
                                     "set_bus": {"substations": [(id_2, arr2)]}})
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
        lor_ids = np.where(self.backend.lines_or_to_subid == id_1)[0]
        assert np.all(
            topo_vect[self.backend.lines_or_pos_topo_vect[lor_ids]] == 1+arr1[self.backend.lines_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.lines_ex_to_subid == id_1)[0]
        assert np.all(
            topo_vect[self.backend.lines_ex_pos_topo_vect[lex_ids]] == 1+arr1[self.backend.lines_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid == id_1)[0]
        assert np.all(topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == 1+arr1[self.backend.gen_to_sub_pos[gen_ids]])


        load_ids = np.where(self.backend.load_to_subid == id_2)[0]
        assert np.all(
            topo_vect[self.backend.load_pos_topo_vect[load_ids]] == arr2[self.backend.load_to_sub_pos[load_ids]])
        lor_ids = np.where(self.backend.lines_or_to_subid == id_2)[0]
        assert np.all(
            topo_vect[self.backend.lines_or_pos_topo_vect[lor_ids]] == arr2[self.backend.lines_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.lines_ex_to_subid == id_2)[0]
        assert np.all(
            topo_vect[self.backend.lines_ex_pos_topo_vect[lex_ids]] == arr2[self.backend.lines_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid == id_2)[0]
        assert np.all(topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == arr2[self.backend.gen_to_sub_pos[gen_ids]])

        after_amps_flow = self.backend.get_line_flow()
        after_amps_flow_th = np.array([6.06484526e+02, 3.53118362e+02, 1.31069171e+04, 2.91362693e+04,
                                       1.81182785e+04, 1.36186919e+04, 1.36186919e+04, 3.26671053e+02,
                                       2.79359713e+02, 4.86200936e-13, 7.92740141e+01, 1.87635540e+02,
                                       2.82695392e+04, 3.02581716e+04, 3.25275006e-10, 1.21389438e+02,
                                       6.44646825e+01, 1.43147411e+02, 7.56229924e+02, 1.07504986e+03])
        assert self.compare_vect(after_amps_flow, after_amps_flow_th)

        try:
            p_subs, q_subs, p_bus, q_bus = self.backend.check_kirchoff()
            assert np.max(np.abs(p_subs)) <= self.tolvect, "problem with active values, at substation"
            assert np.max(np.abs(q_subs)) <= self.tolvect, "problem with reactive values, at substation"
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect, "problem with active values, at a bus"
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect, "problem with reaactive values, at a load"

        except Grid4RLException:
            pass



# class TestEnvPerformsCorrectCascadingFailures(unittest.TestCase):
#     """
#     Test the "next_grid_state" method of the back-end
#     """
#     def setUp(self):
#         self.backend = PandaPowerBackend()
#         self.path_matpower = path
#         self.case_file = "test_case14.json"
#         self.backend.load_grid(self.path_matpower, self.case_file)
#         self.tolvect = 1e-2
#         self.tol_one = 1e-5
#
#         self.action_env = HelperAction(_n_gen=self.backend.n_generators,
#                                   _n_load=self.backend.n_loads,
#                                   _n_lines=self.backend._n_lines,
#                                   _subs_info=self.backend.subs_elements,
#                                   _load_to_subid=self.backend._load_to_subid,
#                                   _gen_to_subid=self.backend._gen_to_subid,
#                                   _lines_or_to_subid=self.backend._lines_or_to_subid,
#                                   _lines_ex_to_subid=self.backend._lines_ex_to_subid, #####
#                                   _load_to_sub_pos=self.backend._load_to_sub_pos,
#                                   _gen_to_sub_pos=self.backend._gen_to_sub_pos,
#                                   _lines_or_to_sub_pos=self.backend._lines_or_to_sub_pos,
#                                   _lines_ex_to_sub_pos=self.backend._lines_ex_to_sub_pos, #####
#                                   _load_pos_topo_vect=self.backend._load_pos_topo_vect,
#                                   _gen_pos_topo_vect=self.backend._gen_pos_topo_vect,
#                                   _lines_or_pos_topo_vect=self.backend._lines_or_pos_topo_vect,
#                                   _lines_ex_pos_topo_vect=self.backend._lines_ex_pos_topo_vect)
#
#
#         # _parameters for the environment
#         self.env_params = Parameters()
#
#         # used for init an env too
#         self.chronics_handler = ChronicsHandler()
#         # # hack here: i don't need to properly use the chronic handler, it's just here for initializing an environment
#         # self.chronics_handler.real_data.load_p = np.zeros(self.backend.n_loads)
#         # self.chronics_handler.real_data.load_q = np.zeros(self.backend.n_loads)
#         # self.chronics_handler.real_data.prod_p = np.zeros(self.backend.n_generators)
#         # self.chronics_handler.real_data.prod_v = np.zeros(self.backend.n_generators)
#         # self.chronics_handler.real_data._maintenance = np.zeros(self.backend._n_lines)
#         # self.chronics_handler.real_data.outage = np.zeros(self.backend._n_lines)
#
#     def next_grid_state_no_overflow(self):
#         # first i test that, when there is no overflow, i dont do a cascading failure
#         env = Environment(init_grid_path=os.path.join(self.path_matpower, self.case_file),
#                           backend=self.backend,
#                           chronics_handler=self.chronics_handler,
#                           _parameters=self.env_params)
#
#         disco, infos = self.backend.next_grid_state(env, is_dc=False)
#         assert not infos
#
#     def test_next_grid_state_1overflow(self):
#         # second i test that, when is one line on hard overflow it is disconnected
#         case_file = "ieee14_ADN_overflow1.xml"
#         env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
#                           backend=self.backend,
#                           chronics_handler=self.chronics_handler,
#                           _parameters=self.env_params)
#         self.backend.load_grid(self.path_matpower, case_file)
#         disco, infos = self.backend.next_grid_state(env, is_dc=False)
#         assert len(infos) == 1  # check that i have only one overflow
#
#     def test_next_grid_state_1overflow_envNoCF(self):
#         # third i test that, if a line is on hard overflow, but i'm on a "no cascading failure" mode,
#         # i don't simulate a cascading failure
#         self.env_params.NO_OVERFLOW_DISCONNECTION = True
#         case_file = "ieee14_ADN_overflow1.xml"
#         env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
#                           backend=self.backend,
#                           chronics_handler=self.chronics_handler,
#                           _parameters=self.env_params)
#         disco, infos = self.backend.next_grid_state(env, is_dc=False)
#         assert not infos # check that don't simulate a cascading failure
#         assert np.sum(disco) == 0
#
#     def test_nb_timestep_overflow(self):
#         case_file = "ieee14_ADN_overflow2.xml"
#         # on this _grid, first line with id 18 is overheated,
#         # it is disconnected
#         # then powerline 16 have a relative flow of 1.5916318201096937
#         # bck_cpy = self.backend.copy()
#         # in this scenario i don't have a second line disconnection.
#         env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
#                           backend=self.backend,
#                           chronics_handler=self.chronics_handler,
#                           _parameters=self.env_params)
#         self.backend.load_grid(self.path_matpower, case_file)
#         disco, infos = self.backend.next_grid_state(env, is_dc=False)
#         assert len(infos) == 1  # check that don't simulate a cascading failure
#         assert disco[17]
#         assert np.sum(disco) == 1
#
#     def test_nb_timestep_overflow_nodisc(self):
#         case_file = "ieee14_ADN_overflow2.xml"
#         # on this _grid, first line with id 18 is overheated,
#         # it is disconnected
#         # then powerline 16 have a relative flow of 1.5916318201096937
#
#         # in this scenario i don't have a second line disconnection because
#         # the overflow is a soft overflow and  the powerline is presumably overflow since 0
#         # timestep
#         self.env_params.NB_TIMESTEP_POWERFLOW_ALLOWED = 2
#         env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
#                           backend=self.backend,
#                           chronics_handler=self.chronics_handler,
#                           _parameters=self.env_params)
#         self.backend.load_grid(self.path_matpower, case_file)
#         disco, infos = self.backend.next_grid_state(env, is_dc=False)
#         assert len(infos) == 1  # check that don't simulate a cascading failure
#         assert disco[17]
#         assert np.sum(disco) == 1
#
#     def test_nb_timestep_overflow_nodisc_2(self):
#         case_file = "ieee14_ADN_overflow2.xml"
#         # on this _grid, first line with id 18 is overheated,
#         # it is disconnected
#         # then powerline 16 have a relative flow of 1.5916318201096937
#
#         # in this scenario i don't have a second line disconnection because
#         # the overflow is a soft overflow and  the powerline is presumably overflow since only 1
#         # timestep
#         self.env_params.NB_TIMESTEP_POWERFLOW_ALLOWED = 2
#         env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
#                           backend=self.backend,
#                           chronics_handler=self.chronics_handler,
#                           _parameters=self.env_params)
#         env._timestep_overflow[15] = 1
#         self.backend.load_grid(self.path_matpower, case_file)
#         disco, infos = self.backend.next_grid_state(env, is_dc=False)
#         assert len(infos) == 1  # check that don't simulate a cascading failure
#         assert disco[17]
#         assert np.sum(disco) == 1
#
#     def test_nb_timestep_overflow_disc(self):
#         case_file = "ieee14_ADN_overflow2.xml"
#         # on this _grid, first line with id 18 is overheated,
#         # it is disconnected
#         # then powerline 16 have a relative flow of 1.5916318201096937
#
#         # in this scenario I have a second disconnection, because the powerline is allowed to be on overflow for 2
#         # timestep and is still on overflow here.
#         self.env_params.NB_TIMESTEP_POWERFLOW_ALLOWED = 2
#         env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
#                           backend=self.backend,
#                           chronics_handler=self.chronics_handler,
#                           _parameters=self.env_params)
#         env._timestep_overflow[15] = 2
#         self.backend.load_grid(self.path_matpower, case_file)
#         disco, infos = self.backend.next_grid_state(env, is_dc=False)
#         assert len(infos) == 2  # check that there is a cascading failure of length 2
#         assert disco[17]
#         assert disco[15]
#         assert np.sum(disco) == 2
#
#         for i, grid_tmp in enumerate(infos):
#             assert not grid_tmp._grid.line_id(18).connected(), "the powerline id 18 should be disconnected"
#             if i == 1:
#                 assert not grid_tmp._grid.line_id(16).connected(), "the powerline id 16 should be disconnected"

    #TODO i need to check that:
    # - nb_timestep_overflow_allowed is working,
    # - cascading failure is working with depth >= 1
    # - make sure the cascading failure disconnect the proper powerlines

# TODO test also the methods added for observation:
"""
        self._line_status = copy.copy(backend.get_line_status())
        self._topo_vect = copy.copy(backend.get_topo_vect())
        # get the values related to continuous values
        self.prod_p, self.prod_q, self.prod_v = backend.generators_info()
        self.load_p, self.load_q, self.load_v = backend.loads_info()
        self.p_or, self.q_or, self.v_or, self.a_or = backend.lines_or_info()
        self.p_ex, self.q_ex, self.v_ex, self.a_ex = backend.lines_ex_info()
"""

# TODO refactor these tests to be both done in ADNBackend and PandaPowerBackend both coming from a file "test_Backend"

if __name__ == "__main__":
    unittest.main()
