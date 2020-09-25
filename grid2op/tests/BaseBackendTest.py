# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

# do some generic tests that can be implemented directly to test if a backend implementation can work out of the box
# with grid2op.
# see an example of test_Pandapower for how to use this suit.


import os
import numpy as np
import copy
from abc import ABC, abstractmethod
import inspect
try:
    # this is only available starting python 3.7 or 3.8... tests are with python 3.6 :-(
    from math import comb
except ImportError:
    def comb(n, k):
        if n == k:
            return 1
        if n < k:
            return 0
        res = 1
        acc = 1
        for i in range(k):
            res *= int((n-i))
        for i in range(1,k+1):
            res /= i
        return res

    """
    test to check that it's working
    for i in range(10):
        for j in range(10):
            me_ = comb(i,j)
            real_ = math.comb(i,j)
            assert me_ == real_, "{}, {}".format(i,j)
    """
import pdb
import warnings

import grid2op
from grid2op.dtypes import dt_int, dt_float
from grid2op.Action import ActionSpace, CompleteAction
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChronicsHandler
from grid2op.Environment import Environment
from grid2op.Exceptions import *
from grid2op.Rules import RulesChecker
from grid2op.MakeEnv import make
from grid2op.Rules import AlwaysLegal
from grid2op.Action._BackendAction import _BackendAction


class MakeBackend(ABC):
    @abstractmethod
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        pass

    def get_path(self):
        raise NotImplementedError("This function should be implemented for the test suit you are developping")

    def get_casefile(self):
        raise NotImplementedError("This function should be implemented for the test suit you are developping")

    def skip_if_needed(self):
        if hasattr(self, "tests_skipped"):
            nm_ = inspect.currentframe().f_back.f_code.co_name
            if nm_ in self.tests_skipped:
                self.skipTest("the test \"{}\" is skipped".format(nm_))


class BaseTestNames(MakeBackend):
    def test_properNames(self):
        backend = self.make_backend()
        path = self.get_path()
        with make(os.path.join(path, "5bus_example_diff_name"),
                  backend=backend,
                  ) as env:
            obs = env.reset()
            assert np.all(obs.name_load == ["tutu", "toto", "tata"])
            assert np.all(env.name_load == ["tutu", "toto", "tata"])


class BaseTestLoadingCase(MakeBackend):
    def test_load_file(self):
        backend = self.make_backend()
        path_matpower = self.get_path()
        case_file = self.get_casefile()
        backend.load_grid(path_matpower, case_file)
        backend.set_env_name("TestLoadingCase_env")
        backend.assert_grid_correct()

        assert backend.n_line == 20
        assert backend.n_gen == 5
        assert backend.n_load == 11
        assert backend.n_sub == 14

        name_line = ['0_1_0', '0_4_1', '8_9_2', '8_13_3', '9_10_4', '11_12_5', '12_13_6', '1_2_7',
                     '1_3_8', '1_4_9', '2_3_10', '3_4_11', '5_10_12', '5_11_13', '5_12_14', '3_6_15',
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

        conv = backend.runpf()
        assert conv, "powerflow diverge it is not supposed to!"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            p_subs, q_subs, p_bus, q_bus = backend.check_kirchoff()

        assert np.max(np.abs(p_subs)) <= self.tolvect
        assert np.max(np.abs(p_bus.flatten())) <= self.tolvect
        if backend.shunts_data_available:
            assert np.max(np.abs(q_subs)) <= self.tolvect
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect

    def test_assert_grid_correct(self):
        backend = self.make_backend()
        path_matpower = self.get_path()
        case_file = self.get_casefile()
        backend.load_grid(path_matpower, case_file)
        backend.set_env_name("TestLoadingCase_env2")
        backend.assert_grid_correct()
        conv = backend.runpf()
        assert conv, "powerflow diverge it is not supposed to!"
        backend.assert_grid_correct_after_powerflow()


class BaseTestLoadingBackendFunc(MakeBackend):
    def setUp(self):
        self.backend = self.make_backend()
        self.path_matpower = self.get_path()
        self.case_file = self.get_casefile()
        self.backend.load_grid(self.path_matpower, self.case_file)
        self.backend.set_env_name("TestLoadingBackendFunc_env")
        self.backend.assert_grid_correct()
        self.game_rules = RulesChecker()
        self.action_env_class = ActionSpace.init_grid(self.backend)
        self.action_env = self.action_env_class(gridobj=self.backend, legal_action=self.game_rules.legal_action)
        self.bkact_class = _BackendAction.init_grid(self.backend)
        self.backend.runpf()
        self.backend.assert_grid_correct_after_powerflow()

    def tearDown(self):
        pass

    def test_runpf_dc(self):
        self.skip_if_needed()
        conv = self.backend.runpf(is_dc=True)
        assert conv
        true_values_dc = np.array([147.83859556, 71.16140444, 5.7716542, 9.64132512,
                                   -3.2283458, 1.50735814, 5.25867488, 70.01463596,
                                   55.1518527, 40.9721069, -24.18536404, -61.74649065,
                                   6.7283458, 7.60735814, 17.25131674, 28.36115279,
                                   16.55182652, 42.78702069, 0., 28.36115279])
        p_or, *_ = self.backend.lines_or_info()
        assert self.compare_vect(p_or, true_values_dc)

    def test_runpf(self):
        self.skip_if_needed()
        true_values_ac = np.array([1.56882891e+02, 7.55103818e+01, 5.22755247e+00, 9.42638103e+00,
                                   -3.78532238e+00, 1.61425777e+00, 5.64385098e+00, 7.32375792e+01,
                                   5.61314959e+01, 4.15162150e+01, -2.32856901e+01, -6.11582304e+01,
                                   7.35327698e+00, 7.78606702e+00, 1.77479769e+01, 2.80741759e+01,
                                   1.60797576e+01, 4.40873209e+01, -1.11022302e-14, 2.80741759e+01])
        conv = self.backend.runpf(is_dc=False)
        assert conv
        p_or, *_ = self.backend.lines_or_info()
        assert self.compare_vect(p_or, true_values_ac)

    def test_voltage_convert_powerlines(self):
        self.skip_if_needed()
        # i have the correct voltages in powerlines if the formula to link mw, mvar, kv and amps is correct
        conv = self.backend.runpf(is_dc=False)
        assert conv, "powerflow diverge at loading"

        p_or, q_or, v_or, a_or = self.backend.lines_or_info()
        a_th = np.sqrt(p_or ** 2 + q_or ** 2) * 1e3 / (np.sqrt(3) * v_or)
        assert self.compare_vect(a_th, a_or)

        p_ex, q_ex, v_ex, a_ex = self.backend.lines_ex_info()
        a_th = np.sqrt(p_ex ** 2 + q_ex ** 2) * 1e3 / (np.sqrt(3) * v_ex)
        assert self.compare_vect(a_th, a_ex)

    def test_voltages_correct_load_gen(self):
        self.skip_if_needed()
        # i have the right voltages to generators and load, if it's the same as the voltage (correct from the above test)
        # of the powerline connected to it.

        conv = self.backend.runpf(is_dc=False)
        assert conv, "powerflow diverge at loading"
        load_p, load_q, load_v = self.backend.loads_info()
        gen_p, gen__q, gen_v = self.backend.generators_info()
        p_or, q_or, v_or, a_or = self.backend.lines_or_info()
        p_ex, q_ex, v_ex, a_ex = self.backend.lines_ex_info()

        for c_id, sub_id in enumerate(self.backend.load_to_subid):
            l_ids = np.where(self.backend.line_or_to_subid == sub_id)[0]
            if len(l_ids):
                l_id = l_ids[0]
                assert np.abs(v_or[l_id] - load_v[c_id]) <= self.tol_one, "problem for load {}".format(c_id)
                continue

            l_ids = np.where(self.backend.line_ex_to_subid == sub_id)[0]
            if len(l_ids):
                l_id = l_ids[0]
                assert np.abs(v_ex[l_id] - load_v[c_id]) <= self.tol_one, "problem for load {}".format(c_id)
                continue
            assert False, "load {} has not been checked".format(c_id)

        for g_id, sub_id in enumerate(self.backend.gen_to_subid):
            l_ids = np.where(self.backend.line_or_to_subid == sub_id)[0]
            if len(l_ids):
                l_id = l_ids[0]
                assert np.abs(v_or[l_id] - gen_v[g_id]) <= self.tol_one, "problem for generator {}".format(g_id)
                continue

            l_ids = np.where(self.backend.line_ex_to_subid == sub_id)[0]
            if len(l_ids):
                l_id = l_ids[0]
                assert np.abs(v_ex[l_id] - gen_v[g_id]) <= self.tol_one, "problem for generator {}".format(g_id)
                continue
            assert False, "generator {} has not been checked".format(g_id)

    def test_copy(self):
        self.skip_if_needed()
        conv = self.backend.runpf(is_dc=False)
        assert conv, "powerflow diverge at loading"
        l_id = 3

        p_or_orig, *_ = self.backend.lines_or_info()
        adn_backend_cpy = self.backend.copy()

        self.backend._disconnect_line(l_id)
        conv = self.backend.runpf(is_dc=False)
        assert conv
        conv2 = adn_backend_cpy.runpf(is_dc=False)
        assert conv2
        p_or_ref, *_ = self.backend.lines_or_info()
        p_or, *_ = adn_backend_cpy.lines_or_info()
        assert self.compare_vect(p_or_orig, p_or), "the copied object affects its original 'parent'"
        assert np.abs(p_or_ref[l_id]) <= self.tol_one, "powerline {} has not been disconnected".format(l_id)

    def test_copy2(self):
        self.skip_if_needed()
        self.backend._disconnect_line(8)
        conv = self.backend.runpf(is_dc=False)
        p_or_orig, *_ = self.backend.lines_or_info()

        adn_backend_cpy = self.backend.copy()
        adn_backend_cpy._disconnect_line(11)
        assert not adn_backend_cpy.get_line_status()[8]
        assert not adn_backend_cpy.get_line_status()[11]
        assert not self.backend.get_line_status()[8]
        assert self.backend.get_line_status()[11]

    def test_get_private_line_status(self):
        self.skip_if_needed()
        if hasattr(self.backend, "_get_line_status"):
            assert np.all(self.backend._get_line_status())
        else:
            assert np.all(self.backend.get_line_status())

        self.backend._disconnect_line(3)
        if hasattr(self.backend, "_get_line_status"):
            vect_ = self.backend._get_line_status()
        else:
            vect_ = self.backend.get_line_status()
        assert np.sum(~vect_) == 1
        assert not vect_[3]

    def test_get_line_flow(self):
        self.skip_if_needed()
        self.backend.runpf(is_dc=False)
        true_values_ac = np.array([-20.40429168, 3.85499114, 4.2191378, 3.61000624,
                                   -1.61506292, 0.75395917, 1.74717378, 3.56020295,
                                   -1.5503504, 1.17099786, 4.47311562, 15.82364194,
                                   3.56047297, 2.50341424, 7.21657539, -9.68106571,
                                   -0.42761118, 12.47067981, -17.16297051, 5.77869057])
        p_or_orig, q_or_orig, *_ = self.backend.lines_or_info()
        assert self.compare_vect(q_or_orig, true_values_ac)

        self.backend._disconnect_line(3)
        a = self.backend.runpf(is_dc=False)
        true_values_ac = np.array([-20.40028207, 3.65600775, 3.77916284, 0.,
                                   -2.10761554, 1.34025308, 5.86505081, 3.58514625,
                                   -2.28717836, 0.81979017, 3.72328838, 17.09556423,
                                   3.9548798, 3.18389804, 11.24144925, -11.09660174,
                                   -1.70423701, 13.14347167, -14.82917601, 2.276297])
        p_or_orig, q_or_orig, *_ = self.backend.lines_or_info()
        assert self.compare_vect(q_or_orig, true_values_ac)

    def test_pf_ac_dc(self):
        self.skip_if_needed()
        true_values_ac = np.array([-20.40429168, 3.85499114, 4.2191378, 3.61000624,
                                   -1.61506292, 0.75395917, 1.74717378, 3.56020295,
                                   -1.5503504, 1.17099786, 4.47311562, 15.82364194,
                                   3.56047297, 2.50341424, 7.21657539, -9.68106571,
                                   -0.42761118, 12.47067981, -17.16297051, 5.77869057])
        conv = self.backend.runpf(is_dc=True)
        assert conv
        p_or_orig, q_or_orig, *_ = self.backend.lines_or_info()
        assert np.all(q_or_orig == 0.), "in dc mode all q must be zero"
        conv = self.backend.runpf(is_dc=False)
        assert conv
        p_or_orig, q_or_orig, *_ = self.backend.lines_or_info()
        assert self.compare_vect(q_or_orig, true_values_ac)

    def test_get_thermal_limit(self):
        self.skip_if_needed()
        res = self.backend.get_thermal_limit()
        true_values_ac = np.array([42339.01974057, 42339.01974057, 27479652.23546777,
                                   27479652.23546777, 27479652.23546777, 27479652.23546777,
                                   27479652.23546777, 42339.01974057, 42339.01974057,
                                   42339.01974057, 42339.01974057, 42339.01974057,
                                   27479652.23546777, 27479652.23546777, 27479652.23546777,
                                   42339.01974057, 42339.01974057, 42339.01974057,
                                   408269.11892695, 408269.11892695],
                                  dtype=dt_float)
        assert self.compare_vect(res, true_values_ac)

    def test_disconnect_line(self):
        self.skip_if_needed()
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
        self.skip_if_needed()
        conv = self.backend.runpf()
        init_flow = self.backend.get_line_flow()
        init_lp, *_ = self.backend.loads_info()
        init_gp, *_ = self.backend.generators_info()
        init_ls = self.backend.get_line_status()

        action = self.action_env({})  # update the action
        bk_action = self.bkact_class()
        bk_action += action
        self.backend.apply_action(bk_action)
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
        self.skip_if_needed()
        # test that i can modify only the load / prod active values of the powergrid
        # to do that i modify the productions and load all of a factor 0.5 and compare that the DC flows are
        # also multiply by 2

        # i set up the stuff to have exactly 0 losses
        conv = self.backend.runpf(is_dc=True)
        assert conv, "powergrid diverge after loading (even in DC)"
        init_flow, *_ = self.backend.lines_or_info()
        init_lp, init_l_q, *_ = self.backend.loads_info()
        init_gp, *_ = self.backend.generators_info()
        init_ls = self.backend.get_line_status()
        ratio = 1.0
        new_cp = ratio * init_lp
        new_pp = ratio * init_gp * np.sum(init_lp) / np.sum(init_gp)
        action = self.action_env({"injection": {"load_p": new_cp,
                                                "prod_p": new_pp}})  # update the action
        bk_action = self.bkact_class()
        bk_action += action
        self.backend.apply_action(bk_action)
        conv = self.backend.runpf(is_dc=True)
        # now the system has exactly 0 losses (ie sum load = sum gen)

        # i check that if i divide by 2, then everything is divided by 2
        assert conv
        init_flow, *_ = self.backend.lines_or_info()
        init_lp, init_l_q, *_ = self.backend.loads_info()
        init_gp, *_ = self.backend.generators_info()
        init_ls = self.backend.get_line_status()
        ratio = 0.5
        new_cp = ratio * init_lp
        new_pp = ratio * init_gp
        action = self.action_env({"injection": {"load_p": new_cp,
                                                "prod_p": new_pp}})  # update the action
        bk_action = self.bkact_class()
        bk_action += action
        self.backend.apply_action(bk_action)
        conv = self.backend.runpf(is_dc=True)
        assert conv, "Cannot perform a powerflow after doing nothing"

        after_lp, after_lq, *_ = self.backend.loads_info()
        after_gp, *_ = self.backend.generators_info()
        after_ls = self.backend.get_line_status()
        assert self.compare_vect(new_cp, after_lp)  # check i didn't modify the loads

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            p_subs, q_subs, p_bus, q_bus = self.backend.check_kirchoff()
        # i'm in DC mode, i can't check for reactive values...
        assert np.max(np.abs(p_subs)) <= self.tolvect, "problem with active values, at substation"
        assert np.max(np.abs(p_bus.flatten())) <= self.tolvect, "problem with active values, at a bus"

        assert self.compare_vect(new_pp, after_gp)  # check i didn't modify the generators
        assert np.all(init_ls == after_ls)  # check i didn't disconnect any powerlines

        after_flow, *_ = self.backend.lines_or_info()
        assert self.compare_vect(ratio * init_flow, after_flow)  # probably an error with the DC approx

    def test_apply_action_prod_v(self):
        self.skip_if_needed()
        conv = self.backend.runpf(is_dc=False)
        assert conv, "powergrid diverge after loading"
        prod_p_init, prod_q_init, prod_v_init = self.backend.generators_info()
        ratio = 1.05
        action = self.action_env({"injection": {"prod_v": ratio * prod_v_init}})  # update the action
        bk_action = self.bkact_class()
        bk_action += action
        self.backend.apply_action(bk_action)
        conv = self.backend.runpf(is_dc=False)
        assert conv, "Cannot perform a powerflow after modifying the powergrid"

        prod_p_after, prod_q_after, prod_v_after = self.backend.generators_info()
        assert self.compare_vect(ratio * prod_v_init, prod_v_after)  # check i didn't modify the generators

    def test_apply_action_maintenance(self):
        self.skip_if_needed()
        # retrieve some initial data to be sure only a subpart of the _grid is modified
        conv = self.backend.runpf()
        init_lp, *_ = self.backend.loads_info()
        init_gp, *_ = self.backend.generators_info()

        # check that maintenance vector is properly taken into account
        maintenance = np.full((self.backend.n_line,), fill_value=False, dtype=np.bool)
        maintenance[19] = True
        action = self.action_env({"maintenance": maintenance})  # update the action
        bk_action = self.bkact_class()
        bk_action += action

        # apply the action here
        self.backend.apply_action(bk_action)

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
        self.skip_if_needed()
        conv = self.backend.runpf()
        assert conv, "powerflow did not converge at iteration 0"
        init_lp, *_ = self.backend.loads_info()
        init_gp, *_ = self.backend.generators_info()

        # check that maintenance vector is properly taken into account
        maintenance = np.full((self.backend.n_line,), fill_value=False, dtype=np.bool)
        maintenance[17] = True
        action = self.action_env({"hazards": maintenance})  # update the action
        bk_action = self.bkact_class()
        bk_action += action
        # apply the action here
        self.backend.apply_action(bk_action)

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

    def test_apply_action_disconnection(self):
        self.skip_if_needed()
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
        bk_action = self.bkact_class()
        bk_action += action
        # apply the action here
        self.backend.apply_action(bk_action)

        # compute a load flow an performs more tests
        conv = self.backend.runpf()
        assert conv, "Powerflow does not converge if lines {} and {} are removed".format(17, 19)

        # performs basic check
        after_lp, *_ = self.backend.loads_info()
        after_gp, *_ = self.backend.generators_info()
        after_ls = self.backend.get_line_status()
        assert self.compare_vect(init_lp, after_lp)  # check i didn't modify the loads
        # assert self.compare_vect(init_gp, after_gp)  # check i didn't modify the generators # TODO here problem with steady state, P=C+L
        assert np.all(
            disc | maintenance == ~after_ls)  # check i didn't disconnect any powerlines beside the correct one

        flows = self.backend.get_line_status()
        assert np.sum(~flows) == 2
        assert not flows[19]
        assert not flows[17]


class BaseTestTopoAction(MakeBackend):
    def setUp(self):
        self.backend = self.make_backend()
        self.path_matpower = self.get_path()
        self.case_file = self.get_casefile()
        self.backend.load_grid(self.path_matpower, self.case_file)
        self.backend.set_env_name("TestTopoAction_env")
        self.backend.assert_grid_correct()
        self.game_rules = RulesChecker()
        as_class = ActionSpace.init_grid(self.backend)
        self.helper_action = as_class(gridobj=self.backend, legal_action=self.game_rules.legal_action)
        self.bkact_class = _BackendAction.init_grid(self.backend)

    def tearDown(self):
        pass

    def compare_vect(self, pred, true):
        return np.max(np.abs(pred - true)) <= self.tolvect

    def _check_kirchoff(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            p_subs, q_subs, p_bus, q_bus = self.backend.check_kirchoff()
            assert np.max(np.abs(p_subs)) <= self.tolvect, "problem with active values, at substation"
            assert np.max(np.abs(p_bus.flatten())) <= self.tolvect, "problem with active values, at a bus"

        if self.backend.shunts_data_available:
            assert np.max(np.abs(q_subs)) <= self.tolvect, "problem with reactive values, at substation"
            assert np.max(np.abs(q_bus.flatten())) <= self.tolvect, "problem with reaactive values, at a bus"

    def test_get_topo_vect_speed(self):
        # retrieve some initial data to be sure only a subpart of the _grid is modified
        self.skip_if_needed()
        conv = self.backend.runpf()
        init_amps_flow = self.backend.get_line_flow()

        # check that maintenance vector is properly taken into account
        arr = np.array([1, 1, 1, 2, 2, 2], dtype=np.int)
        id_ = 1
        action = self.helper_action({"set_bus": {"substations_id": [(id_, arr)]}})
        bk_action = self.bkact_class()
        bk_action += action

        # apply the action here
        self.backend.apply_action(bk_action)
        conv = self.backend.runpf()
        assert conv
        after_amps_flow = self.backend.get_line_flow()

        topo_vect = self.backend.get_topo_vect()
        topo_vect_old = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=dt_int)
        assert self.compare_vect(topo_vect, topo_vect_old) == True

    def test_topo_set1sub(self):
        # retrieve some initial data to be sure only a subpart of the _grid is modified
        self.skip_if_needed()
        conv = self.backend.runpf()
        init_amps_flow = self.backend.get_line_flow()

        # check that maintenance vector is properly taken into account
        arr = np.array([1, 1, 1, 2, 2, 2], dtype=np.int)
        id_ = 1
        action = self.helper_action({"set_bus": {"substations_id": [(id_, arr)]}})
        bk_action = self.bkact_class()
        bk_action += action

        # apply the action here
        self.backend.apply_action(bk_action)
        conv = self.backend.runpf()
        assert conv
        after_amps_flow = self.backend.get_line_flow()

        topo_vect = self.backend.get_topo_vect()
        assert np.min(topo_vect) == 1, "all buses have been changed"
        assert np.max(topo_vect) == 2, "no buses have been changed"

        # check that the objects have been properly moved
        load_ids = np.where(self.backend.load_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.load_pos_topo_vect[load_ids]] == arr[self.backend.load_to_sub_pos[load_ids]])
        lor_ids = np.where(self.backend.line_or_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.line_or_pos_topo_vect[lor_ids]] == arr[self.backend.line_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.line_ex_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.line_ex_pos_topo_vect[lex_ids]] == arr[self.backend.line_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid == id_)[0]
        assert np.all(topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == arr[self.backend.gen_to_sub_pos[gen_ids]])

        after_amps_flow_th = np.array([6.38865247e+02, 3.81726828e+02, 1.78001287e+04, 2.70742428e+04,
                                       1.06755055e+04, 4.71160165e+03, 1.52265925e+04, 3.37755751e+02,
                                       3.00535519e+02, 5.01164454e-13, 7.01900962e+01, 1.73874580e+02,
                                       2.08904697e+04, 2.11757439e+04, 4.93863382e+04, 1.31935835e+02,
                                       6.99779475e+01, 1.85068609e+02, 7.47283039e+02, 1.14125596e+03])

        after_amps_flow_th = np.array([596.58386539, 342.31364678, 18142.87789987, 27084.37162086,
                                       10155.86483194, 4625.93022957, 15064.92626615, 322.59381855,
                                       273.6977149, 82.21908229, 80.91290202, 206.04740125,
                                       20480.81970337, 21126.22533095, 49275.71520428, 128.04429617,
                                       69.00661266, 188.44754187, 688.1371226, 1132.42521887])
        assert self.compare_vect(after_amps_flow, after_amps_flow_th)
        self._check_kirchoff()

    def test_topo_change1sub(self):
        # check that switching the bus of 3 object is equivalent to set them to bus 2 (as above)
        self.skip_if_needed()
        conv = self.backend.runpf()
        init_amps_flow = self.backend.get_line_flow()

        # check that maintenance vector is properly taken into account
        arr = np.array([False, False, False, True, True, True], dtype=np.bool)
        id_ = 1
        action = self.helper_action({"change_bus": {"substations_id": [(id_, arr)]}})
        bk_action = self.bkact_class()
        bk_action += action

        # apply the action here
        self.backend.apply_action(bk_action)

        # run the powerflow
        conv = self.backend.runpf()
        assert conv
        after_amps_flow = self.backend.get_line_flow()

        topo_vect = self.backend.get_topo_vect()
        assert np.min(topo_vect) == 1, "all buses have been changed"
        assert np.max(topo_vect) == 2, "no buses have been changed"

        # check that the objects have been properly moved
        load_ids = np.where(self.backend.load_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.load_pos_topo_vect[load_ids]] == 1 + arr[self.backend.load_to_sub_pos[load_ids]])
        lor_ids = np.where(self.backend.line_or_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.line_or_pos_topo_vect[lor_ids]] == 1 + arr[self.backend.line_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.line_ex_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.line_ex_pos_topo_vect[lex_ids]] == 1 + arr[self.backend.line_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == 1 + arr[self.backend.gen_to_sub_pos[gen_ids]])

        after_amps_flow_th = np.array([596.58386539, 342.31364678, 18142.87789987, 27084.37162086,
                                       10155.86483194, 4625.93022957, 15064.92626615, 322.59381855,
                                       273.6977149, 82.21908229, 80.91290202, 206.04740125,
                                       20480.81970337, 21126.22533095, 49275.71520428, 128.04429617,
                                       69.00661266, 188.44754187, 688.1371226, 1132.42521887])
        assert self.compare_vect(after_amps_flow, after_amps_flow_th)
        self._check_kirchoff()

    def test_topo_change_1sub_twice(self):
        # check that switching the bus of 3 object is equivalent to set them to bus 2 (as above)
        # and that setting it again is equivalent to doing nothing
        self.skip_if_needed()
        conv = self.backend.runpf()
        init_amps_flow = copy.deepcopy(self.backend.get_line_flow())

        # check that maintenance vector is properly taken into account
        arr = np.array([False, False, False, True, True, True], dtype=np.bool)
        id_ = 1
        action = self.helper_action({"change_bus": {"substations_id": [(id_, arr)]}})
        bk_action = self.bkact_class()
        bk_action += action

        # apply the action here
        self.backend.apply_action(bk_action)
        conv = self.backend.runpf()
        bk_action.reset()
        assert conv
        after_amps_flow = self.backend.get_line_flow()

        topo_vect = self.backend.get_topo_vect()
        assert np.min(topo_vect) == 1, "all buses have been changed"
        assert np.max(topo_vect) == 2, "no buses have been changed"

        # check that the objects have been properly moved
        load_ids = np.where(self.backend.load_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.load_pos_topo_vect[load_ids]] == 1 + arr[self.backend.load_to_sub_pos[load_ids]])
        lor_ids = np.where(self.backend.line_or_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.line_or_pos_topo_vect[lor_ids]] == 1 + arr[self.backend.line_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.line_ex_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.line_ex_pos_topo_vect[lex_ids]] == 1 + arr[self.backend.line_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid == id_)[0]
        assert np.all(
            topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == 1 + arr[self.backend.gen_to_sub_pos[gen_ids]])

        after_amps_flow_th = np.array([596.58386539, 342.31364678, 18142.87789987, 27084.37162086,
                                       10155.86483194, 4625.93022957, 15064.92626615, 322.59381855,
                                       273.6977149, 82.21908229, 80.91290202, 206.04740125,
                                       20480.81970337, 21126.22533095, 49275.71520428, 128.04429617,
                                       69.00661266, 188.44754187, 688.1371226, 1132.42521887])
        assert self.compare_vect(after_amps_flow, after_amps_flow_th)
        self._check_kirchoff()

        action = self.helper_action({"change_bus": {"substations_id": [(id_, arr)]}})
        bk_action += action

        # apply the action here
        self.backend.apply_action(bk_action)
        conv = self.backend.runpf()
        assert conv

        after_amps_flow = self.backend.get_line_flow()
        assert self.compare_vect(after_amps_flow, init_amps_flow)
        topo_vect = self.backend.get_topo_vect()
        assert np.min(topo_vect) == 1
        assert np.max(topo_vect) == 1
        self._check_kirchoff()

    def test_topo_change_2sub(self):
        # check that maintenance vector is properly taken into account
        self.skip_if_needed()
        arr1 = np.array([False, False, False, True, True, True], dtype=np.bool)
        arr2 = np.array([1, 1, 2, 2], dtype=np.int)
        id_1 = 1
        id_2 = 12
        action = self.helper_action({"change_bus": {"substations_id": [(id_1, arr1)]},
                                     "set_bus": {"substations_id": [(id_2, arr2)]}})
        bk_action = self.bkact_class()
        bk_action += action

        # apply the action here
        self.backend.apply_action(bk_action)
        conv = self.backend.runpf()
        assert conv, "powerflow diverge it should not"

        # check the _grid is correct
        topo_vect = self.backend.get_topo_vect()
        assert np.min(topo_vect) == 1, "all buses have been changed"
        assert np.max(topo_vect) == 2, "no buses have been changed"

        # check that the objects have been properly moved
        load_ids = np.where(self.backend.load_to_subid == id_1)[0]
        assert np.all(
            topo_vect[self.backend.load_pos_topo_vect[load_ids]] == 1 + arr1[self.backend.load_to_sub_pos[load_ids]])
        lor_ids = np.where(self.backend.line_or_to_subid == id_1)[0]
        assert np.all(
            topo_vect[self.backend.line_or_pos_topo_vect[lor_ids]] == 1 + arr1[
                self.backend.line_or_to_sub_pos[lor_ids]])
        lex_ids = np.where(self.backend.line_ex_to_subid == id_1)[0]
        assert np.all(
            topo_vect[self.backend.line_ex_pos_topo_vect[lex_ids]] == 1 + arr1[
                self.backend.line_ex_to_sub_pos[lex_ids]])
        gen_ids = np.where(self.backend.gen_to_subid == id_1)[0]
        assert np.all(
            topo_vect[self.backend.gen_pos_topo_vect[gen_ids]] == 1 + arr1[self.backend.gen_to_sub_pos[gen_ids]])

        load_ids = np.where(self.backend.load_to_subid == id_2)[0]
        #TODO check the topology symmetry
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
        after_amps_flow_th = np.array([596.97014348, 342.10559579, 16615.11815357, 31328.50690716,
                                       11832.77202397, 11043.10650167, 11043.10650167, 322.79533908,
                                       273.86501458, 82.34066647, 80.89289074, 208.42396413,
                                       22178.16766548, 27690.51322075, 38684.31540646, 129.44842477,
                                       70.02629553, 185.67687123, 706.77680037, 1155.45754617])
        assert self.compare_vect(after_amps_flow, after_amps_flow_th)
        self._check_kirchoff()


class BaseTestEnvPerformsCorrectCascadingFailures(MakeBackend):
    """
    Test the "next_grid_state" method of the back-end
    """
    def setUp(self):
        self.backend = self.make_backend(detailed_infos_for_cascading_failures=True)
        self.path_matpower = self.get_path()
        self.case_file = self.get_casefile()
        self.backend.load_grid(self.path_matpower, self.case_file)
        self.backend.set_env_name("TestEnvPerformsCorrectCascadingFailures_env")
        self.backend.assert_grid_correct()
        self.game_rules = RulesChecker()
        self.action_env = ActionSpace(gridobj=self.backend, legal_action=self.game_rules.legal_action)

        self.lines_flows_init = np.array([638.28966637, 305.05042301, 17658.9674809, 26534.04334098,
                                          10869.23856329, 4686.71726729, 15612.65903298, 300.07915572,
                                          229.8060832, 169.97292682, 100.40192958, 265.47505664,
                                          21193.86923911, 21216.44452327, 49701.1565287, 124.79684388,
                                          67.59759985, 192.19424706, 666.76961936, 1113.52773632])
        # _parameters for the environment
        self.env_params = Parameters()

        # used for init an env too
        self.chronics_handler = ChronicsHandler()
        self.id_first_line_disco = 8  # due to hard overflow
        self.id_2nd_line_disco = 11  # due to soft overflow

    def tearDown(self):
        pass

    def next_grid_state_no_overflow(self):
        # first i test that, when there is no overflow, i dont do a cascading failure
        env = Environment(init_grid_path=os.path.join(self.path_matpower, self.case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=self.env_params,
                          name="test_pp_env1")

        disco, infos, conv_ = self.backend.next_grid_state(env, is_dc=False)
        assert conv_ is None
        assert not infos

    def test_next_grid_state_1overflow(self):
        # second i test that, when is one line on hard overflow it is disconnected
        self.skip_if_needed()
        case_file = self.case_file
        env_params = copy.deepcopy(self.env_params)
        env_params.HARD_OVERFLOW_THRESHOLD = 1.5
        env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=env_params,
                          name="test_pp_env2")
        self.backend.load_grid(self.path_matpower, case_file)
        self.backend.assert_grid_correct()

        thermal_limit = 10 * self.lines_flows_init
        thermal_limit[self.id_first_line_disco] = self.lines_flows_init[self.id_first_line_disco] / 2
        self.backend.set_thermal_limit(thermal_limit)

        disco, infos, conv_ = self.backend.next_grid_state(env, is_dc=False)
        assert conv_ is None
        assert len(infos) == 1  # check that i have only one overflow
        assert np.sum(disco) == 1

    def test_next_grid_state_1overflow_envNoCF(self):
        # third i test that, if a line is on hard overflow, but i'm on a "no cascading failure" mode,
        # i don't simulate a cascading failure
        self.skip_if_needed()
        self.env_params.NO_OVERFLOW_DISCONNECTION = True
        case_file = self.case_file
        env_params = copy.deepcopy(self.env_params)
        env_params.HARD_OVERFLOW_THRESHOLD = 1.5
        env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=self.env_params,
                          name="test_pp_env3")
        self.backend.load_grid(self.path_matpower, case_file)
        self.backend.assert_grid_correct()
        conv = self.backend.runpf()
        assert conv, "powerflow should converge at loading"
        lines_flows_init = self.backend.get_line_flow()
        thermal_limit = 10 * lines_flows_init
        thermal_limit[self.id_first_line_disco] = lines_flows_init[self.id_first_line_disco] / 2
        self.backend.set_thermal_limit(thermal_limit)

        disco, infos, conv_ = self.backend.next_grid_state(env, is_dc=False)
        assert conv_ is None
        assert not infos  # check that don't simulate a cascading failure
        assert np.sum(disco) == 0

    def test_set_thermal_limit(self):
        thermal_limit = np.arange(self.backend.n_line)
        self.backend.set_thermal_limit(thermal_limit)
        assert np.all(self.backend.thermal_limit_a == thermal_limit)

    def test_nb_timestep_overflow_disc0(self):
        # on this _grid, first line with id 5 is overheated,
        # it is disconnected
        # then powerline 16 have a relative flow of 1.5916318201096937
        # in this scenario i don't have a second line disconnection.
        self.skip_if_needed()
        case_file = self.case_file
        env_params = copy.deepcopy(self.env_params)
        env_params.HARD_OVERFLOW_THRESHOLD = 1.5
        env_params.NB_TIMESTEP_OVERFLOW_ALLOWED = 0
        env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=env_params,
                          name="test_pp_env4")
        self.backend.load_grid(self.path_matpower, case_file)
        self.backend.assert_grid_correct()
        conv = self.backend.runpf()
        assert conv, "powerflow should converge at loading"
        lines_flows_init = self.backend.get_line_flow()

        thermal_limit = 10 * lines_flows_init
        thermal_limit[self.id_first_line_disco] = lines_flows_init[self.id_first_line_disco] / 2
        thermal_limit[self.id_2nd_line_disco] = 400
        self.backend.set_thermal_limit(thermal_limit)

        disco, infos, conv_ = self.backend.next_grid_state(env, is_dc=False)
        assert conv_ is None
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
        self.skip_if_needed()
        case_file = self.case_file
        env_params = copy.deepcopy(self.env_params)
        env_params.HARD_OVERFLOW_THRESHOLD = 1.5
        env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=env_params,
                          name="test_pp_env5")
        self.backend.load_grid(self.path_matpower, case_file)
        self.backend.assert_grid_correct()

        env._timestep_overflow[self.id_2nd_line_disco] = 0
        thermal_limit = 10 * self.lines_flows_init
        thermal_limit[self.id_first_line_disco] = self.lines_flows_init[self.id_first_line_disco] / 2
        thermal_limit[self.id_2nd_line_disco] = 400
        self.backend.set_thermal_limit(thermal_limit)

        disco, infos, conv_ = self.backend.next_grid_state(env, is_dc=False)
        assert conv_ is None
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
        self.skip_if_needed()
        case_file = self.case_file
        env_params = copy.deepcopy(self.env_params)
        env_params.HARD_OVERFLOW_THRESHOLD = 1.5
        env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=env_params,
                          name="test_pp_env6")
        self.backend.load_grid(self.path_matpower, case_file)
        self.backend.assert_grid_correct()

        env._timestep_overflow[self.id_2nd_line_disco] = 1

        thermal_limit = 10 * self.lines_flows_init
        thermal_limit[self.id_first_line_disco] = self.lines_flows_init[self.id_first_line_disco] / 2
        thermal_limit[self.id_2nd_line_disco] = 400
        self.backend.set_thermal_limit(thermal_limit)

        disco, infos, conv_ = self.backend.next_grid_state(env, is_dc=False)
        assert conv_ is None
        assert len(infos) == 1  # check that don't simulate a cascading failure
        assert disco[self.id_first_line_disco]
        assert np.sum(disco) == 1

    def test_nb_timestep_overflow_disc2(self):
        # on this _grid, first line with id 18 is overheated,
        # it is disconnected
        # then powerline 16 have a relative flow of 1.5916318201096937

        # in this scenario I have a second disconnection, because the powerline is allowed to be on overflow for 2
        # timestep and is still on overflow here.
        self.skip_if_needed()
        case_file = self.case_file
        env_params = copy.deepcopy(self.env_params)
        env_params.HARD_OVERFLOW_THRESHOLD = 1.5
        env_params.NB_TIMESTEP_OVERFLOW_ALLOWED = 2
        env = Environment(init_grid_path=os.path.join(self.path_matpower, case_file),
                          backend=self.backend,
                          chronics_handler=self.chronics_handler,
                          parameters=env_params,
                          name="test_pp_env7")
        self.backend.load_grid(self.path_matpower, case_file)
        self.backend.assert_grid_correct()

        env._timestep_overflow[self.id_2nd_line_disco] = 2

        thermal_limit = 10 * self.lines_flows_init
        thermal_limit[self.id_first_line_disco] = self.lines_flows_init[self.id_first_line_disco] / 2
        thermal_limit[self.id_2nd_line_disco] = 400
        self.backend.set_thermal_limit(thermal_limit)

        disco, infos, conv_ = self.backend.next_grid_state(env, is_dc=False)
        assert conv_ is None
        assert len(infos) == 2  # check that there is a cascading failure of length 2
        assert disco[self.id_first_line_disco]
        assert disco[self.id_2nd_line_disco]
        assert np.sum(disco) == 2
        for i, grid_tmp in enumerate(infos):
            assert (not grid_tmp.get_line_status()[self.id_first_line_disco])
            if i == 1:
                assert (not grid_tmp.get_line_status()[self.id_2nd_line_disco])


class BaseTestChangeBusAffectRightBus(MakeBackend):
    def test_set_bus(self):
        self.skip_if_needed()
        # print("test_set_bus")
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make(test=True, backend=backend)
        env.reset()
        action = env.action_space({"set_bus": {"lines_or_id": [(17, 2)]}})
        obs, reward, done, info = env.step(action)
        assert np.all(np.isfinite(obs.v_or))
        assert np.sum(env.backend.get_topo_vect() == 2) == 1
        assert np.all(np.isfinite(obs.to_vect()))

    def test_change_bus(self):
        self.skip_if_needed()
        # print("test_change_bus")
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make(test=True, backend=backend)
        env.reset()
        action = env.action_space({"change_bus": {"lines_or_id": [17]}})
        obs, reward, done, info = env.step(action)
        assert np.all(np.isfinite(obs.v_or))
        assert np.sum(env.backend.get_topo_vect() == 2) == 1

    def test_change_bustwice(self):
        self.skip_if_needed()
        # print("test_change_bustwice")
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make(test=True, backend=backend)
        env.reset()
        action = env.action_space({"change_bus": {"lines_or_id": [17]}})
        obs, reward, done, info = env.step(action)
        assert not done
        assert np.all(np.isfinite(obs.v_or))
        assert np.sum(env.backend.get_topo_vect() == 2) == 1

        action = env.action_space({"change_bus": {"lines_or_id": [17]}})
        obs, reward, done, info = env.step(action)
        assert not done
        assert np.all(np.isfinite(obs.v_or))
        assert np.sum(env.backend.get_topo_vect() == 2) == 0

    def test_isolate_load(self):
        self.skip_if_needed()
        # print("test_isolate_load")
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make(test=True, backend=backend)
        act = env.action_space({"set_bus": {"loads_id": [(0, 2)]}})
        obs, reward, done, info = env.step(act)
        assert done, "an isolated load has not lead to a game over"

    def test_reco_disco_bus(self):
        self.skip_if_needed()
        # print("test_reco_disco_bus")
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_case1 = make("rte_case5_example", test=True, gamerules_class=AlwaysLegal, backend=backend)
        obs = env_case1.reset()  # reset is good
        act = env_case1.action_space.disconnect_powerline(line_id=5)  # I disconnect a powerline
        obs, reward, done, info = env_case1.step(act)  # do the action, it's valid
        act_case1 = env_case1.action_space.reconnect_powerline(line_id=5, bus_or=2,
                                                               bus_ex=2)  # reconnect powerline on bus 2 both ends
        # this should lead to a game over a the powerline is out of the grid, 2 buses are, but without anything
        # this is a non connex grid
        obs_case1, reward_case1, done_case1, info_case1 = env_case1.step(act_case1)
        assert done_case1

    def test_reco_disco_bus2(self):
        self.skip_if_needed()
        # print("test_reco_disco_bus2")
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_case2 = make("rte_case5_example", test=True, gamerules_class=AlwaysLegal, backend=backend)
        obs = env_case2.reset()  # reset is good
        obs, reward, done, info = env_case2.step(env_case2.action_space())  # do the action, it's valid
        act_case2 = env_case2.action_space.reconnect_powerline(line_id=5, bus_or=2,
                                                               bus_ex=2)  # reconnect powerline on bus 2 both ends
        # this should lead to a game over a the powerline is out of the grid, 2 buses are, but without anything
        # this is a non connex grid
        obs_case2, reward_case2, done_case2, info_case2 = env_case2.step(act_case2)
        # this was illegal before, but test it is still illegal
        assert done_case2

    def test_reco_disco_bus3(self):
        self.skip_if_needed()
        # print("test_reco_disco_bus3")
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_case2 = make("rte_case5_example", test=True, gamerules_class=AlwaysLegal, backend=backend)
        obs = env_case2.reset()  # reset is good
        obs, reward, done, info = env_case2.step(env_case2.action_space())  # do the action, it's valid
        act_case2 = env_case2.action_space.reconnect_powerline(line_id=5, bus_or=1,
                                                               bus_ex=2)  # reconnect powerline on bus 2 both ends
        # this should not lead to a game over this time, the grid is connex!
        obs_case2, reward_case2, done_case2, info_case2 = env_case2.step(act_case2)
        assert done_case2 is False

    def test_reco_disco_bus4(self):
        self.skip_if_needed()
        # print("test_reco_disco_bus4")
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_case2 = make("rte_case5_example", test=True, gamerules_class=AlwaysLegal, backend=backend)
        obs = env_case2.reset()  # reset is good
        obs, reward, done, info = env_case2.step(env_case2.action_space())  # do the action, it's valid
        act_case2 = env_case2.action_space.reconnect_powerline(line_id=5, bus_or=2,
                                                               bus_ex=1)  # reconnect powerline on bus 2 both ends
        # this should not lead to a game over this time, the grid is connex!
        obs_case2, reward_case2, done_case2, info_case2 = env_case2.step(act_case2)
        assert done_case2 is False

    def test_reco_disco_bus5(self):
        self.skip_if_needed()
        # print("test_reco_disco_bus5")
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_case2 = make("rte_case5_example", test=True, gamerules_class=AlwaysLegal, backend=backend)
        obs = env_case2.reset()  # reset is good
        act_case2 = env_case2.action_space(
            {"set_bus": {"lines_or_id": [(5, 2)], "lines_ex_id": [(5, 2)]}})  # reconnect powerline on bus 2 both ends
        # this should not lead to a game over this time, the grid is connex!
        obs_case2, reward_case2, done_case2, info_case2 = env_case2.step(act_case2)
        assert done_case2


class BaseTestShuntAction(MakeBackend):
    def test_shunt_ambiguous_id_incorrect(self):
        self.skip_if_needed()
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True, gamerules_class=AlwaysLegal,
                      action_class=CompleteAction, backend=backend) as env_case2:
                with self.assertRaises(AmbiguousAction):
                    act = env_case2.action_space({"shunt": {"set_bus": [(0, 2)]}})

    def test_shunt_effect(self):
        self.skip_if_needed()
        backend1 = self.make_backend()
        backend2 = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_ref = make("rte_case14_realistic", test=True, gamerules_class=AlwaysLegal,
                           action_class=CompleteAction, backend=backend1)
            env_change_q = make("rte_case14_realistic", test=True, gamerules_class=AlwaysLegal,
                                action_class=CompleteAction, backend=backend2)

        obs_ref, *_ = env_ref.step(env_ref.action_space())
        obs_change_p_down, *_ = env_change_q.step(env_change_q.action_space({"shunt": {"shunt_q": [(0, -30)]}}))
        assert obs_ref.v_or[10] < obs_change_p_down.v_or[10] - self.tol_one
        obs_change_p_up, *_ = env_change_q.step(env_change_q.action_space({"shunt": {"shunt_q": [(0, +30)]}}))
        obs_ref, *_ = env_ref.step(env_ref.action_space())
        assert obs_ref.v_or[10] > obs_change_p_up.v_or[10] + self.tol_one
        obs_disco_sh, *_ = env_change_q.step(env_change_q.action_space({"shunt": {"set_bus": [(0, -1)]}}))
        # given the shunt amount at first, this is the right test to do
        assert obs_ref.v_or[10] > obs_disco_sh.v_or[10] + self.tol_one


class BaseTestResetEqualsLoadGrid(MakeBackend):
    def setUp(self):
        backend1 = self.make_backend()
        backend2 = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env1 = make("rte_case5_example", test=True, backend=backend1)
            self.backend1 = self.env1.backend
            self.env2 = make("rte_case5_example", test=True, backend=backend2)
            self.backend2 = self.env2.backend
        np.random.seed(69)

    def tearDown(self):
        self.env1.close()
        self.env2.close()

    def test_reset_equals_reset(self):
        self.skip_if_needed()
        # Reset backend1 with reset
        self.env1.reset()
        # Reset backend2 with reset
        self.env2.reset()
        self._compare_backends()

    def _compare_backends(self):
        # Compare
        if hasattr(self.backend1, "prod_pu_to_kv") and hasattr(self.backend2, "prod_pu_to_kv"):
            assert np.all(self.backend1.prod_pu_to_kv == self.backend2.prod_pu_to_kv)
        if hasattr(self.backend1, "load_pu_to_kv") and hasattr(self.backend2, "load_pu_to_kv"):
            assert np.all(self.backend1.load_pu_to_kv == self.backend2.load_pu_to_kv)
        if hasattr(self.backend1, "lines_or_pu_to_kv") and hasattr(self.backend2, "lines_or_pu_to_kv"):
            assert np.all(self.backend1.lines_or_pu_to_kv == self.backend2.lines_or_pu_to_kv)
        if hasattr(self.backend1, "lines_ex_pu_to_kv") and hasattr(self.backend2, "lines_ex_pu_to_kv"):
            assert np.all(self.backend1.lines_ex_pu_to_kv == self.backend2.lines_ex_pu_to_kv)
        if hasattr(self.backend1, "p_or") and hasattr(self.backend2, "p_or"):
            assert np.all(self.backend1.p_or == self.backend2.p_or)
        if hasattr(self.backend1, "q_or") and hasattr(self.backend2, "q_or"):
            assert np.all(self.backend1.q_or == self.backend2.q_or)
        if hasattr(self.backend1, "v_or") and hasattr(self.backend2, "v_or"):
            assert np.all(self.backend1.v_or == self.backend2.v_or)
        if hasattr(self.backend1, "a_or") and hasattr(self.backend2, "a_or"):
            assert np.all(self.backend1.a_or == self.backend2.a_or)
        if hasattr(self.backend1, "p_ex") and hasattr(self.backend2, "p_ex"):
            assert np.all(self.backend1.p_ex == self.backend2.p_ex)
        if hasattr(self.backend1, "a_ex") and hasattr(self.backend2, "a_ex"):
            assert np.all(self.backend1.a_ex == self.backend2.a_ex)
        if hasattr(self.backend1, "v_ex") and hasattr(self.backend2, "v_ex"):
            assert np.all(self.backend1.v_ex == self.backend2.v_ex)

    def test_reset_equals_load_grid(self):
        self.skip_if_needed()
        # Reset backend1 with reset
        self.env1.reset()
        # Reset backend2 with load_grid
        self.backend2.reset = self.backend2.load_grid
        self.env2.reset()

        # Compare
        self._compare_backends()

    def test_load_grid_equals_load_grid(self):
        self.skip_if_needed()
        # Reset backend1 with load_grid
        self.backend1.reset = self.backend1.load_grid
        self.env1.reset()
        # Reset backend2 with load_grid
        self.backend2.reset = self.backend2.load_grid
        self.env2.reset()

        # Compare
        self._compare_backends()

    def test_obs_from_same_chronic(self):
        self.skip_if_needed()
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
        assert np.all(obs1.time_next_maintenance == obs2.time_next_maintenance)
        assert np.all(obs1.duration_next_maintenance == obs2.duration_next_maintenance)
        assert np.all(obs1.target_dispatch == obs2.target_dispatch)
        assert np.all(obs1.actual_dispatch == obs2.actual_dispatch)

    def test_combined_changes(self):
        # Unlimited sub changes
        backend = self.make_backend()
        params = grid2op.Parameters.Parameters()
        params.MAX_SUB_CHANGED = 999

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_realistic",
                               test=True,
                               backend=backend,
                               param=params)

        # Find N valid iadd combination of R change actions
        acts = self.aux_random_topos_act(env, n=16, r=3)
        # Pick one at random
        act = np.random.choice(acts)

        # Reset env
        obs = env.reset()
        # At t=0 everything is on bus 1 normally
        assert np.all(obs.topo_vect == 1)

        # Step
        obs, _, done, _ = env.step(act)
        # This should use valid actions
        assert done == False
        # At t=1, unchanged elements should be on bus 1
        assert np.all(obs.topo_vect[~act._change_bus_vect] == 1)

    def aux_nth_combination(self, iterable, r, index):
        'Equivalent to list(combinations(iterable, r))[index]'
        pool = tuple(iterable)
        n = len(pool)
        if r < 0 or r > n:
            raise ValueError
        c = 1
        k = min(r, n-r)
        for i in range(1, k+1):
            c = c * (n - k + i) // i
        if index < 0:
            index += c
        if index < 0 or index >= c:
            raise IndexError
        result = []
        while r:
            c, n, r = c*r//n, n-1, r-1
            while index >= c:
                index -= c
                c, n = c*(n-r)//n, n-1
            result.append(pool[-1-n])
        return tuple(result)

    def aux_random_topos_act(self, env, n=128, r=2):
        actsp = env.action_space
        acts = actsp.get_all_unitary_topologies_change(actsp)
        res = []
        n_comb = comb(len(acts), r)
        while len(res) < n:
            env.reset()
            rnd_idx = np.random.randint(n_comb)
            a = self.aux_nth_combination(acts, r, rnd_idx)
            atest = env.action_space({})
            for atmp in a:
                atest += atmp
            _, _, done, _ = env.step(atest)
            if not done:
                res.append(copy.deepcopy(atest))
        return res


class BaseTestVoltageOWhenDisco(MakeBackend):
    def test_this(self):
        self.skip_if_needed()
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case14_realistic", test=True, backend=backend) as env:
                line_id = 1
                act = env.action_space({"set_line_status": [(line_id, -1)]})
                obs, *_ = env.step(act)
                assert obs.v_or[line_id] == 0.  # is not 0 however line is not connected


class BaseTestChangeBusSlack(MakeBackend):
    def test_change_slack_case14(self):
        self.skip_if_needed()
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_realistic", test=True, backend=backend)
        action = env.action_space({"set_bus": {"generators_id": [(-1, 2)], "lines_or_id": [(0, 2)]}})
        obs, reward, am_i_done, info = env.step(action)
        assert am_i_done is False
        assert np.all(obs.prod_p >= 0.)
        assert np.sum(obs.prod_p) >= np.sum(obs.load_p)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            p_subs, q_subs, p_bus, q_bus = env.backend.check_kirchoff()
        assert np.all(np.abs(p_subs) <= self.tol_one)
        assert np.all(np.abs(p_bus) <= self.tol_one)


class BaseIssuesTest(MakeBackend):
    def test_issue_125(self):
        # https://github.com/rte-france/Grid2Op/issues/125
        self.skip_if_needed()
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_realistic", test=True,
                               backend=backend)
        action = env.action_space({"set_bus": {"loads_id": [(1, -1)]}})
        obs, reward, am_i_done, info = env.step(action)
        assert info["is_illegal"] is False
        assert info["is_ambiguous"] is False
        assert len(info["exception"])
        assert am_i_done

        env.reset()
        action = env.action_space({"set_bus": {"generators_id": [(1, -1)]}})
        obs, reward, am_i_done, info = env.step(action)
        assert info["is_illegal"] is False
        assert info["is_ambiguous"] is False
        assert len(info["exception"])
        assert am_i_done

    def test_issue_134(self):
        self.skip_if_needed()
        backend = self.make_backend()
        param = Parameters()

        param.NB_TIMESTEP_COOLDOWN_LINE = 0
        param.NB_TIMESTEP_COOLDOWN_SUB = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_realistic",
                               test=True,
                               backend=backend,
                               param=param)
        LINE_ID = 2

        # Disconnect ex
        action = env.action_space({
            'set_bus': {
                "lines_or_id": [(LINE_ID, 0)],
                "lines_ex_id": [(LINE_ID, -1)],
            }
        })
        obs, reward, done, info = env.step(action)
        assert obs.line_status[LINE_ID] == False
        assert obs.topo_vect[obs.line_or_pos_topo_vect[LINE_ID]] == -1
        assert obs.topo_vect[obs.line_ex_pos_topo_vect[LINE_ID]] == -1
        
        # Reconnect ex on bus 2
        action = env.action_space({
            'set_bus': {
                "lines_or_id": [(LINE_ID, 0)],
                "lines_ex_id": [(LINE_ID, 2)],
            }
        })
        obs, reward, done, info = env.step(action)
        assert obs.line_status[LINE_ID] == True
        assert obs.topo_vect[obs.line_or_pos_topo_vect[LINE_ID]] == 1
        assert obs.topo_vect[obs.line_ex_pos_topo_vect[LINE_ID]] == 2
    
        # Disconnect or
        action = env.action_space({
            'set_bus': {
                "lines_or_id": [(LINE_ID, -1)],
                "lines_ex_id": [(LINE_ID, 0)],
            }
        })
        obs, reward, done, info = env.step(action)
        assert obs.line_status[LINE_ID] == False
        assert obs.topo_vect[obs.line_or_pos_topo_vect[LINE_ID]] == -1
        assert obs.topo_vect[obs.line_ex_pos_topo_vect[LINE_ID]] == -1

        # Reconnect or on bus 1
        action = env.action_space({
            'set_bus': {
                "lines_or_id": [(LINE_ID, 1)],
                "lines_ex_id": [(LINE_ID, 0)],
            }
        })
        obs, reward, done, info = env.step(action)
        assert obs.line_status[LINE_ID] == True
        assert obs.topo_vect[obs.line_or_pos_topo_vect[LINE_ID]] == 1
        assert obs.topo_vect[obs.line_ex_pos_topo_vect[LINE_ID]] == 2

    def test_issue_134_check_ambiguity(self):
        self.skip_if_needed()
        backend = self.make_backend()
        param = Parameters()

        param.MAX_LINE_STATUS_CHANGED = 9999
        param.MAX_SUB_CHANGED = 99999
        param.NB_TIMESTEP_COOLDOWN_LINE = 0
        param.NB_TIMESTEP_COOLDOWN_SUB = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_realistic",
                               test=True,
                               backend=backend,
                               param=param)
        LINE_ID = 2

        # Reconnect or on bus 1 disconnect on bus ex -> this should be ambiguous
        action = env.action_space({
            'set_bus': {
                "lines_or_id": [(LINE_ID, 1)],
                "lines_ex_id": [(LINE_ID, -1)],
            }
        })
        obs, reward, done, info = env.step(action)
        assert info["is_ambiguous"] == True

    def test_issue_134_withcooldown_forrules(self):
        self.skip_if_needed()
        backend = self.make_backend()
        param = Parameters()

        param.NB_TIMESTEP_COOLDOWN_LINE = 20
        param.NB_TIMESTEP_COOLDOWN_SUB = 2
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_realistic",
                               test=True,
                               backend=backend,
                               param=param)
        LINE_ID = 2

        # Disconnect ex -> this is an action on the powerline
        for (or_, ex_) in [(0, -1), (-1, 0)]:
            obs = env.reset()
            action = env.action_space({
                'set_bus': {
                    "lines_or_id": [(LINE_ID, or_)],
                    "lines_ex_id": [(LINE_ID, ex_)],
                }
            })
            # i disconnect a powerline, i should not act on the substation but on the line LINE_ID
            obs, reward, done, info = env.step(action)
            assert np.all(obs.time_before_cooldown_sub == 0)
            assert obs.time_before_cooldown_line[LINE_ID] == param.NB_TIMESTEP_COOLDOWN_LINE
            assert obs.line_status[LINE_ID] == False
            assert obs.topo_vect[obs.line_or_pos_topo_vect[LINE_ID]] == -1
            assert obs.topo_vect[obs.line_ex_pos_topo_vect[LINE_ID]] == -1

            # i try to reconnect it, should not be possible whether i do it from
            # setting a bus at one extremity or playing with the status
            obs, *_ = env.step(env.action_space({"set_line_status": [(LINE_ID, 1)]}))
            assert obs.line_status[LINE_ID] == False
            assert np.all(obs.time_before_cooldown_sub == 0)
            assert obs.time_before_cooldown_line[LINE_ID] == param.NB_TIMESTEP_COOLDOWN_LINE - 1

            obs, *_ = env.step(env.action_space({"change_line_status": [LINE_ID]}))
            assert obs.line_status[LINE_ID] == False
            assert np.all(obs.time_before_cooldown_sub == 0)
            assert obs.time_before_cooldown_line[LINE_ID] == param.NB_TIMESTEP_COOLDOWN_LINE - 2

            obs, *_ = env.step(env.action_space({
                'set_bus': {
                    "lines_or_id": [(LINE_ID, 0)],
                    "lines_ex_id": [(LINE_ID, 1)],
                }
            }))
            assert obs.line_status[LINE_ID] == False
            assert np.all(obs.time_before_cooldown_sub == 0)
            assert obs.time_before_cooldown_line[LINE_ID] == param.NB_TIMESTEP_COOLDOWN_LINE - 3

            obs, *_ = env.step(env.action_space({
                'set_bus': {
                    "lines_or_id": [(LINE_ID, 1)],
                    "lines_ex_id": [(LINE_ID, 0)],
                }
            }))
            assert obs.line_status[LINE_ID] == False
            assert np.all(obs.time_before_cooldown_sub == 0)
            assert obs.time_before_cooldown_line[LINE_ID] == param.NB_TIMESTEP_COOLDOWN_LINE - 4

            # i wait enough for the cooldown to pass
            for _ in range(param.NB_TIMESTEP_COOLDOWN_LINE - 4):
                obs, *_ = env.step(env.action_space())
            assert np.all(obs.time_before_cooldown_sub == 0)

            # and now i try to reconnect, this should not affect the substation but the cooldown on the line
            obs, *_ = env.step(env.action_space({
                'set_bus': {
                    "lines_or_id": [(LINE_ID, -2 * or_)],
                    "lines_ex_id": [(LINE_ID, -2 * ex_)],
                }
            }))
            assert obs.line_status[LINE_ID] == True
            assert np.all(obs.time_before_cooldown_sub == 0)
            assert obs.time_before_cooldown_line[LINE_ID] == param.NB_TIMESTEP_COOLDOWN_LINE

            # and now i try to modify the buses at one end of the powerline,
            # this should affect the substation and NOT the line (so be possible)
            obs, *_ = env.step(env.action_space({
                'set_bus': {
                    "lines_or_id": [(LINE_ID, -1 * or_)],
                    "lines_ex_id": [(LINE_ID, -1 * ex_)],
                }
            }))
            assert obs.line_status[LINE_ID] == True
            if or_ != 0:
                assert obs.time_before_cooldown_sub[obs.line_or_to_subid[LINE_ID]] == param.NB_TIMESTEP_COOLDOWN_SUB
            else:
                assert obs.time_before_cooldown_sub[obs.line_ex_to_subid[LINE_ID]] == param.NB_TIMESTEP_COOLDOWN_SUB
            assert obs.time_before_cooldown_line[LINE_ID] == param.NB_TIMESTEP_COOLDOWN_LINE - 1

    def test_issue_copyenv(self):
        # https://github.com/BDonnot/lightsim2grid/issues/10
        from grid2op import make
        backend = self.make_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env1 = grid2op.make("rte_case14_realistic",
                                test=True,
                                backend=backend)
        env2 = env1.copy()
        obs1 = env1.reset()
        obs2 = env2.get_obs()
        assert np.any(obs1.prod_p != obs2.prod_p)


class BaseStatusActions(MakeBackend):
    def _make_my_env(self):
        backend = self.make_backend()
        param = Parameters()
        param.NB_TIMESTEP_COOLDOWN_LINE = 0
        param.NB_TIMESTEP_COOLDOWN_SUB = 0
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_realistic",
                               test=True,
                               backend=backend,
                               param=param
                               )
        return env

    def _init_disco_or_not(self, LINE_ID, env, disco_before_the_action):
        if not disco_before_the_action:
            # powerline is supposed to be connected before the action takes place
            statuses = env.get_obs().line_status
        else:
            # i disconnect it
            action = env.action_space({"set_line_status": [(LINE_ID, -1)]})
            obs, reward, done, info = env.step(action)
            statuses = obs.line_status
        return statuses

    def _line_connected(self, LINE_ID, obs, busor=1):
        assert obs.line_status[LINE_ID]
        assert obs.topo_vect[obs.line_or_pos_topo_vect[LINE_ID]] == busor
        assert obs.topo_vect[obs.line_ex_pos_topo_vect[LINE_ID]] == 1

    def _line_disconnected(self, LINE_ID, obs):
        assert not obs.line_status[LINE_ID]
        assert obs.topo_vect[obs.line_or_pos_topo_vect[LINE_ID]] == -1
        assert obs.topo_vect[obs.line_ex_pos_topo_vect[LINE_ID]] == -1

    def _only_line_impacted(self, LINE_ID, action, statuses):
        lines_impacted, subs_impacted = action.get_topological_impact(statuses)
        assert np.sum(subs_impacted) == 0
        assert np.sum(lines_impacted) == 1 and lines_impacted[LINE_ID]

    def _only_sub_impacted(self, LINE_ID, action, statuses):
        lines_impacted, subs_impacted = action.get_topological_impact(statuses)
        assert np.sum(subs_impacted) == 1 and subs_impacted[action.line_or_to_subid[LINE_ID]]
        assert np.sum(lines_impacted) == 0

    def test_setmin1_prevConn(self):
        """{"set_line_status": [(LINE_ID, -1)]} when connected"""
        self.skip_if_needed()
        env = self._make_my_env()
        LINE_ID = 1

        # set the grid to right configuration
        statuses = self._init_disco_or_not(LINE_ID, env, disco_before_the_action=False)

        action = env.action_space({"set_line_status": [(LINE_ID, -1)]})
        obs, reward, done, info = env.step(action)

        # right consequences
        self._line_disconnected(LINE_ID, obs)

        # right way to count it
        self._only_line_impacted(LINE_ID, action, statuses)

    def test_set1_prevConn(self):
        """{"set_line_status": [(LINE_ID, +1)]} when connected"""
        self.skip_if_needed()
        env = self._make_my_env()
        LINE_ID = 1

        # set the grid to right configuration
        statuses = self._init_disco_or_not(LINE_ID, env, disco_before_the_action=False)

        action = env.action_space({"set_line_status": [(LINE_ID, +1)]})
        obs, reward, done, info = env.step(action)

        # right consequences
        self._line_connected(LINE_ID, obs)


        # right way to count it
        self._only_line_impacted(LINE_ID, action, statuses)

    def test_setmin1_prevDisc(self):
        """{"set_line_status": [(LINE_ID, -1)]} when disconnected"""
        self.skip_if_needed()
        env = self._make_my_env()
        LINE_ID = 1

        # set the grid to right configuration
        statuses = self._init_disco_or_not(LINE_ID, env, disco_before_the_action=True)

        # and now i test the impact of the action
        action = env.action_space({"set_line_status": [(LINE_ID, -1)]})
        obs, reward, done, info = env.step(action)

        # right consequences
        self._line_disconnected(LINE_ID, obs)

        # right way to count it
        self._only_line_impacted(LINE_ID, action, statuses)

    def test_set1_prevDisc(self):
        """{"set_line_status": [(LINE_ID, +1)]} when disconnected"""
        self.skip_if_needed()
        env = self._make_my_env()
        LINE_ID = 1

        # set the grid to right configuration
        statuses = self._init_disco_or_not(LINE_ID, env, disco_before_the_action=True)

        # and now i test the impact of the action
        action = env.action_space({"set_line_status": [(LINE_ID, +1)]})
        obs, reward, done, info = env.step(action)

        # right consequences
        self._line_connected(LINE_ID, obs)

        # right way to count it
        self._only_line_impacted(LINE_ID, action, statuses)

    def test_chgt_prevConn(self):
        """{"change_line_status": [LINE_ID]} when connected"""
        self.skip_if_needed()
        env = self._make_my_env()
        LINE_ID = 1

        # set the grid to right configuration
        statuses = self._init_disco_or_not(LINE_ID, env, disco_before_the_action=False)

        # and now i test the impact of the action
        action = env.action_space({"change_line_status": [LINE_ID]})
        obs, reward, done, info = env.step(action)

        # right consequences
        self._line_disconnected(LINE_ID, obs)

        # right way to count it
        self._only_line_impacted(LINE_ID, action, statuses)

    def test_chgt_prevDisc(self):
        """{"change_line_status": [LINE_ID]} when disconnected"""
        self.skip_if_needed()
        env = self._make_my_env()
        LINE_ID = 1

        # set the grid to right configuration
        statuses = self._init_disco_or_not(LINE_ID, env, disco_before_the_action=True)

        # and now i test the impact of the action
        action = env.action_space({"change_line_status": [LINE_ID]})
        obs, reward, done, info = env.step(action)

        # right consequences
        self._line_connected(LINE_ID, obs)

        # right way to count it
        self._only_line_impacted(LINE_ID, action, statuses)

    def test_setbusmin1_prevConn(self):
        """{"set_bus": {"lines_or_id": [(LINE_ID, -1)]}} when connected"""
        self.skip_if_needed()
        env = self._make_my_env()
        LINE_ID = 1

        # set the grid to right configuration
        statuses = self._init_disco_or_not(LINE_ID, env, disco_before_the_action=False)

        # and now i test the impact of the action
        action = env.action_space({"set_bus": {"lines_or_id": [(LINE_ID, -1)]}} )
        obs, reward, done, info = env.step(action)

        # right consequences
        self._line_disconnected(LINE_ID, obs)

        # right way to count it
        self._only_line_impacted(LINE_ID, action, statuses)

    def test_setbusmin1_prevDisc(self):
        """{"set_bus": {"lines_or_id": [(LINE_ID, -1)]}} when disco"""
        self.skip_if_needed()
        env = self._make_my_env()
        LINE_ID = 1

        # set the grid to right configuration
        statuses = self._init_disco_or_not(LINE_ID, env, disco_before_the_action=True)

        # and now i test the impact of the action
        action = env.action_space({"set_bus": {"lines_or_id": [(LINE_ID, -1)]}} )
        obs, reward, done, info = env.step(action)

        # right consequences
        self._line_disconnected(LINE_ID, obs)

        # right way to count it
        self._only_sub_impacted(LINE_ID, action, statuses)

    def test_setbus2_prevConn(self):
        """{"set_bus": {"lines_or_id": [(LINE_ID, 2)]}} when connected"""
        self.skip_if_needed()
        env = self._make_my_env()
        LINE_ID = 1

        # set the grid to right configuration
        statuses = self._init_disco_or_not(LINE_ID, env, disco_before_the_action=False)

        # and now i test the impact of the action
        action = env.action_space({"set_bus": {"lines_or_id": [(LINE_ID, 2)]}} )
        obs, reward, done, info = env.step(action)

        # right consequences
        self._line_connected(LINE_ID, obs, busor=2)

        # right way to count it
        self._only_sub_impacted(LINE_ID, action, statuses)

    def test_setbus2_prevDisc(self):
        """{"set_bus": {"lines_or_id": [(LINE_ID, 2)]}} when disconnected"""
        self.skip_if_needed()
        env = self._make_my_env()
        LINE_ID = 1

        # set the grid to right configuration
        statuses = self._init_disco_or_not(LINE_ID, env, disco_before_the_action=True)

        # and now i test the impact of the action
        action = env.action_space({"set_bus": {"lines_or_id": [(LINE_ID, 2)]}})
        obs, reward, done, info = env.step(action)

        # right consequences
        self._line_connected(LINE_ID, obs, busor=2)

        # right way to count it
        self._only_line_impacted(LINE_ID, action, statuses)

    def test_chgtbus_prevConn(self):
        """{"change_bus": {"lines_or_id": [LINE_ID]}}  when connected"""
        self.skip_if_needed()
        env = self._make_my_env()
        LINE_ID = 1

        # set the grid to right configuration
        statuses = self._init_disco_or_not(LINE_ID, env, disco_before_the_action=False)

        # and now i test the impact of the action
        action = env.action_space({"change_bus": {"lines_or_id": [LINE_ID]}})
        obs, reward, done, info = env.step(action)

        # right consequences
        self._line_connected(LINE_ID, obs, busor=2)

        # right way to count it
        self._only_sub_impacted(LINE_ID, action, statuses)

    def test_chgtbus_prevDisc(self):
        """{"change_bus": {"lines_or_id": [LINE_ID]}}  when discconnected"""
        self.skip_if_needed()
        env = self._make_my_env()
        LINE_ID = 1

        # set the grid to right configuration
        statuses = self._init_disco_or_not(LINE_ID, env, disco_before_the_action=True)

        # and now i test the impact of the action
        action = env.action_space({"change_bus": {"lines_or_id": [LINE_ID]}})
        obs, reward, done, info = env.step(action)

        # right consequences
        self._line_disconnected(LINE_ID, obs)

        # right way to count it
        self._only_sub_impacted(LINE_ID, action, statuses)