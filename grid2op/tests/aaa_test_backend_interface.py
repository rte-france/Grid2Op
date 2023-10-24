# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import numpy as np
import warnings
from grid2op.tests.helper_path_test import HelperTests, MakeBackend, PATH_DATA
from grid2op.Exceptions import BackendError


class AAATestBackendAPI(MakeBackend):
    # def make_backend(self, detailed_infos_for_cascading_failures=False):
    #     return PandaPowerBackend()  # TODO REMOVE
    #     # from lightsim2grid import LightSimBackend
    #     # return LightSimBackend()  # TODO REMOVE
    
    
    # if same ordering as pandapower
    # init_load_p = np.array([21.7, 94.2, 47.8,  7.6, 11.2, 29.5,  9. ,  3.5,  6.1, 13.5, 14.9])
    # init_load_q = np.array([12.7, 19. , -3.9,  1.6,  7.5, 16.6,  5.8,  1.8,  1.6,  5.8,  5. ])
    # init_gen_p = np.array([  40.,    0.,    0.,    0.,    0., 219.])
    # init_gen_v = np.array([144.21, 139.38,  21.4 ,  21.4 ,  13.08, 146.28])
                        
    def get_path(self):
        return os.path.join(PATH_DATA, "educ_case14_storage")
    
    def get_casefile(self):
        return "grid.json"
    
    def aux_get_env_name(self):
        """do not run nor modify ! (used for this test class only)"""
        return "BasicTest_load_grid_" + type(self).__name__

    def aux_make_backend(self):
        """do not run nor modify ! (used for this test class only)"""
        backend = self.make_backend()
        backend.load_grid(self.get_path(), self.get_casefile())
        backend.load_redispacthing_data("tmp")  # pretend there is no generator
        backend.load_storage_data(self.get_path())
        env_name = self.aux_get_env_name()
        backend.env_name = env_name
        backend.assert_grid_correct()  
        return backend
    
    def test_00create_backend(self):
        """Tests the backend can be created (not integrated in a grid2op environment yet)"""
        self.skip_if_needed()
        backend = self.make_backend()
    
    def test_01load_grid(self):
        """Tests the grid can be loaded (supposes that your backend can read the grid.json in educ_case14_storage)*
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        
        .. danger::
            This test will NOT pass if the grid is not the "educ_case14_storage" file.
        
        """
        self.skip_if_needed()
        backend = self.make_backend()
        backend.load_grid(self.get_path(), self.get_casefile())  # both argument filled
        backend.load_redispacthing_data(self.get_path())
        backend.load_storage_data(self.get_path())
        env_name = "BasicTest_load_grid0_" + type(self).__name__
        backend.env_name = env_name
        backend.assert_grid_correct() 
        cls = type(backend)
        assert cls.n_line == 20, f"there should be 20 lines / trafos on the grid (if you used the pandapower default grid), found {cls.n_line} (remember trafo are conted grid2op side as powerline)"
        assert cls.n_gen == 6, f"there should be 6 generators on the grid (if you used the pandapower default grid) found {cls.n_gen} (remember a generator is added to the slack if none are present)"
        assert cls.n_load == 11, f"there should be 11 loads on the grid (if you used the pandapower default grid), found {cls.n_load}"
        assert cls.n_sub == 14, f"there should be 14 substations on this grid (if you used the pandapower default grid), found {cls.n_sub}"
        if cls.shunts_data_available:
            assert cls.n_shunt == 1, f"there should be 1 shunt on the grid (if you used the pandapower default grid), found {cls.n_shunt}"
        if cls.n_storage > 0:
            assert cls.n_storage == 2, f"there should be 2 storage units on this grid (if you used the pandapower default grid), found {cls.n_storage}"
        assert env_name in cls.env_name, f"you probably should not have overidden the assert_grid_correct function !"
        backend.close()
        
        backend = self.make_backend()
        backend.load_grid(os.path.join(self.get_path(), self.get_casefile()))  # first argument filled, second None
        backend.load_redispacthing_data(self.get_path())
        backend.load_storage_data(self.get_path())
        backend.env_name = "BasicTest_load_grid2_" + type(self).__name__
        backend.assert_grid_correct() 
        backend.close()
        
        backend = self.make_backend()
        with self.assertRaises(Exception):
            backend.load_grid()  # should raise if nothing is loaded 

    def test_02modify_load(self):
        """Tests the loads can be modified        

        This test supposes that :
        
        - backend.load_grid(...) is implemented        
        - backend.apply_action(...) for modification of loads is implemented        

        NB: it does not check whether or not the modification is
        consistent with the input. This will be done in a later test"""
        self.skip_if_needed()
        backend = self.aux_make_backend()
        np.random.seed(0)
        random_load_p = np.random.uniform(0, 1, size=type(backend).n_load)
        random_load_q = np.random.uniform(0, 1, size=type(backend).n_load)
        
        # try to modify load_p
        action = type(backend)._complete_action_class()
        action.update({"injection": {"load_p": 1.01 * random_load_p}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # modification of load_p only
        
        # try to modify load_q
        action = type(backend)._complete_action_class()
        action.update({"injection": {"load_q": 1.01 * random_load_q}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # modification of load_q only
        
    def test_03modify_gen(self):
        """Tests the generators (including slack !) can be modified 

        This test supposes that :
        
        - backend.load_grid(...) is implemented        
        - backend.apply_action(...) for modification of generators is implemented
                
        NB: it does not check whether or not the modification is
        consistent with the input. This will be done in a later test"""
        self.skip_if_needed()
        backend = self.aux_make_backend()
        np.random.seed(0)
        random_gen_p = np.random.uniform(0, 1, size=type(backend).n_gen)
        random_gen_v = np.random.uniform(0, 1, size=type(backend).n_gen)
        
        # try to modify gen_p
        action = type(backend)._complete_action_class()
        action.update({"injection": {"prod_p": 1.01 * random_gen_p}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # modification of prod_p / gen_p only
        
        # try to modify prod_v only
        action = type(backend)._complete_action_class()
        action.update({"injection": {"prod_v": random_gen_v + 0.1}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # modification of prod_v / gen_v only
        
    def test_04disco_reco_lines(self):
        """Tests the powerlines can be disconnected and connected

        This test supposes that :
        
        - backend.load_grid(...) is implemented        
        - backend.apply_action(...) for connection / reconnection of powerline is implemented
                
        NB: it does not check whether or not the modification is
        consistent with the input. This will be done in a later test"""
        self.skip_if_needed()
        backend = self.aux_make_backend()
        
        line_id = 0
        # try to disconnect line 0
        action = type(backend)._complete_action_class()
        action.update({"set_line_status": [(line_id, -1)]})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # disconnection of line 0 only
        
        # try to reconnect line 0
        action = type(backend)._complete_action_class()
        action.update({"set_line_status": [(line_id, +1)]})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # reconnection of line 0 only
    
    def test_05change_topology(self):
        """try to change the topology of 2 different substations : connect their elements to different busbars

        This test supposes that :
        
        - backend.load_grid(...) is implemented        
        - backend.apply_action(...) for topology (change of busbars) is implemented
        
        NB: it does not check whether or not the modification is
        consistent with the input. This will be done in a later test"""
        self.skip_if_needed()
        backend = self.aux_make_backend()
        
        sub_id = 0
        # everything on busbar 2 at sub 0 (should have no impact)
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"substations_id": [(sub_id, [2 for _ in range(type(backend).sub_info[sub_id])])]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # everything on busbar 2 at sub 0
        
        sub_id = 1
        # mix of bus 1 and 2 on substation 1
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"substations_id": [(sub_id, [i % 2 + 1 for i in range(type(backend).sub_info[sub_id])])]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        
    def test_06modify_shunt(self):
        """Tests the shunt can be modified (p, q and topology)

        This test supposes that :
        
        - backend.load_grid(...) is implemented        
        - backend.apply_action(...) for shunts is implemented
        
        NB: this test is skipped if your backend does not support (yet :-) ) shunts
                
        NB: it does not check whether or not the modification is
        consistent with the input. This will be done in a later test
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        cls = type(backend)
        if not cls.shunts_data_available:
            self.skipTest("Your backend does not support shunts")
        
        init_shunt_p = np.array([0.0])
        init_shunt_q = np.array([-19.])
        init_shunt_bus = np.array([1])
        
        # try to modify shunt_p
        action = type(backend)._complete_action_class()
        action.update({"shunt": {"shunt_p": init_shunt_p + 0.01}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # modification of shunt_p only
        
        # try to modify shunt_q only
        action = type(backend)._complete_action_class()
        action.update({"shunt": {"shunt_q": init_shunt_q * 1.01}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # modification of shunt_q only
        
        # try to modify shunt_bus only
        action = type(backend)._complete_action_class()
        action.update({"shunt": {"shunt_bus": init_shunt_bus + 1}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # modification of shunt_bus only

    def test_07modify_storage(self):
        """Tests the modification of storage unit (active power)
        
        NB it does not check whether or not the modification is
        consistent with the input. This will be done in a later test
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented        
        - backend.apply_action(...) for storage units is implemented
        
        NB: this test is skipped if your backend does not support (yet :-) ) storage units
                
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        cls = type(backend)
        if cls.n_storage == 0:
            self.skipTest("Your backend does not support storage units")
        
        storage_power = np.array([-0.5, +0.5])
        # try to modify storage active power only
        action = type(backend)._complete_action_class()
        action.update({"set_storage": storage_power})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # try to modify storage active power only     
        
    def test_08run_ac_pf(self):
        """Tests the runpf method (AC) without modification
    
        This test supposes that :
        
        - backend.load_grid(...) is implemented        
        - backend.runpf() (DC mode) is implemented
                
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        
        res = backend.runpf(is_dc=False)
        assert len(res) == 2, "runpf should return tuple of size 2"
        converged, exc_ = res
        if converged:
            assert exc_ is None, "when a powerflow converges, we expect exc_ (2nd returned value) to be None"
        else:
            warnings.warn("It is surprising that your backend diverges without any modification (AC)")
            assert isinstance(exc_, Exception), "when a powerflow diverges, we expect exc_ (2nd returned value) to be an exception"
            
    def test_09run_dc_pf(self):
        """Tests the runpf method (DC) without modification
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented        
        - backend.runpf() (DC mode) is implemented
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        
        res = backend.runpf(is_dc=True)
        assert len(res) == 2, "runpf should return tuple of size 2"
        converged, exc_ = res
        if converged:
            assert exc_ is None, "when a powerflow converges, we expect exc_ (2nd returned value) to be None"
        else:
            warnings.warn("It is surprising that your backend diverges without any modification (DC)")
            assert isinstance(exc_, Exception), "when a powerflow diverges, we expect exc_ (2nd returned value) to be an exception"
    
    def test_10_ac_forced_divergence(self):
        """increase the load / generation until the powerflow diverges, and check the flags are properly returned
        
        This test supposes that :

        - backend.load_grid(...) is implemented        
        - backend.runpf() (AC mode) is implemented
        - backend.apply_action() for generator and load is implemented
        - backend.generators_info() is implemented
        - backend.loads_info() is implemented
        
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        
        res = backend.runpf(is_dc=False)
        assert len(res) == 2, "runpf should return tuple of size 2"
        
        init_gen_p, *_ = backend.generators_info() 
        init_load_p, *_ = backend.loads_info()
        
        gen_p = 1. * init_gen_p
        load_p = 1. * init_load_p
        nb_iter = 0
        while True:
            gen_p *= 1.5
            load_p *= 1.5
            action = type(backend)._complete_action_class()
            action.update({"injection": {"prod_p": gen_p,
                                         "load_p": load_p}})
            bk_act = type(backend).my_bk_act_class()
            bk_act += action
            backend.apply_action(bk_act)
            res = backend.runpf(is_dc=False)
            converged, exc_ = res
            if converged:
                assert exc_ is None, "when a powerflow converges, we expect exc_ (2nd returned value) to be None"
            else:
                assert isinstance(exc_, Exception), "when a powerflow diverges, we expect exc_ (2nd returned value) to be an exception"
                break
            nb_iter += 1
            if nb_iter >= 10:
                raise RuntimeError("It is surprising that your backend still converges when the load / generation are multiplied by "
                                   "something like 50 (1.5**10). I suppose it's an error. "
                                   "It should stop in approx 3 iteration (so when multiplied by 1.5**3)")
                
    def test_11_modify_load_pf_getter(self):
        """Tests that the modification of loads has an impact on the backend (by reading back the states)
        
        This test supposes that :

        - backend.load_grid(...) is implemented        
        - backend.runpf() (AC mode) is implemented
        - backend.apply_action() for generator and load is implemented
        - backend.loads_info() is implemented
        - backend.generators_info() is implemented
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        
        res = backend.runpf(is_dc=False)
        tmp = backend.loads_info()
        assert len(tmp) == 3, "loads_info() should return 3 elements: load_p, load_q, load_v (see doc)"
        load_p_init, load_q_init, load_v_init = tmp 
        init_gen_p, *_ = backend.generators_info() 
        
        # try to modify load_p
        action = type(backend)._complete_action_class()
        action.update({"injection": {"load_p": 1.01 * load_p_init,
                                     "prod_p": 1.01 * init_gen_p,
                                     "load_q": 1.01 * load_q_init}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # modification of load_p, load_q and gen_p        
        
        res2 = backend.runpf(is_dc=False)
        assert res2[0], "backend should not have diverge after such a little perturbation"
        tmp2 = backend.loads_info()
        assert len(tmp) == 3, "loads_info() should return 3 elements: load_p, load_q, load_v (see doc)"
        load_p_after, load_q_after, load_v_after = tmp2 
        assert not np.allclose(load_p_after, load_p_init), f"load_p does not seemed to be modified by apply_action when loads are impacted (active value): check `apply_action` for load_p"
        assert not np.allclose(load_q_after, load_q_init), f"load_q does not seemed to be modified by apply_action when loads are impacted (reactive value): check `apply_action` for load_q"
        
        # now a basic check for "one load at a time"
        delta_mw = 1.
        delta_mvar = 1.
        for load_id in range(backend.n_load):
            this_load_p = 1. * load_p_init
            this_load_p[load_id] += delta_mw  # add 1 MW
            this_load_q = 1. * load_q_init
            this_load_q[load_id] += delta_mvar  # add 1 MVAr
            action = type(backend)._complete_action_class()
            action.update({"injection": {"load_p": this_load_p,
                                         "prod_p": init_gen_p,
                                         "load_q": this_load_q}})
            bk_act = type(backend).my_bk_act_class()
            bk_act += action
            backend.apply_action(bk_act)  # modification of load_p, load_q and gen_p   
            res_tmp = backend.runpf(is_dc=False)
            assert res_tmp[0], "backend should not have diverge after such a little perturbation"
            tmp = backend.loads_info() 
            assert np.abs(tmp[0][load_id] - load_p_init[load_id]) >= delta_mw / 2., f"error when trying to modify load {load_id}: check the consistency between backend.loads_info() and backend.apply_action for load_p"
            assert np.abs(tmp[1][load_id] - load_q_init[load_id]) >= delta_mvar / 2., f"error when trying to modify load {load_id}: check the consistency between backend.loads_info() and backend.apply_action for load_q"

    def test_12_modify_gen_pf_getter(self):
        """Tests that the modification of generators has an impact on the backend (by reading back the states)
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC mode) is implemented
        - backend.apply_action() for generator and load is implemented
        - backend.generators_info() is implemented
        - backend.loads_info() is implemented
        
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()        
        res = backend.runpf(is_dc=False)
        tmp = backend.generators_info()
        assert len(tmp) == 3, "generators_info() should return 3 elements: gen_p, gen_q, gen_v (see doc)"
        gen_p_init, gen_q_init, gen_v_init = tmp 
        load_p_init, *_ = backend.loads_info()
        
        # try to modify load_p
        action = type(backend)._complete_action_class()
        action.update({"injection": {"load_p": 1.01 * load_p_init,
                                     "prod_p": 1.01 * gen_p_init,
                                     "prod_v": gen_v_init + 0.1}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # modification of load_p, load_q and gen_p        
        
        res2 = backend.runpf(is_dc=False)
        assert res2[0], "backend should not have diverge after such a little perturbation"
        tmp2 = backend.generators_info()
        assert len(tmp) == 3, "generators_info() should return 3 elements: gen_p, gen_q, gen_v (see doc)"
        gen_p_after, gen_q_after, gen_v_after = tmp2 
        assert not np.allclose(gen_p_after, gen_p_init), f"gen_p does not seemed to be modified by apply_action when generators are impacted (active value): check `apply_action` for gen_p / prod_p"
        assert not np.allclose(gen_v_after, gen_v_init), f"gen_v does not seemed to be modified by apply_action when generators are impacted (voltage setpoint value): check `apply_action` for gen_v / prod_v"

        # now a basic check for "one gen at a time"
        # NB this test cannot be done like this for "prod_v" / gen_v because two generators might be connected to the same
        # bus, and changing only one would cause an issue !
        delta_mw = 1.
        nb_error = 0
        prev_exc = None
        for gen_id in range(backend.n_gen):
            this_gen_p = 1. * gen_p_init
            this_gen_p[gen_id] += delta_mw  # remove 1 MW
            action = type(backend)._complete_action_class()
            action.update({"injection": {"load_p": load_p_init,
                                         "prod_p": this_gen_p}})
            bk_act = type(backend).my_bk_act_class()
            bk_act += action
            backend.apply_action(bk_act)
            res_tmp = backend.runpf(is_dc=False)
            assert res_tmp[0], "backend should not have diverge after such a little perturbation"
            tmp = backend.generators_info() 
            if np.abs(tmp[0][gen_id] - gen_p_init[gen_id]) <= delta_mw / 2.:
                # in case of non distributed slack, backend cannot control the generator acting as the slack.
                # this is why this test is expected to fail at most once.
                # if it fails twice, then there is a bug.
                if prev_exc is None:
                    prev_exc = AssertionError(f"error when trying to modify active generator of gen {gen_id}: check the consistency between backend.generators_info() and backend.apply_action for gen_p / prod_p")
                else:
                    raise AssertionError(f"error when trying to modify active generator of gen {gen_id}: check the consistency between backend.generators_info() and backend.apply_action for gen_p / prod_p") from prev_exc

    def test_13_disco_reco_lines_pf_getter(self):
        """Tests the powerlines can be disconnected and connected and that getter info are consistent
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC mode) is implemented
        - backend.apply_action() for line reconnection / disconnection
        - backend.generators_info(), loads_info(), storages_info(), shunts_info(), lines_or_info(), lines_ex_info() is implemented
        - backend.get_topo_vect() is implemented
        
        It is expected that this test fails if there are shunts (or storage units) in 
        the grid (modeled by your powerflow) but that you did not yet coded the interface
        between said element and grid2op (the backend you are creating)
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        cls = type(backend)
        
        res = backend.runpf(is_dc=False)
        tmp_or = backend.lines_or_info()
        assert len(tmp_or) == 4, "lines_or_info() should return 4 elements: p, q, v, a (see doc)"
        p_or, q_or, v_or, a_or = tmp_or 
        for arr, arr_nm in zip([p_or, q_or, v_or, a_or],
                               ["p_or", "q_or", "v_or", "a_or"]):
            if arr.shape[0] != cls.n_line:
                raise RuntimeError(f"{arr_nm} should have size {cls.n_line} (number of lines) but has size {arr.shape[0]}")
        tmp_ex = backend.lines_ex_info()
        assert len(tmp_ex) == 4, "lines_ex_info() should return 4 elements: p, q, v, a (see doc)"
        p_ex, q_ex, v_ex, a_ex = tmp_ex
        for arr, arr_nm in zip([p_ex, q_ex, v_ex, a_ex],
                               ["p_ex", "q_ex", "v_ex", "a_ex"]):
            if arr.shape[0] != cls.n_line:
                raise RuntimeError(f"{arr_nm} should have size {cls.n_line} (number of lines) but has size {arr.shape[0]}")
        
        line_id = 0
        
        # try to disconnect line 0
        action1 = type(backend)._complete_action_class()
        action1.update({"set_line_status": [(line_id, -1)]})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action1
        backend.apply_action(bk_act)  # disconnection of line 0 only
        res_disco = backend.runpf(is_dc=False)
        assert res_disco[0], f"your backend diverge after disconnection of line {line_id}, which should not be the case"
        tmp_or_disco = backend.lines_or_info()
        tmp_ex_disco = backend.lines_ex_info()
        assert not np.allclose(tmp_or_disco[0], p_or), f"p_or does not seemed to be modified by apply_action when a powerline is disconnected (active value): check `apply_action` for line connection disconnection"
        assert not np.allclose(tmp_or_disco[1], p_or), f"q_or does not seemed to be modified by apply_action when a powerline is disconnected (active value): check `apply_action` for line connection disconnection"
        assert not np.allclose(tmp_ex_disco[0], p_ex), f"p_ex does not seemed to be modified by apply_action when a powerline is disconnected (active value): check `apply_action` for line connection disconnection"
        assert not np.allclose(tmp_ex_disco[1], p_ex), f"q_ex does not seemed to be modified by apply_action when a powerline is disconnected (active value): check `apply_action` for line connection disconnection"
        assert np.allclose(tmp_or_disco[0][line_id], 0.), f"origin flow (active) on disconnected line {line_id} is > 0."
        assert np.allclose(tmp_or_disco[1][line_id], 0.), f"origin flow (reactive) on disconnected line {line_id} is > 0."
        assert np.allclose(tmp_or_disco[2][line_id], 0.), f"origin voltage on disconnected line {line_id} is > 0."
        assert np.allclose(tmp_or_disco[3][line_id], 0.), f"origin flow (amps) on disconnected line {line_id} is > 0."
        assert np.allclose(tmp_ex_disco[0][line_id], 0.), f"extremity flow (active) on disconnected line {line_id} is > 0."
        assert np.allclose(tmp_ex_disco[1][line_id], 0.), f"extremity flow (reactive) on disconnected line {line_id} is > 0."
        assert np.allclose(tmp_ex_disco[2][line_id], 0.), f"extremity voltage on disconnected line {line_id} is > 0."
        assert np.allclose(tmp_ex_disco[3][line_id], 0.), f"extremity flow (amps) on disconnected line {line_id} is > 0."
        
        # try to reconnect line 0
        action2 = type(backend)._complete_action_class()
        action2.update({"set_line_status": [(line_id, +1)]})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action1
        bk_act += action2
        backend.apply_action(bk_act)  # disconnection of line 0 only
        res_disco = backend.runpf(is_dc=False)
        assert res_disco[0], f"your backend diverge after disconnection of line {line_id}, which should not be the case"
        tmp_or_reco = backend.lines_or_info()
        tmp_ex_reco = backend.lines_ex_info()
        assert not np.allclose(tmp_or_disco[0], tmp_or_reco[0]), f"p_or does not seemed to be modified by apply_action when a powerline is reconnected (active value): check `apply_action` for line connection reconnection"
        assert not np.allclose(tmp_or_disco[1], tmp_or_reco[1]), f"q_or does not seemed to be modified by apply_action when a powerline is reconnected (active value): check `apply_action` for line connection reconnection"
        assert not np.allclose(tmp_ex_disco[0], tmp_ex_reco[0]), f"p_ex does not seemed to be modified by apply_action when a powerline is reconnected (active value): check `apply_action` for line connection reconnection"
        assert not np.allclose(tmp_ex_disco[1], tmp_ex_reco[0]), f"q_ex does not seemed to be modified by apply_action when a powerline is reconnected (active value): check `apply_action` for line connection reconnection"
        assert not np.allclose(tmp_or_reco[0][line_id], 0.), f"origin flow (active) on connected line {line_id} is 0."
        assert not np.allclose(tmp_or_reco[1][line_id], 0.), f"origin flow (reactive) on connected line {line_id} is 0."
        assert not np.allclose(tmp_or_reco[2][line_id], 0.), f"origin voltage on connected line {line_id} is > 0."
        assert not np.allclose(tmp_or_reco[3][line_id], 0.), f"origin flow (amps) on connected line {line_id} is > 0."
        assert not np.allclose(tmp_ex_reco[0][line_id], 0.), f"extremity flow (active) on connected line {line_id} is > 0."
        assert not np.allclose(tmp_ex_reco[1][line_id], 0.), f"extremity flow (reactive) on connected line {line_id} is > 0."
        assert not np.allclose(tmp_ex_reco[2][line_id], 0.), f"extremity voltage on connected line {line_id} is > 0."
        assert not np.allclose(tmp_ex_reco[3][line_id], 0.), f"extremity flow (amps) on connected line {line_id} is > 0."

    def test_14change_topology(self):
        """try to change the topology of 2 different substations : connect their elements to different busbars and check consistency
        
        The same action as for test test_05change_topology.
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC mode) is implemented
        - backend.apply_action() for topology modification
        - backend.generators_info(), loads_info(), storages_info(), shunts_info(), lines_or_info(), lines_ex_info() is implemented
        - backend.get_topo_vect() is implemented
        
        It is expected that this test fails if there are shunts (or storage units) in 
        the grid (modeled by your powerflow) but that you did not yet coded the interface
        between said element and grid2op (the backend you are creating)
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        cls = type(backend)
        
        res = backend.runpf(is_dc=False)
        p_subs, q_subs, p_bus, q_bus, diff_v_bus = backend.check_kirchoff()
        assert np.allclose(p_subs, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (no modif): kirchoff laws are not met for p (creation or suppression of active). Check the handling of the slack bus(se) maybe ?"
        assert np.allclose(q_subs, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (no modif): kirchoff laws are not met for q (creation or suppression of reactive). Check the handling of the slack bus(se) maybe ?"
        assert np.allclose(p_bus, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (no modif): kirchoff laws are not met for p (creation or suppression of active). Check the handling of the slack bus(se) maybe ?"
        assert np.allclose(q_bus, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (no modif): kirchoff laws are not met for q (creation or suppression of reactive). Check the handling of the slack bus(se) maybe ?"
        assert np.allclose(diff_v_bus, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (no modif): some nodes have two different voltages. Check the accessor for voltage in all the `***_info()` (*eg* `loads_info()`)"
        
        p_or, q_or, v_or, a_or = backend.lines_or_info()
        
        sub_id = 0
        # everything on busbar 2 at sub 0 (should have no impact)
        action1 = type(backend)._complete_action_class()
        action1.update({"set_bus": {"substations_id": [(sub_id, [2 for _ in range(type(backend).sub_info[sub_id])])]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action1
        backend.apply_action(bk_act)  # everything on busbar 2 at sub 0
        res = backend.runpf(is_dc=False)
        assert res[0], "Your powerflow has diverged after the loading of the file, which should not happen"
        p_subs, q_subs, p_bus, q_bus, diff_v_bus = backend.check_kirchoff()
        assert np.allclose(p_subs, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (modif with no impact): kirchoff laws are not met for p (creation or suppression of active)."
        assert np.allclose(q_subs, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (modif with no impact): kirchoff laws are not met for q (creation or suppression of reactive)."
        assert np.allclose(p_bus, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (modif with no impact): kirchoff laws are not met for p (creation or suppression of active)."
        assert np.allclose(q_bus, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (modif with no impact): kirchoff laws are not met for q (creation or suppression of reactive)."
        assert np.allclose(diff_v_bus, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow: some nodes have two different voltages. Check the accessor for voltage in all the `***_info()` (*eg* `loads_info()`)"
        p_after_or, q_after_or, v_after_or, a_after_or = backend.lines_or_info()
        assert np.allclose(p_after_or, p_or), f"The p_or flow changed while the topology action is supposed to have no impact, check the `apply_action` for topology"
        assert np.allclose(q_after_or, q_or), f"The q_or flow changed while the topology action is supposed to do nothing, check the `apply_action` for topology"
        assert np.allclose(v_after_or, v_or), f"The v_or changed while the topology action is supposed to do nothing, check the `apply_action` for topology"
        assert np.allclose(a_after_or, a_or), f"The a_or flow changed while the topology action is supposed to do nothing, check the `apply_action` for topology"
        
        sub_id = 1
        # mix of bus 1 and 2 on substation 1
        action2 = type(backend)._complete_action_class()
        action2.update({"set_bus": {"substations_id": [(sub_id, [i % 2 + 1 for i in range(type(backend).sub_info[sub_id])])]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action1
        bk_act += action2
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=False)
        assert res[0], "Your powerflow has diverged after a topology action (but should not). Check `apply_action` for topology"
        p_subs, q_subs, p_bus, q_bus, diff_v_bus = backend.check_kirchoff()
        assert np.allclose(p_subs, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (modif with a real impact): kirchoff laws are not met for p (creation or suppression of active)."
        assert np.allclose(q_subs, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (modif with a real impact): kirchoff laws are not met for q (creation or suppression of reactive)."
        assert np.allclose(p_bus, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (modif with a real impact): kirchoff laws are not met for p (creation or suppression of active)."
        assert np.allclose(q_bus, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow (modif with a real impact): kirchoff laws are not met for q (creation or suppression of reactive)."
        assert np.allclose(diff_v_bus, 0., atol=3 * self.tol_one), "there are some discrepency in the backend after a powerflow: some nodes have two different voltages. Check the accessor for voltage in all the `***_info()` (*eg* `loads_info()`)"
        p_after_or, q_after_or, v_after_or, a_after_or = backend.lines_or_info()
        assert not np.allclose(p_after_or, p_or), f"The p_or flow doesn't change while the topology action is supposed to have a real impact, check the `apply_action` for topology"
        assert not np.allclose(q_after_or, q_or), f"The q_or flow doesn't change while the topology action is supposed to have a real impact, check the `apply_action` for topology"
        assert not np.allclose(v_after_or, v_or), f"The v_or doesn't change while the topology action is supposed to have a real impact, check the `apply_action` for topology"
        assert not np.allclose(a_after_or, a_or), f"The a_or flow doesn't change while the topology action is supposed to have a real impact, check the `apply_action` for topology"
    
    def test_15_reset(self):
        """Tests that when a backend is reset, it is indeed reset in the original state
        
        This test supposes that :

        - backend.load_grid(...) is implemented        
        - backend.runpf() (AC mode) is implemented
        - backend.apply_action() for generator and load is implemented
        - backend.loads_info() is implemented
        - backend.generators_info() is implemented
        - backend.lines_or_info() is implemented
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        cls = type(backend)
        backend2 = self.aux_make_backend()
        res = backend.runpf(is_dc=False)
        assert len(res) == 2, "runpf should return tuple of size 2"
        
        # create a situation where the backend diverges
        init_gen_p, *_ = backend.generators_info() 
        init_load_p, *_ = backend.loads_info()
        gen_p = 1. * init_gen_p
        load_p = 1. * init_load_p
        nb_iter = 0
        while True:
            gen_p *= 1.5
            load_p *= 1.5
            action = type(backend)._complete_action_class()
            action.update({"injection": {"prod_p": gen_p,
                                         "load_p": load_p}})
            bk_act = type(backend).my_bk_act_class()
            bk_act += action
            backend.apply_action(bk_act)
            res = backend.runpf(is_dc=False)
            converged, exc_ = res
            if converged:
                assert exc_ is None, "when a powerflow converges, we expect exc_ (2nd returned value) to be None"
            else:
                assert isinstance(exc_, Exception), "when a powerflow diverges, we expect exc_ (2nd returned value) to be an exception"
                break
            nb_iter += 1
            if nb_iter >= 10:
                raise RuntimeError("It is surprising that your backend still converges when the load / generation are multiplied by "
                                   "something like 50 (1.5**10). I suppose it's an error. "
                                   "It should stop in approx 3 iteration (so when multiplied by 1.5**3)")
                
        backend.reset(self.get_path(), self.get_casefile())
        res = backend.runpf(is_dc=False)
        assert res[0], "your backend has diverged after being reset"
        res_ref = backend2.runpf(is_dc=False)
        assert res_ref[0], "your backend has diverged after just loading the grid"
        p_or, q_or, v_or, a_or = backend.lines_or_info()
        p2_or, q2_or, v2_or, a2_or = backend2.lines_or_info()
        assert np.allclose(p2_or, p_or), f"The p_or flow differ between its original value and after a reset. Check backend.reset()"
        assert np.allclose(q2_or, q_or), f"The q_or flow differ between its original value and after a reset. Check backend.reset()"
        assert np.allclose(v2_or, v_or), f"The v_or differ between its original value and after a reset. Check backend.reset()"
        assert np.allclose(a2_or, a_or), f"The a_or flow differ between its original value and after a reset. Check backend.reset()"
                                       
    def test_16_isolated_load_make_divergence(self):
        """Tests that an isolated load will make the method `run_pf` "diverge" (in AC and DC) [behaviour might change in the future]
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC and DC mode) is implemented
        - backend.apply_action() for topology modification
        - backend.reset() is implemented
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        cls = type(backend)
        
        # a load alone on a bus
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"loads_id": [(0, 2)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=False)  
        assert not res[0], "It is expected (at time of writing) that your backend 'diverges' in case of isolated loads in AC."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  
        
        backend.reset(self.get_path(), self.get_casefile())
        # a load alone on a bus
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"loads_id": [(0, 2)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=True)  
        assert not res[0], "It is expected (at time of writing) that your backend 'diverges' in case of isolated loads in DC."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  
    
    def test_17_isolated_gen_make_divergence(self):
        """Tests that an isolated generator will make the method `run_pf` "diverge" (in AC and DC) [behaviour might change in the future]
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC and DC mode) is implemented
        - backend.apply_action() for topology modification
        - backend.reset() is implemented
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        cls = type(backend)
        
        # disconnect a gen
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"generators_id": [(0, 2)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=False)  
        assert not res[0], "It is expected (at time of writing) that your backend 'diverges' in case of isolated gen."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  
        
        backend.reset(self.get_path(), self.get_casefile())
        # disconnect a gen
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"generators_id": [(0, 2)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=True)  
        assert not res[0], "It is expected (at time of writing) that your backend 'diverges' in case of isolated gen."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  
    
    def test_18_isolated_shunt_make_divergence(self):
        """Tests test that an isolated shunt will make the method `run_pf` "diverge" (in AC and DC) [behaviour might change in the future]
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC and DC mode) is implemented
        - backend.apply_action() for topology modification
        - backend.reset() is implemented
        
        NB: this test is skipped if your backend does not (yet :-) ) supports shunt
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        cls = type(backend)
        if not cls.shunts_data_available:
            self.skipTest("Your backend does not support shunts")
        if cls.n_shunt == 0:
            self.skipTest("Your grid has no shunt in it")
                        
        # make a shunt alone on a bus
        action = type(backend)._complete_action_class()
        action.update({"shunt": {"shunt_bus": [(0, 2)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=False)  
        assert not res[0], "It is expected (at time of writing) that your backend 'diverges' in case of isolated shunt."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  
        
        backend.reset(self.get_path(), self.get_casefile())
        # make a shunt alone on a bus
        action = type(backend)._complete_action_class()
        action.update({"shunt": {"shunt_bus": [(0, 2)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=True)  
        assert not res[0], "It is expected (at time of writing) that your backend 'diverges' in case of isolated shunt."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  
                       
    def test_19_isolated_storage_make_divergence(self):
        """Teststest that an isolated storage unit will make the method `run_pf` "diverge" (in AC and DC) [behaviour might change in the future]
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC and DC mode) is implemented
        - backend.apply_action() for topology modification
        - backend.reset() is implemented
        
        NB: this test is skipped if your backend does not (yet :-) ) supports storage units
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        cls = type(backend)
        if cls.n_storage == 0:
            self.skipTest("Your backend does not support storage units")
            
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"storages_id": [(0, 2)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=False)  
        assert not res[0], "It is expected (at time of writing) that your backend 'diverges' in case of isolated storage unit."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  
        
        backend.reset(self.get_path(), self.get_casefile())
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"storages_id": [(0, 2)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=True)  
        assert not res[0], "It is expected (at time of writing) that your backend 'diverges' in case of isolated storage unit."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  

    def test_20_disconnected_load_make_divergence(self):
        """Tests that a disconnected load unit will make the method `run_pf` "diverge" (in AC and DC) [behaviour might change in the future]
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC and DC mode) is implemented
        - backend.apply_action() for topology modification
        - backend.reset() is implemented
        
        NB: this test is skipped if your backend does not (yet :-) ) supports storage units
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        
        # a load alone on a bus
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"loads_id": [(0, -1)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=False)  
        assert not res[0], "It is expected (at time of writing) that your backend 'diverges' in case of disconnected load in AC."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  
        
        backend.reset(self.get_path(), self.get_casefile())
        # a load alone on a bus
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"loads_id": [(0, -1)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=True)  
        assert not res[0], "It is expected (at time of writing) that your backend 'diverges' in case of disconnected load in DC."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  
                                
    def test_21_disconnected_gen_make_divergence(self):
        """Tests that a disconnected generator will make the method `run_pf` "diverge" (in AC and DC) [behaviour might change in the future]
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC and DC mode) is implemented
        - backend.apply_action() for topology modification
        - backend.reset() is implemented
        
        NB: this test is skipped if your backend does not (yet :-) ) supports storage units
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        
        # a disconnected generator
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"generators_id": [(0, -1)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=False)  
        assert not res[0], "It is expected (at time of writing) that your backend 'diverges' in case of disconnected gen in AC."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  
        
        backend.reset(self.get_path(), self.get_casefile())
        # a disconnected generator
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"generators_id": [(0, -1)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=True)  
        assert not res[0], "It is expected (at time of writing) that your backend 'diverges' in case of disconnected gen in DC."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  

    def test_22_islanded_grid_make_divergence(self):
        """Tests that when the grid is split in two different "sub_grid" it makes the runpf diverge both in AC and DC
        
        For information, this is suppose to make a subgrid with substation 5, 11 and 12 on one side
        and all the rest on the other.
        
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC and DC mode) is implemented
        - backend.apply_action() for topology modification
        - backend.reset() is implemented
        
        NB: this test is skipped if your backend does not (yet :-) ) supports storage units
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        # a non connected grid
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"lines_or_id": [(17, 2)],
                                   "lines_ex_id": [(7, 2), (14, 2)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=False)  
        assert not res[0], "It is expected that your backend 'diverges' in case of non connected grid in AC."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  
        backend.reset(self.get_path(), self.get_casefile())
        
        # a non connected grid
        backend.reset(self.get_path(), self.get_casefile())
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"lines_or_id": [(17, 2)],
                                   "lines_ex_id": [(7, 2), (14, 2)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)  # mix of bus 1 and 2 on substation 1
        res = backend.runpf(is_dc=True)  
        assert not res[0], "It is expected that your backend 'diverges' in case of non connected grid in DC."                 
        assert res[1] is not None, "When your backend diverges, we expect it throws an exception (second return value)"  

    def test_23_disco_line_v_null(self):
        """Tests that disconnected elements shunt have v = 0. (and not nan or something)
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC and DC mode) is implemented
        - backend.apply_action() for topology modification
        - backend.shunt() and lines_ex_info() are implemented
        - backend.reset() is implemented
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        
        line_id = 0
        # a disconnected line
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"lines_or_id": [(line_id, -1)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)
        
        res = backend.runpf(is_dc=False)  
        p_or, q_or, v_or, a_or = backend.lines_or_info()
        p_ex, q_ex, v_ex, a_ex = backend.lines_ex_info()
        assert np.allclose(v_or[line_id], 0.), f"v_or should be 0. for disconnected line, but is currently {v_or[line_id]} (AC)" 
        assert np.allclose(v_ex[line_id], 0.), f"v_ex should be 0. for disconnected line, but is currently {v_ex[line_id]} (AC)" 
        assert np.allclose(a_or[line_id], 0.), f"v_or should be 0. for disconnected line, but is currently {v_or[line_id]} (AC)" 
        assert np.allclose(a_ex[line_id], 0.), f"v_ex should be 0. for disconnected line, but is currently {v_ex[line_id]} (AC)" 
        
        # reset and do the same in DC
        backend.reset(self.get_path(), self.get_casefile())
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"lines_or_id": [(line_id, -1)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)
        
        res = backend.runpf(is_dc=True)  
        p_or, q_or, v_or, a_or = backend.lines_or_info()
        p_ex, q_ex, v_ex, a_ex = backend.lines_ex_info()
        assert np.allclose(v_or[line_id], 0.), f"v_or should be 0. for disconnected line, but is currently {v_or[line_id]} (DC)" 
        assert np.allclose(v_ex[line_id], 0.), f"v_ex should be 0. for disconnected line, but is currently {v_ex[line_id]} (DC)" 
        assert np.allclose(a_or[line_id], 0.), f"v_or should be 0. for disconnected line, but is currently {v_or[line_id]} (DC)" 
        assert np.allclose(a_ex[line_id], 0.), f"v_ex should be 0. for disconnected line, but is currently {v_ex[line_id]} (DC)" 
        
    def test_24_disco_shunt_v_null(self):
        """Tests that disconnected shunts have v = 0. (and not nan or something)
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC and DC mode) is implemented
        - backend.apply_action() for topology modification
        - backend.storages_info() are implemented
        - backend.reset() is implemented
        
        NB: this test is skipped if your backend does not support shunt (yet :-) )
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        cls = type(backend)
        if not cls.shunts_data_available:
            self.skipTest("Your backend does not support shunts")
        if cls.n_shunt == 0:
            self.skipTest("Your grid has no shunt in it")
            
        shunt_id = 0
        # a disconnected shunt
        action = type(backend)._complete_action_class()
        action.update({"shunt": {"shunt_bus": [(shunt_id, -1)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)
        res = backend.runpf(is_dc=False)  
        p_, q_, v_, bus_ = backend.shunt_info()
        assert np.allclose(v_[shunt_id], 0.), f"v should be 0. for disconnected shunt, but is currently {v_[shunt_id]} (AC)" 
        assert bus_[shunt_id] == -1, f"bus_ should be -1 for disconnected shunt, but is currently {bus_[shunt_id]} (AC)" 
        
        # a disconnected shunt
        backend.reset(self.get_path(), self.get_casefile())
        action = type(backend)._complete_action_class()
        action.update({"shunt": {"shunt_bus": [(shunt_id, -1)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)
        res = backend.runpf(is_dc=True)  
        p_, q_, v_, bus_ = backend.shunt_info()
        assert np.allclose(v_[shunt_id], 0.), f"v should be 0. for disconnected shunt, but is currently {v_[shunt_id]} (DC)" 
        assert bus_[shunt_id] == -1, f"bus_ should be -1 for disconnected shunt, but is currently {bus_[shunt_id]} (DC)" 
        
    def test_25_disco_storage_v_null(self):
        """Tests that disconnected shunts have v = 0. (and not nan or something)
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC and DC mode) is implemented
        - backend.apply_action() for topology modification
        - backend.storages_info() are implemented
        - backend.reset() is implemented
        
        NB: this test is skipped if your backend does not support storage unit (yet :-) )
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        cls = type(backend)
        if cls.n_storage == 0:
            self.skipTest("Your backend does not support storage unit")
        
        storage_id = 0
        # a disconnected storage
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"storages_id": [(storage_id, -1)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)
        
        res = backend.runpf(is_dc=False)  
        p_, q_, v_ = backend.storages_info()
        assert np.allclose(v_[storage_id], 0.), f"v should be 0. for disconnected storage, but is currently {v_[storage_id]} (AC)" 
        
        # disconnect a storage
        backend.reset(self.get_path(), self.get_casefile())
        action = type(backend)._complete_action_class()
        action.update({"set_bus": {"storages_id": [(storage_id, -1)]}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)
        res = backend.runpf(is_dc=True)  
        p_, q_, v_ = backend.storages_info()
        assert np.allclose(v_[storage_id], 0.), f"v should be 0. for disconnected storage, but is currently {v_[storage_id]} (AC)" 
    
    def test_26_copy(self):    
        """Tests that the backend can be copied (and that the copied backend and the 
        original one can be modified independantly)
        
        This test supposes that :
        
        - backend.load_grid(...) is implemented
        - backend.runpf() (AC and DC mode) is implemented
        - backend.apply_action() for prod / gen modification
        - backend.loads_info() are implemented
        - backend.generators_info() are implemented
        - backend.lines_or_info() are implemented
        - backend.reset() is implemented
        
        NB: this test is skipped if the backend cannot be copied
        """
        self.skip_if_needed()
        backend = self.aux_make_backend()
        if not backend._can_be_copied:
            with self.assertRaises(BackendError):
                backend_cpy = backend.copy()
            return
        
        # backend can be copied
        backend_cpy = backend.copy()
        assert isinstance(backend_cpy, type(backend)), f"backend.copy() is supposed to return an object of the same type as your backend. Check backend.copy()"
        backend.runpf(is_dc=False)
        # now modify original one
        init_gen_p, *_ = backend.generators_info() 
        init_load_p, *_ = backend.loads_info()
        
        action = type(backend)._complete_action_class()
        action.update({"injection": {"prod_p": 1.1 * init_gen_p,
                                     "load_p": 1.1 * init_load_p}})
        bk_act = type(backend).my_bk_act_class()
        bk_act += action
        backend.apply_action(bk_act)
        res = backend.runpf(is_dc=True) 
        res_cpy = backend_cpy.runpf(is_dc=True) 
        
        p_or, *_ = backend.lines_or_info()
        p_or_cpy, *_ = backend_cpy.lines_or_info()
        assert not np.allclose(p_or, p_or_cpy), (f"The p_or for your backend and its copy are identical though one has been modify and not the other. "
                                                  "It is likely that backend.copy implementation does not perform a deep copy")            
    