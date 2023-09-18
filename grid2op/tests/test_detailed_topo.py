# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import numpy as np
import hashlib

import grid2op
from grid2op.Action import BaseAction, CompleteAction
from grid2op.Observation import BaseObservation
from grid2op.Runner import Runner
from grid2op.Backend import PandaPowerBackend
from grid2op.Space import DetailedTopoDescription
from grid2op.Agent import BaseAgent

import pdb
REF_HASH = '7d79e8debc7403dae95bd95a023d5627a8a760e34bb26e3adfd2b842446830d455b53aeb5d89276b0e431f9022dc1c73e77ff3ecb10df0f60aaaf65754bbdf87'


def _aux_test_correct(detailed_topo_desc, dim_topo):
    assert detailed_topo_desc is not None
    assert  detailed_topo_desc.load_to_busbar_id == [
        (1, 15), (2, 16), (3, 17), (4, 18), (5, 19), (8, 22), (9, 23), (10, 24), (11, 25), (12, 26), (13, 27)
    ]
    assert detailed_topo_desc.gen_to_busbar_id == [(1, 15), (2, 16), (5, 19), (5, 19), (7, 21), (0, 14)]
    
    # test the switches (but i don't want to copy this huge data here)
    assert (detailed_topo_desc.switches.sum(axis=0) == np.array([712, 310, 902, 180])).all()
    ref_1 = np.array([ 7,  8,  4,  5,  5,  6,  2,  3,  3,  4,  7,  8,  8,  9,  9, 10,  6,
                       7,  4,  5,  5,  6, 11, 12,  9, 10,  6,  7, 13, 14, 22, 23, 23, 24,
                      11, 12, 13, 14,  8,  9, 25, 26, 10, 11, 13, 14, 15, 16, 10, 11,  9,
                      10, 10, 11, 16, 17, 17, 18, 18, 19, 27, 28,  8,  9, 28, 29, 26, 27,
                      30, 31, 13, 14, 30, 31, 11, 12, 14, 15, 22, 23, 23, 24, 31, 32, 29,
                      30, 14, 15, 16, 17, 25, 26, 24, 25, 18, 19, 22, 23, 27, 28, 20, 21,
                      28, 29, 24, 25, 22, 23, 30, 31, 26, 27, 30, 31, 24, 25, 29, 30, 32,
                      33])
    assert (detailed_topo_desc.switches.sum(axis=1) == ref_1).all()
    assert hashlib.blake2b((detailed_topo_desc.switches.tobytes())).hexdigest() == REF_HASH, f"{hashlib.blake2b((detailed_topo_desc.switches.tobytes())).hexdigest()}"
    
    # siwtches to pos topo vect
    ref_switches_pos_topo_vect = np.array([ 2,  2,  0,  0,  1,  1,  8,  8,  7,  7,  4,  4,  5,  5,  6,  6,  3,
                                            3, 12, 12, 11, 11, 10, 10,  9,  9, 18, 18, 15, 15, 16, 16, 17, 17,
                                           13, 13, 14, 14, 23, 23, 22, 22, 19, 19, 20, 20, 21, 21, 30, 30, 28,
                                           28, 29, 29, 24, 24, 25, 25, 26, 26, 27, 27, 31, 31, 33, 33, 32, 32,
                                           34, 34, 36, 36, 35, 35, 37, 37, 42, 42, 38, 38, 39, 39, 41, 41, 40,
                                           40, -1, -1, 45, 45, 44, 44, 43, 43, 48, 48, 46, 46, 47, 47, 51, 51,
                                           50, 50, 49, 49, 55, 55, 54, 54, 52, 52, 53, 53, 58, 58, 56, 56, 57,
                                           57], dtype=np.int32)
    for i in range(-1, dim_topo):
        assert np.sum(ref_switches_pos_topo_vect == i).sum() == 2, f"error for topo_vect_id = {i}"
    assert np.all(detailed_topo_desc.switches_to_topovect_id == ref_switches_pos_topo_vect)
    
    
class _PPBkForTestDetTopo(PandaPowerBackend):
    def load_grid(self, path=None, filename=None):
        super().load_grid(path, filename)
        self.detailed_topo_desc = DetailedTopoDescription.from_init_grid(self)


class TestDTDAgent(BaseAgent):
    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        _aux_test_correct(type(observation).detailed_topo_desc, type(observation).dim_topo)
        return super().act(observation, reward, done)

        
class DetailedTopoTester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
            "educ_case14_storage",
            test=True,
            backend=_PPBkForTestDetTopo(),
            action_class=CompleteAction,
            _add_to_name="DetailedTopoTester",
        )
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_init_ok(self):
        obs = self.env.reset()
        _aux_test_correct(type(obs).detailed_topo_desc, type(obs).dim_topo)

    def test_work_simulate(self):
        obs = self.env.reset()
        _aux_test_correct(type(obs).detailed_topo_desc, type(obs).dim_topo)
        sim_o, *_ = obs.simulate(self.env.action_space())
        _aux_test_correct(type(sim_o).detailed_topo_desc, type(sim_o).dim_topo)
    
    def test_runner_seq(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner(), agentClass=TestDTDAgent)
        runner.run(nb_episode=1, max_iter=10)
        runner.run(nb_episode=1, max_iter=10, add_detailed_output=True)
    
    def test_runner_par(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner(), agentClass=TestDTDAgent)
        runner.run(nb_episode=2, max_iter=10, nb_process=2)
        runner.run(nb_episode=2, max_iter=10, add_detailed_output=True, nb_process=2)
    
    def test_env_cpy(self):
        obs = self.env.reset()
        env_cpy = self.env.copy()
        obs_cpy = env_cpy.reset()
        _aux_test_correct(type(obs_cpy).detailed_topo_desc, type(obs_cpy).dim_topo)
    
    def test_get_loads_bus_switches(self):
        """test I can acess the loads and also that the results is correctly computed by _backendaction._aux_get_bus_detailed_topo"""
        obs = self.env.reset()
        bk_act = self.env._backend_action
        # nothing modified
        loads_switches = bk_act.get_loads_bus_switches()
        assert loads_switches == []
        
        # I modified the position of a load
        bk_act += self.env.action_space({"set_bus": {"loads_id": [(1, 2)]}})
        loads_switches = bk_act.get_loads_bus_switches()
        assert loads_switches == [(1, (False, True))]  # modified load 1, first switch is opened (False) second one is closed (True)
        
        # I modified the position of a load
        bk_act += self.env.action_space({"set_bus": {"loads_id": [(1, 1)]}})
        loads_switches = bk_act.get_loads_bus_switches()
        assert loads_switches == [(1, (True, False))]  # modified load 1, first switch is closed (True) second one is opened (False) 
        
        # I disconnect a load
        bk_act += self.env.action_space({"set_bus": {"loads_id": [(1, -1)]}})
        loads_switches = bk_act.get_loads_bus_switches()
        assert loads_switches == [(1, (False, False))]  # modified load 1, first switch is closed (False) second one is opened (False) 
    
    def test_get_xxx_bus_switches(self):
        """test I can retrieve the switch of all the element types"""
        
        # generators
        obs = self.env.reset()
        bk_act = self.env._backend_action
        els_switches = bk_act.get_gens_bus_switches()
        assert els_switches == []
        bk_act += self.env.action_space({"set_bus": {"generators_id": [(1, 1)]}})
        els_switches = bk_act.get_gens_bus_switches()
        assert els_switches == [(1, (True, False))]  # modified gen 1, first switch is closed (True) second one is opened (False) 
        
        # line or
        obs = self.env.reset()
        bk_act = self.env._backend_action
        els_switches = bk_act.get_lines_or_bus_switches()
        assert els_switches == []
        bk_act += self.env.action_space({"set_bus": {"lines_or_id": [(1, 1)]}})
        els_switches = bk_act.get_lines_or_bus_switches()
        assert els_switches == [(1, (True, False))]  # modified line or 1, first switch is closed (True) second one is opened (False) 
        
        # line ex
        obs = self.env.reset()
        bk_act = self.env._backend_action
        els_switches = bk_act.get_lines_ex_bus_switches()
        assert els_switches == []
        bk_act += self.env.action_space({"set_bus": {"lines_ex_id": [(1, 1)]}})
        els_switches = bk_act.get_lines_ex_bus_switches()
        assert els_switches == [(1, (True, False))]  # modified line ex 1, first switch is closed (True) second one is opened (False) 
        
        # storage
        obs = self.env.reset()
        bk_act = self.env._backend_action
        els_switches = bk_act.get_storages_bus_switches()
        assert els_switches == []
        bk_act += self.env.action_space({"set_bus": {"storages_id": [(1, 1)]}})
        els_switches = bk_act.get_storages_bus_switches()
        assert els_switches == [(1, (True, False))]  # modified storage 1, first switch is closed (True) second one is opened (False) 
        
        # shunt
        obs = self.env.reset()
        bk_act = self.env._backend_action
        els_switches = bk_act.get_shunts_bus_switches()
        assert els_switches == []
        bk_act += self.env.action_space({"shunt": {"set_bus": [(0, 1)]}})
        els_switches = bk_act.get_shunts_bus_switches()
        assert els_switches == [(0, (True, False))]  # modified shunt 0, first switch is closed (True) second one is opened (False) 
        
    def test_compute_switches_position(self):
        obs = self.env.reset()
        busbar_connectors_state, switches_state = type(obs).detailed_topo_desc.compute_switches_position(obs.topo_vect, obs._shunt_bus)
        assert np.sum(switches_state) == 60
        assert switches_state[::2].all()  # all on bus 1
        assert (~switches_state[1::2]).all()  # nothing on busbar 2
        
        # move everything to bus 2
        busbar_connectors_state, switches_state = type(obs).detailed_topo_desc.compute_switches_position(np.full(obs.topo_vect.shape, fill_value=2),
                                                                                                         np.full(obs._shunt_bus.shape, fill_value=2))
        assert np.sum(switches_state) == 60
        assert switches_state[1::2].all()
        assert (~switches_state[::2]).all()
        
        # now check some disconnected elements (line id 0)
        topo_vect = 1 * obs.topo_vect
        topo_vect[type(obs).line_or_pos_topo_vect[0]] = -1
        topo_vect[type(obs).line_ex_pos_topo_vect[0]] = -1
        busbar_connectors_state, switches_state = type(obs).detailed_topo_desc.compute_switches_position(topo_vect, obs._shunt_bus)
        assert np.sum(switches_state) == 58
        assert switches_state[::2].sum() == 58
        assert switches_state[1::2].sum() == 0
        assert (~switches_state[type(obs).detailed_topo_desc.switches_to_topovect_id == type(obs).line_or_pos_topo_vect[0]]).all()
        assert (~switches_state[type(obs).detailed_topo_desc.switches_to_topovect_id == type(obs).line_ex_pos_topo_vect[0]]).all()
        
        # and now elements per elements
        # load 3 to bus 2
        topo_vect = 1 * obs.topo_vect
        topo_vect[type(obs).load_pos_topo_vect[3]] = 2
        busbar_connectors_state, switches_state = type(obs).detailed_topo_desc.compute_switches_position(topo_vect, obs._shunt_bus)
        assert np.sum(switches_state) == 60
        assert switches_state[::2].sum() == 59
        assert switches_state[1::2].sum() == 1
        assert not switches_state[type(obs).detailed_topo_desc.switches_to_topovect_id == type(obs).load_pos_topo_vect[3]][0]
        assert switches_state[type(obs).detailed_topo_desc.switches_to_topovect_id == type(obs).load_pos_topo_vect[3]][1]
        
        # gen 1 to bus 2
        topo_vect = 1 * obs.topo_vect
        topo_vect[type(obs).gen_pos_topo_vect[1]] = 2
        busbar_connectors_state, switches_state = type(obs).detailed_topo_desc.compute_switches_position(topo_vect, obs._shunt_bus)
        assert np.sum(switches_state) == 60
        assert switches_state[::2].sum() == 59
        assert switches_state[1::2].sum() == 1
        assert not switches_state[type(obs).detailed_topo_desc.switches_to_topovect_id == type(obs).gen_pos_topo_vect[1]][0]
        assert switches_state[type(obs).detailed_topo_desc.switches_to_topovect_id == type(obs).gen_pos_topo_vect[1]][1]
        
        # line or 6 to bus 2
        topo_vect = 1 * obs.topo_vect
        el_id = 6
        topo_vect[type(obs).line_or_pos_topo_vect[el_id]] = 2
        busbar_connectors_state, switches_state = type(obs).detailed_topo_desc.compute_switches_position(topo_vect, obs._shunt_bus)
        assert np.sum(switches_state) == 60
        assert switches_state[::2].sum() == 59
        assert switches_state[1::2].sum() == 1
        assert not switches_state[type(obs).detailed_topo_desc.switches_to_topovect_id == type(obs).line_or_pos_topo_vect[el_id]][0]
        assert switches_state[type(obs).detailed_topo_desc.switches_to_topovect_id == type(obs).line_or_pos_topo_vect[el_id]][1]
        
        # line ex 9 to bus 2
        topo_vect = 1 * obs.topo_vect
        el_id = 9
        topo_vect[type(obs).line_ex_pos_topo_vect[el_id]] = 2
        busbar_connectors_state, switches_state = type(obs).detailed_topo_desc.compute_switches_position(topo_vect, obs._shunt_bus)
        assert np.sum(switches_state) == 60
        assert switches_state[::2].sum() == 59
        assert switches_state[1::2].sum() == 1
        assert not switches_state[type(obs).detailed_topo_desc.switches_to_topovect_id == type(obs).line_ex_pos_topo_vect[el_id]][0]
        assert switches_state[type(obs).detailed_topo_desc.switches_to_topovect_id == type(obs).line_ex_pos_topo_vect[el_id]][1]
        
        # storage 0 to bus 2
        topo_vect = 1 * obs.topo_vect
        el_id = 0
        topo_vect[type(obs).storage_pos_topo_vect[el_id]] = 2
        busbar_connectors_state, switches_state = type(obs).detailed_topo_desc.compute_switches_position(topo_vect, obs._shunt_bus)
        assert np.sum(switches_state) == 60
        assert switches_state[::2].sum() == 59
        assert switches_state[1::2].sum() == 1
        assert not switches_state[type(obs).detailed_topo_desc.switches_to_topovect_id == type(obs).storage_pos_topo_vect[el_id]][0]
        assert switches_state[type(obs).detailed_topo_desc.switches_to_topovect_id == type(obs).storage_pos_topo_vect[el_id]][1]
        
        # shunt 0 to bus 2
        shunt_bus = 1 * obs._shunt_bus
        el_id = 0
        shunt_bus[el_id] = 2
        busbar_connectors_state, switches_state = type(obs).detailed_topo_desc.compute_switches_position(obs.topo_vect, shunt_bus)
        assert np.sum(switches_state) == 60
        assert switches_state[::2].sum() == 59
        assert switches_state[1::2].sum() == 1
        assert not switches_state[type(obs).detailed_topo_desc.switches_to_shunt_id == el_id][0]
        assert switches_state[type(obs).detailed_topo_desc.switches_to_shunt_id == el_id][1]
        
    # TODO detailed topo
        
 # TODO test no shunt too
 # TODO implement and test compute_switches_position !!!
 
if __name__ == "__main__":
    unittest.main()
   