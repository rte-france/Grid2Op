# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import re
import numpy as np


import grid2op
from grid2op.Action import ActionSpace, BaseAction, PlayableAction
from grid2op.Exceptions import IllegalAction, SimulateError
from grid2op.Observation import CompleteObservation
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from grid2op.multi_agent.multiAgentEnv import MultiAgentEnv
        from grid2op.multi_agent.subgridAction import SubGridAction
        from grid2op.multi_agent.ma_exceptions import DomainException
        from grid2op.multi_agent.subgridObservation import SubGridObservation
except ImportError as exc_:
    print(f"Impossible to load with error {exc_}")
    raise

import pdb

def _aux_sample_withtout_interco(act_sp: ActionSpace):
    res: SubGridAction = act_sp.sample()
    if res._modif_interco_set_status or res._modif_interco_change_status:
        # if the action sample the interconnection, i resample it
        res = _aux_sample_withtout_interco(act_sp)
    return res
    

def _aux_sample_without_interco_from_global(global_act_sp: ActionSpace,
                                            local_act_spaces):
    res: BaseAction = global_act_sp.sample()
    if res._modif_set_status:
        # if the action sample the interconnection, i resample it
        if np.any(res._set_line_status[local_act_spaces["agent_0"].mask_interco] != 0) or np.any(res._set_line_status[local_act_spaces["agent_1"].mask_interco] != 0):
            res = _aux_sample_without_interco_from_global(global_act_sp, local_act_spaces)
    elif res._modif_change_status:
        # if the action sample the interconnection, i resample it
        if np.any(res._switch_line_status[local_act_spaces["agent_0"].mask_interco]) or np.any(res._switch_line_status[local_act_spaces["agent_1"].mask_interco]):
            res = _aux_sample_without_interco_from_global(global_act_sp, local_act_spaces)
    return res
    
    
class MATesterGlobalObs(unittest.TestCase):
    def setUp(self) -> None:
        
        self.action_domains = {
            'agent_0' : [0, 1, 2, 3, 4],
            'agent_1' : [5, 6, 7, 8, 9, 10, 11, 12, 13]
        }
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    action_class=PlayableAction,
                                    _add_to_name=type(self).__name__)

        
            self.ma_env = MultiAgentEnv(self.env, self.action_domains)
            self.ma_env.seed(0)
            self.ma_env.reset()
            
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_verify_domains(self):
        """test the MultiAgentEnv._verify_domains method """
        # it should test that :
        # 1) the function works (does not throw an error) when the input domains are correct
        # 2) the function throws an error when the input domains are wrong
        # (the more "wrong" cases tested the better)
        
        action_domains = {
            'agent_0' : 0,
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        # above action domain should raise an error: "0" is not an iterable !
        with self.assertRaises(DomainException) as de:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                MultiAgentEnv(self.env, action_domains, _add_to_name="test_verify_domains_0")
        
        action_domains = {
            'agent_0' : [0],
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        # above action domain should raise an error: substations are not fully allocated !
        with self.assertRaises(DomainException) as de:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                MultiAgentEnv(self.env, action_domains, _add_to_name="test_verify_domains_1")
            
        action_domains = {
            'agent_0' : [],
            'agent_1' : list(range(self.env.n_sub))
        }
        # above action domain should raise an error: agents must have at least one substation !
        with self.assertRaises(DomainException) as de:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                MultiAgentEnv(self.env, action_domains, _add_to_name="test_verify_domains_2")
            
        action_domains = {
            'agent_0' : [0,1,6,3, 4],
            'agent_1' : [5,2,7,8,9,10,11,12,13]
        }
        # this domain is valid even if it is not connected
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                MultiAgentEnv(self.env, action_domains, _add_to_name="test_verify_domains_3")
        except DomainException:
            self.fail("action_domains raised Domain Exception unexpectedly!")
            
        action_domains = {
            'agent_0' : [0,1,6,3, 4, 5],
            'agent_1' : [5,2,7,8,9,10,11,12,13]
        }
        # this domain is not a partition ; it should raise an error
        with self.assertRaises(DomainException) as de:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                MultiAgentEnv(self.env, action_domains, _add_to_name="test_verify_domains_4")
        
    
    def test_build_subgrids_action_domains(self):
        # Simple test to verify if action domains are correctly
        # taken into accaount by the env
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.ma_env = MultiAgentEnv(self.env,
                                        self.action_domains,
                                        _add_to_name="test_build_subgrids_action_domains")
        
        assert self.ma_env._action_domains['agent_0']['sub_id'] == self.action_domains['agent_0']
        assert self.ma_env._action_domains['agent_1']['sub_id'] == self.action_domains['agent_1']
    
    def test_masks(self):
        # We compare the masks with known values for every agent
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.ma_env = MultiAgentEnv(self.env, self.action_domains, _add_to_name="test_masks")
            
        mask_load_agent0 = np.array([True,  True,  True,  True, False, False, False, 
                            False, False, False, False])
        # We compare the load masks with known values for every agent
        assert (self.ma_env._action_domains['agent_0']['mask_load'] == mask_load_agent0).all()
        assert (self.ma_env._action_domains['agent_1']['mask_load'] == ~mask_load_agent0).all()
        
        mask_gen_agent0 = np.array([ True,  True, False, False, False,  True])
        # We compare the generator masks with known values for every agent
        assert (self.ma_env._action_domains['agent_0']['mask_gen'] == mask_gen_agent0).all()
        assert (self.ma_env._action_domains['agent_1']['mask_gen'] == ~mask_gen_agent0).all()
        # We compare the storage masks with known values for every agent
        mask_storage = np.array([False, False])
        assert (self.ma_env._action_domains['agent_0']['mask_storage'] == mask_storage).all()
        assert (self.ma_env._action_domains['agent_1']['mask_storage'] == ~mask_storage).all()
        
        mask_line_agent0 = np.array([ True,  True,  True,  True,  True,  True,  True, 
                                False, False,False, False, False, False, False, False, 
                                False, False, False,False, False])
        # We compare the line_ex masks with known values for every agent
        assert (self.ma_env._action_domains['agent_0']['mask_line'] == mask_line_agent0).all()
        mask_line_agent1 = np.array([False, False, False, False, False, False, False,
                                        True,  True,  True, True,  True, True,  True,  True,
                                        False, False, False,  True,  True])
        assert (self.ma_env._action_domains['agent_1']['mask_line'] == mask_line_agent1).all()
        # We compare the line_or masks with known values for every agent
        assert (self.ma_env._action_domains['agent_0']['mask_line'] == mask_line_agent0).all()
        assert (self.ma_env._action_domains['agent_1']['mask_line'] == mask_line_agent1).all()
        
        mask_shunt_agent0 = np.array([False])
        # We compare the shunt masks with known values for every agent
        assert (self.ma_env._action_domains['agent_0']['mask_shunt'] == mask_shunt_agent0).all()
        assert (self.ma_env._action_domains['agent_1']['mask_shunt'] == ~mask_shunt_agent0).all()
        
    def test_interco(self):
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.ma_env = MultiAgentEnv(self.env, self.action_domains, _add_to_name="test_interco")
        # Tests on interconnections with known values for every agent
        mask_interco_ref = np.array([False, False, False, False, False, False, False, False, False,
                                False, False, False, False, False, False,  True,  True,  True,
                                False, False])
        # We compare the interconnection masks with known values for every agent
        assert (self.ma_env._action_domains['agent_0']['mask_interco'] == mask_interco_ref).all()
        assert (self.ma_env._action_domains['agent_1']['mask_interco'] == mask_interco_ref).all()
        interco_is_origin_agent0 = np.array([True, True, True])
        # We compare the interco_is_origin masks with known values for every agent
        # interco_is_origin tells whether the corresponding interco is a line_or or not
        assert (self.ma_env._action_domains['agent_0']['interco_is_origin'] == interco_is_origin_agent0).all()
        assert (self.ma_env._action_domains['agent_1']['interco_is_origin'] == ~interco_is_origin_agent0).all()
        
        # In the two-agent case, they must have the same number of interconnections
        assert self.ma_env._subgrids_cls['action']['agent_0'].n_interco == self.ma_env._subgrids_cls['action']['agent_1'].n_interco
        
        # A line in the original grid is either a line in 
        # one subgrid (if both its end are in the subdomain) or an interconnection. 
        # By summing all lines in both subdomains and the interconnection, 
        # we should get the total number of lines.
        assert self.ma_env._subgrids_cls['action']['agent_0'].n_line \
             + self.ma_env._subgrids_cls['action']['agent_1'].n_line \
             + self.ma_env._subgrids_cls['action']['agent_0'].n_interco == self.env.n_line
                

    def test_build_subgrid_obj(self):
        """test the MultiAgentEnv._build_subgrid_obj_from_domain"""
        
        # 1
        action_domains = {
            'agent_0' : [0, 1, 2, 3, 4],
            'agent_1' : [5, 6, 7, 8, 9, 10, 11, 12, 13]
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ma_env = MultiAgentEnv(self.env, action_domains, _add_to_name="test_build_subgrid_obj")
        
        # We compare the number of generators for every agents' subgrids with known values
        assert ma_env._subgrids_cls['action']['agent_0'].n_gen == 3
        assert ma_env._subgrids_cls['action']['agent_1'].n_gen == 3
        
        self.check_subgrid_consistency(ma_env, action_domains)
        assert ma_env.agents == ['agent_0', 'agent_1']
        assert ma_env.agent_order == ma_env.agents
        # We compare the interconnection original line ids with known values for every agent
        interco_lineid_ref = np.array([15,16,17])
        assert (ma_env._subgrids_cls['action']['agent_0'].interco_to_lineid == interco_lineid_ref).all()
        assert (ma_env._subgrids_cls['action']['agent_1'].interco_to_lineid == interco_lineid_ref).all()
        
        ref = np.array([[ 0., -1., -1.,  0., -1., -1, -1.],
                        [ 0., -1., -1.,  1., -1., -1, -1.],
                        [ 0., -1.,  2., -1., -1., -1, -1.],
                        [ 1., -1., -1., -1.,  0., -1, -1.],
                        [ 1., -1., -1.,  2., -1., -1, -1.],
                        [ 1., -1., -1.,  3., -1., -1, -1.],
                        [ 1., -1., -1.,  4., -1., -1, -1.],
                        [ 1., -1.,  0., -1., -1., -1, -1.],
                        [ 1.,  0., -1., -1., -1., -1, -1.],
                        [ 2., -1., -1., -1.,  2., -1, -1.],
                        [ 2., -1., -1.,  5., -1., -1, -1.],
                        [ 2., -1.,  1., -1., -1., -1, -1.],
                        [ 2.,  1., -1., -1., -1., -1, -1.],
                        [ 3., -1., -1., -1.,  3., -1, -1.],
                        [ 3., -1., -1., -1.,  5., -1, -1.],
                        [ 3., -1., -1.,  6., -1., -1, -1.],
                        [ 3., -1., -1., -1., -1., -1,  0.],
                        [ 3., -1., -1., -1., -1., -1,  1.],
                        [ 3.,  2., -1., -1., -1., -1, -1.],
                        [ 4., -1., -1., -1.,  1., -1, -1.],
                        [ 4., -1., -1., -1.,  4., -1, -1.],
                        [ 4., -1., -1., -1.,  6., -1, -1.],
                        [ 4., -1., -1., -1., -1., -1,  2.],
                        [ 4.,  3., -1., -1., -1., -1, -1.]])
        # We compare with a known value
        assert (ma_env._subgrids_cls['action']['agent_0'].grid_objects_types == ref).all()
        
        # test the observation are complete (rapid tests)
        assert ma_env._subgrids_cls['observation']['agent_0'].n_interco == 0
        assert ma_env._subgrids_cls['observation']['agent_1'].n_interco == 0
        
    def test_build_subgrid_obj2(self):    
        # 2
        # Test with 3 agents
        action_domains = {
            'test_2_agent_0' : [0,1, 2, 3, 4],
            'test_2_agent_2' : [10, 11, 12, 13],
            'test_2_agent_1' : [5, 6, 7, 8, 9],
            
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ma_env = MultiAgentEnv(self.env, action_domains, _add_to_name="test_build_subgrid_obj2")
        assert ma_env.agents == ['test_2_agent_0', 'test_2_agent_1', 'test_2_agent_2']
        assert ma_env.agents != ['test_2_agent_0', 'test_2_agent_2', 'test_2_agent_1']
        assert ma_env.agent_order == ma_env.agents
        self.check_subgrid_consistency(ma_env, action_domains)

    
    def test_build_subgrid_obj3(self):    
        # 3 random sub ids
        np.random.seed(0)
        for it in range(10):
            sub_ids = list(range(14))
            np.random.shuffle(sub_ids)  # you should see it for reproducible results
            pivot = np.random.randint(low=1, high=13)
            action_domains = {
                'agent_0' : sub_ids[:pivot],
                'agent_1' : sub_ids[pivot:],
            }
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                ma_env = MultiAgentEnv(self.env,
                                       action_domains,
                                       _add_to_name=f"_test_build_subgrid_obj3_it_{it}")
            assert ma_env.agents == ['agent_0', 'agent_1']
            assert ma_env.agent_order == ma_env.agents
            self.check_subgrid_consistency(ma_env, action_domains, add_msg=f"error for iter {it}")
            
            
    def check_subgrid_consistency(self, ma_env, action_domains, add_msg=""):
        # Regroups all the checks to be done
        self.check_orig_ids(ma_env, action_domains)
        self.check_n_objects(ma_env, action_domains, add_msg=add_msg)
        self.check_objects_to_subid(ma_env, action_domains)
        self.check_connections(ma_env, action_domains)
        self.check_shunt(ma_env)
        self.check_mask_topo_vect(ma_env, action_domains)
        self.check_action_spaces(ma_env)
        self.check_reset(ma_env)
        self.check_dispatch_reward_done_info(ma_env)
        
    def check_reset(self, ma_env):
        ma_env.reset()
        first_obs = None
        for agent in ma_env.agents:
            # We check if the _cent_observation is copied and not pointed
            assert ma_env.observations[agent] is not first_obs
            if first_obs is None:
                first_obs = ma_env.observations[agent]
            # We check if observations have same values
            assert ma_env.observations[agent] == first_obs
            
    def check_dispatch_reward_done_info(self, ma_env):
        reward = 42.
        done = False
        info = {'test' : True}
        ma_env._dispatch_reward_done_info(reward, done, info)

        for agent in ma_env.agents:
            # We check if rewards have same values
            assert ma_env.rewards[agent] == reward
            # We check if dones have same values
            assert ma_env.done[agent] == done
            # We check if infos have same values
            assert ma_env.info[agent] == info
            # We check if infos are copied and not pointed
            assert ma_env.info[agent] is not info

    
    def check_n_objects(self, ma_env, domain, space = 'action', add_msg = ""):
        # Check the number of objects in subgrids. The sum must be equal 
        # to the number in the global grid
        
        # Tests if the sum of generator numbers in subgrids is equal to the number of  
        # generators in the original grid.
        assert np.sum([ma_env._subgrids_cls[space][a].n_gen for a in domain.keys()]) == self.env.n_gen, add_msg
        # Tests if the sum of load numbers in subgrids is equal to the number of  
        # load in the original grid.
        assert np.sum([ma_env._subgrids_cls[space][a].n_load for a in domain.keys()]) == self.env.n_load, add_msg
        # Tests if the sum of shunt numbers in subgrids is equal to the number of  
        # shunt in the original grid.
        assert np.sum([ma_env._subgrids_cls[space][a].n_shunt for a in domain.keys()]) == self.env.n_shunt, add_msg
        # Tests if the sum of storage numbers in subgrids is equal to the number of  
        # storage in the original grid.
        assert np.sum([ma_env._subgrids_cls[space][a].n_storage for a in domain.keys()]) == self.env.n_storage, add_msg
        # Tests if the sum of line numbers and interco numbers divided by 2 (since they are counted twice) 
        # in subgrids is equal to the number of lines in the original grid.
        assert np.sum([ma_env._subgrids_cls[space][a].n_line for a in domain.keys()])\
             + np.sum([ma_env._subgrids_cls[space][a].n_interco for a in domain.keys()])/2\
            ==\
                self.env.n_line, add_msg
        
        for agent in domain.keys():
            # We check that local sub_info has good number of objects
            assert np.sum(ma_env._subgrids_cls[space][agent].sub_info)\
                ==\
                ma_env._subgrids_cls[space][agent].n_gen+\
                ma_env._subgrids_cls[space][agent].n_load+\
                ma_env._subgrids_cls[space][agent].n_line*2+\
                ma_env._subgrids_cls[space][agent].n_interco+\
                ma_env._subgrids_cls[space][agent].n_storage, add_msg
        
            # The number of line_ex/line_or should be equal to the number of lines
            assert len(ma_env._subgrids_cls[space][agent].line_ex_to_subid) == ma_env._subgrids_cls[space][agent].n_line, add_msg
            assert len(ma_env._subgrids_cls[space][agent].line_or_to_subid) == ma_env._subgrids_cls[space][agent].n_line, add_msg
        
        
    def check_objects_to_subid(self, ma_env, domain, space = 'action'):
        
        # Verifies if sub ids are correct   
        for agent in domain.keys():
            
            # Check if load to sub ids are smaller than the number of substations
            assert (ma_env._subgrids_cls[space][agent].load_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            # Check if line_or to sub ids are smaller than the number of substations
            assert (ma_env._subgrids_cls[space][agent].line_or_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            # Check if line_ex to sub ids are smaller than the number of substations
            assert (ma_env._subgrids_cls[space][agent].line_ex_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            # Check if storage to sub ids are smaller than the number of substations
            assert (ma_env._subgrids_cls[space][agent].storage_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            # Check if gen to sub ids are smaller than the number of substations
            assert (ma_env._subgrids_cls[space][agent].gen_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            # Check if interco to sub ids are smaller than the number of substations
            assert (ma_env._subgrids_cls[space][agent].interco_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            if ma_env._subgrids_cls[space][agent].n_shunt:
                # Check if shunt to sub ids are smaller than the number of substations
                assert (ma_env._subgrids_cls[space][agent].shunt_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
                
            
            for subid in range(ma_env._subgrids_cls[space][agent].n_sub):
                dict_connected_objects = ma_env._subgrids_cls[space][agent].get_obj_connect_to(substation_id=subid)
                # Check if loads are correctly connected to the same substation
                # as it is given by load_to_subid
                assert (np.where(ma_env._subgrids_cls[space][agent].load_to_subid == subid)\
                    ==\
                    dict_connected_objects["loads_id"]).all()
                # Check if generators are correctly connected to the same substation
                # as it is given by gen_to_subid
                assert (np.where(ma_env._subgrids_cls[space][agent].gen_to_subid == subid)\
                    ==\
                    dict_connected_objects["generators_id"]).all()
                # Check if origin lines are correctly connected to the same substation
                # as it is given by line_or_to_subid
                assert (np.where(ma_env._subgrids_cls[space][agent].line_or_to_subid == subid)\
                    ==\
                    dict_connected_objects["lines_or_id"]).all()
                # Check if extremity lines are correctly connected to the same substation
                # as it is given by line_ex_to_subid
                assert (np.where(ma_env._subgrids_cls[space][agent].line_ex_to_subid == subid)\
                    ==\
                    dict_connected_objects["lines_ex_id"]).all()
                # Check if storages are correctly connected to the same substation
                # as it is given by storage_to_subid
                assert (np.where(ma_env._subgrids_cls[space][agent].storage_to_subid == subid)\
                    ==\
                    dict_connected_objects["storages_id"]).all()
            
    def check_orig_ids(self, ma_env, domain : dict, space = 'action'):
        
        # It tests if the sub_orig_ids give the set of sub ids
        # by concatenating and sorting them
        assert (np.sort(np.concatenate([
            ma_env._subgrids_cls[space][agent].sub_orig_ids 
            for agent in domain.keys()])) == np.arange(ma_env._cent_env.n_sub)).all()
        
        for agent in domain.keys():
            mask_load = ma_env._subgrids_cls[space][agent].mask_load
            mask_gen = ma_env._subgrids_cls[space][agent].mask_gen
            mask_storage = ma_env._subgrids_cls[space][agent].mask_storage
            mask_line = ma_env._subgrids_cls[space][agent].mask_line
            mask_shunt = ma_env._subgrids_cls[space][agent].mask_shunt
            mask_interco = ma_env._subgrids_cls[space][agent].mask_interco
            # It tests if the sub_orig_ids are correct and
            # equal to the domain given 
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids == np.sort(domain[agent])).all()
            # We check that we have the correct generators original ids
            # Ids should be the same as those given by masks 
            assert (ma_env._subgrids_cls[space][agent].gen_orig_ids\
                ==\
                    np.arange(ma_env._cent_env.n_gen)[mask_gen]).all()
            # We check that we have the correct loads original ids
            # Ids should be the same as those given by masks 
            assert (ma_env._subgrids_cls[space][agent].load_orig_ids\
                ==\
                    np.arange(ma_env._cent_env.n_load)[mask_load]).all()
            # We check that we have the correct storage original ids
            # Ids should be the same as those given by masks 
            assert (ma_env._subgrids_cls[space][agent].storage_orig_ids\
                ==\
                    np.arange(ma_env._cent_env.n_storage)[mask_storage]).all()
            # We check that we have the correct lines original ids
            # Ids should be the same as those given by line_or masks 
            assert (ma_env._subgrids_cls[space][agent].line_orig_ids\
                ==\
                    np.arange(ma_env._cent_env.n_line)[mask_line]).all()
            # We check that we have the correct generators original ids
            # Ids should be the same as those given by line_ex masks 
            assert (ma_env._subgrids_cls[space][agent].line_orig_ids\
                ==\
                    np.arange(ma_env._cent_env.n_line)[mask_line]).all()
            
            if ma_env._subgrids_cls[space][agent].n_shunt > 0:
                # We check that we have the correct shunt original ids
                # Ids should be the same as those given by masks 
                assert (ma_env._subgrids_cls[space][agent].shunt_orig_ids\
                    ==\
                        np.arange(ma_env._cent_env.n_shunt)[mask_shunt]).all()
            # We check that we have the correct interconnections original ids
            # Ids should be the same as those given by masks     
            assert (ma_env._subgrids_cls[space][agent].interco_to_lineid\
                ==\
                    np.arange(ma_env._cent_env.n_line)[mask_interco]).all()
            
    def check_connections(self, ma_env, domain, space = 'action'):
        # We check if the objects are connected to same subids
        # in local/global grids and vice-versa.
        
        for agent in domain.keys():
            # Assert that the local sub_info is equal to the global sub_info
            # in the same substations 
            assert (ma_env._subgrids_cls[space][agent].sub_info\
                ==\
                    ma_env._cent_env.sub_info[
                        ma_env._subgrids_cls[space][agent].sub_orig_ids
                    ]).all()
            
            # We check if a load is on a substation on the subgrid,
            # it is also on the original grid
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].load_to_subid]\
                ==\
                    ma_env._cent_env.load_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_load
                    ]).all()
            # We check if a generator is on a substation on the subgrid,
            # it is also on the original grid
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].gen_to_subid]\
                ==\
                    ma_env._cent_env.gen_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_gen
                    ]).all()
            # We check if a storage is on a substation on the subgrid,
            # it is also on the original grid
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].storage_to_subid]\
                ==\
                    ma_env._cent_env.storage_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_storage
                    ]).all()
            # We check if a shunt is on a substation on the subgrid,
            # it is also on the original grid
            # This test didn't pass with the previous version
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].shunt_to_subid]\
                ==\
                    ma_env._cent_env.shunt_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_shunt
                    ]).all()
            # We check if a line_ex is on a substation on the subgrid,
            # it is also on the original grid
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].line_ex_to_subid]\
                ==\
                    ma_env._cent_env.line_ex_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_line
                    ]).all()
            # We check if a line_or is on a substation on the subgrid,
            # it is also on the original grid
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].line_or_to_subid]\
                ==\
                    ma_env._cent_env.line_or_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_line
                    ]).all()
            # We check if an interconnection is on a substation on the subgrid,
            # it is also on the original grid (line_or)
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].interco_to_subid][
                            ma_env._subgrids_cls[space][agent].interco_is_origin]\
                ==\
                    ma_env._cent_env.line_or_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_interco
                    ][
                        ma_env._subgrids_cls[space][agent].interco_is_origin
                ]).all()
            # We check if an interconnection is on a substation on the subgrid,
            # it is also on the original grid (line_ex)
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].interco_to_subid][
                            ~ma_env._subgrids_cls[space][agent].interco_is_origin]\
                ==\
                    ma_env._cent_env.line_ex_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_interco
                    ][
                        ~ma_env._subgrids_cls[space][agent].interco_is_origin
                ]).all()
            
        # we check that if 2 elements are on the same substation in the original grid, 
        # they are on the same substation in the corresponding subgrid
        for sub_origin_id in range(ma_env._cent_env.n_sub):
            
            agents = [k for k, v in domain.items() if sub_origin_id in v]
            # The corresponding sustation must be only in one domain
            # i.e. no intersection between domains
            assert len(agents) == 1
            agent = agents[0]
            
            subids = np.where(ma_env._subgrids_cls[space][agent].sub_orig_ids == sub_origin_id)
            # domains must not have duplicates
            assert len(subids) == 1
            subid = subids[0]
            
            # We extract the local and global connection dicts
            dict_local = ma_env._subgrids_cls[space][agent].get_obj_connect_to(substation_id=subid)
            dict_global = self.env.get_obj_connect_to(
                substation_id=sub_origin_id
            )
            # We check if we have the same number of elements
            assert dict_local['nb_elements'] == dict_global['nb_elements']
            # We check if the original ids of loads in the local dict
            # are the same as the loads in the global dict 
            assert (dict_global['loads_id']\
                ==\
                    ma_env._subgrids_cls[space][agent].load_orig_ids[dict_local['loads_id']]
            ).all()
            # We check if the original ids of generators in the local dict
            # are the same as the generators in the global dict 
            assert (dict_global['generators_id']\
                ==\
                    ma_env._subgrids_cls[space][agent].gen_orig_ids[dict_local['generators_id']]
            ).all()
            # We check if the original ids of storages in the local dict
            # are the same as the storages in the global dict 
            assert (dict_global['storages_id']\
                ==\
                    ma_env._subgrids_cls[space][agent].storage_orig_ids[dict_local['storages_id']]
            ).all()
            # We check if the original ids of line_or in the local dict
            # are the same as the line_or in the global dict 
            # For lines, we take into account interconnections based on their nature (line_or or line_ex)
            assert (dict_global['lines_or_id']\
                ==\
                    np.sort(np.concatenate((
                        ma_env._subgrids_cls[space][agent].line_orig_ids[dict_local['lines_or_id']],
                        ma_env._subgrids_cls[space][agent].interco_to_lineid[dict_local['intercos_id']][
                            ma_env._subgrids_cls[space][agent].interco_is_origin[dict_local['intercos_id']]
                        ]
                    )))
            ).all()
            # We check if the original ids of line_ex in the local dict
            # are the same as the line_ex in the global dict 
            # For lines, we take into account interconnections based on their nature (line_or or line_ex)
            assert (dict_global['lines_ex_id']\
                ==\
                    np.sort(np.concatenate((
                        ma_env._subgrids_cls[space][agent].line_orig_ids[dict_local['lines_ex_id']],
                        ma_env._subgrids_cls[space][agent].interco_to_lineid[dict_local['intercos_id']][
                            ~ma_env._subgrids_cls[space][agent].interco_is_origin[dict_local['intercos_id']]
                        ]
                    )))
            ).all()
                
    def check_shunt(self, ma_env, space = "action"):
        # Test for shunts
        for agent in ma_env.agents:
            assert ma_env._subgrids_cls[space][agent].shunts_data_available \
                ==\
                    self.env.shunts_data_available 
                    
            if ma_env._subgrids_cls[space][agent].n_shunt > 0:
                # We check that we have the correct shunt original ids
                # Ids should be the same as those given by masks 
                assert (ma_env._subgrids_cls[space][agent].shunt_orig_ids\
                    ==\
                        np.arange(ma_env._cent_env.n_shunt)[
                            ma_env._subgrids_cls[space][agent].mask_shunt
                        ]
                ).all()
                # Check if shunt to sub ids are smaller than the number of substations
                assert (ma_env._subgrids_cls[space][agent].shunt_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            # This test didn't pass with the previous version
            # We check if the shunt is connected to the same
            # substation in both local and global grids
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].shunt_to_subid]\
                ==\
                    ma_env._cent_env.shunt_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_shunt
                    ]).all()
    
    def check_mask_topo_vect(self, ma_env, domain, space="action"):
        for agent in domain.keys():
            subgrid_cls = ma_env._subgrids_cls[space][agent]
            orig_grid_cls = type(ma_env._cent_env)
            mask_orig_pos_topo_vect = subgrid_cls.mask_orig_pos_topo_vect
            if np.all(mask_orig_pos_topo_vect):
                raise AssertionError("Some agent would have all the grid")
            if np.all(~mask_orig_pos_topo_vect):
                raise AssertionError("Some agent would have all the grid")
            
            assert mask_orig_pos_topo_vect.size == orig_grid_cls.dim_topo, "mask do not corresponds to original grid size"
            
            if np.any(~mask_orig_pos_topo_vect[orig_grid_cls.load_pos_topo_vect[subgrid_cls.load_orig_ids]]):
                raise AssertionError("some loads are deactivated in the mask pos topo vect")
    
            if np.any(~mask_orig_pos_topo_vect[orig_grid_cls.gen_pos_topo_vect[subgrid_cls.gen_orig_ids]]):
                raise AssertionError("some gens are deactivated in the mask pos topo vect")
    
            if np.any(~mask_orig_pos_topo_vect[orig_grid_cls.storage_pos_topo_vect[subgrid_cls.storage_orig_ids]]):
                raise AssertionError("some gens are deactivated in the mask pos topo vect")
    
            if np.any(~mask_orig_pos_topo_vect[orig_grid_cls.line_or_pos_topo_vect[subgrid_cls.line_orig_ids]]):
                raise AssertionError("some line or are deactivated in the mask pos topo vect")
            
            if np.any(~mask_orig_pos_topo_vect[orig_grid_cls.line_ex_pos_topo_vect[subgrid_cls.line_orig_ids]]):
                raise AssertionError("some line ex are deactivated in the mask pos topo vect")
            
            interco_pos_topo_vect = orig_grid_cls.line_or_pos_topo_vect[subgrid_cls.interco_to_lineid]
            interco_pos_topo_vect[~subgrid_cls.interco_is_origin] = orig_grid_cls.line_ex_pos_topo_vect[subgrid_cls.interco_to_lineid][~subgrid_cls.interco_is_origin]
            if np.any(~mask_orig_pos_topo_vect[interco_pos_topo_vect]):
                raise AssertionError("some interco are deactivated in the mask pos topo vect")
            
    
    def check_action_spaces(self, ma_env):
        # This function checks if the action space is correctly
        # created from _subgrids_cls['action']
        # TODO more tests 
        for agent in ma_env.agents:
            # We test if they have the same topological dimension
            assert ma_env.action_spaces[agent].dim_topo\
                ==\
                    ma_env._subgrids_cls['action'][agent].dim_topo
                    
            # It checks if the number of True values in the mask is
            # equal to the length of actions' _set_topo_vect
            assert len(ma_env.action_spaces[agent]({})._set_topo_vect)\
                ==\
                    np.sum(ma_env._subgrids_cls['action'][agent].mask_orig_pos_topo_vect)
            
            for object in ['load', 'gen', 'line_or', 'line_ex', 'storage', 'interco']:
                
                # Checks if they both have the same object_pos_topo_vect
                assert (getattr(ma_env.action_spaces[agent], f'{object}_pos_topo_vect')\
                    ==\
                        getattr(ma_env._subgrids_cls['action'][agent], f'{object}_pos_topo_vect')).all()
            
        
    def are_same_actions(self, a1, a2, type_action, 
                         add_msg = ""):
        modif_type_action = f'_modif_{type_action}'
        # Checks both actions have the same action type
        # meaning if the same action is done, the array
        # representing this action is the same in both
        # actions.
        assert (getattr(a1, type_action) == getattr(a2, type_action)).all(), add_msg
        # Chacks if the same action type is applied 
        assert getattr(a1, modif_type_action) == getattr(a2, modif_type_action), add_msg
        # Checks if the actions are the same
        # meaning they have the same effect
        # on the grid
        assert a1 == a2, add_msg
        
        
    def check_local2global(self, agent, n, orig_ids, type_action, is_set, value,
                           ):
        # We check for every local id if _local_action_to_global 
        # gives the correspondig global action on the same object
        # and they have the same effect.
        
        # We check if lengths match 
        assert n == len(orig_ids)
        
        for local_id in range(n):
            
            local_act = self.ma_env.action_spaces[agent]({})
            if is_set:
                # We create the local action
                setattr(local_act, type_action, [(local_id, value)])
            else:
                # We create the local action
                setattr(local_act, type_action, [local_id])
            
            global_act = self.ma_env._local_action_to_global(local_act)
            global_id = orig_ids[local_id]
            
            ref_global_act = self.ma_env._cent_env.action_space({})
            # if an action is of type set (like set_bus, redispatch, ...)
            # we give the tuple (id, value)
            # otherwise we give only the id
            if is_set:
                # We create the global action
                setattr(ref_global_act, type_action, [(global_id, value)])
            else:
                # We create the global action
                setattr(ref_global_act, type_action, [global_id])
            
            self.are_same_actions(
                global_act, ref_global_act, 'set_bus',
                add_msg = f"""
                    agent : {agent}, 
                    local id : {local_id}, 
                    global_id : {global_id},
                    action type : {type_action}
                """
            )
                    
        
    def test_local_action_to_global_set_bus(self):
        # tests for set bus actions
        
        # The global env must have at least one storage
        assert self.ma_env._cent_env.n_storage > 0
        
        for agent in self.ma_env.agents:
            
            # Test for loads
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_load,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].load_orig_ids,
                type_action = 'load_set_bus',
                is_set = True,
                value = 2
            )

            
            # Test for gens
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_gen,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].gen_orig_ids,
                type_action = 'gen_set_bus',
                is_set = True,
                value = 2
            )
            
            # Test for line_ors
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_line,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].line_orig_ids,
                type_action = 'line_or_set_bus',
                is_set = True,
                value = 2
            )
            
            # Test for line_exs
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_line,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].line_orig_ids,
                type_action = 'line_ex_set_bus',
                is_set = True,
                value = 2
            )
            
            
            # Test for storages
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_storage,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].storage_orig_ids,
                type_action = 'storage_set_bus',
                is_set = True,
                value = 2
            )


    def test_local_action_to_global_change_bus(self):
        # tests for change bus actions
        
        # The global env must have at least one storage
        assert self.ma_env._cent_env.n_storage > 0
        
        for agent in self.ma_env.agents:
            
            # Test for loads
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_load,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].load_orig_ids,
                type_action = 'load_change_bus',
                is_set = False,
                value = 2
            )

            
            # Test for gens
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_gen,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].gen_orig_ids,
                type_action = 'gen_change_bus',
                is_set = False,
                value = 2
            )
            
            # Test for line_ors
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_line,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].line_orig_ids,
                type_action = 'line_or_change_bus',
                is_set = False,
                value = 2
            )
            
            # Test for line_exs
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_line,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].line_orig_ids,
                type_action = 'line_ex_change_bus',
                is_set = False,
                value = 2
            )
            
            
            # Test for storages
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_storage,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].storage_orig_ids,
                type_action = 'storage_change_bus',
                is_set = False,
                value = 2
            )
        
    
    def test_local_action_to_global_change_line_status(self):
        # tests for change line status actions
        for agent in self.ma_env.agents:
            
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_line,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].line_orig_ids,
                type_action = 'line_change_status',
                is_set = False,
                value = 2,
            )
            
            
    def test_local_action_to_global_set_line_status(self):
        # tests for set line status actions
        for agent in self.ma_env.agents:
            
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_line,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].line_orig_ids,
                type_action = 'line_set_status',
                is_set = True,
                value = -1,
            )
            
            
    def test_local_action_to_global_redispatch(self):
        # tests for redispatch actions
        
        # We should have at least one redispatchable generator
        assert (self.ma_env._cent_env.gen_redispatchable).any()
        
        for agent in self.ma_env.agents:
            
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_gen,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].gen_orig_ids,
                type_action = 'redispatch',
                is_set = True,
                value = 0.42,
            )

                
    def test_local_action_to_global_curtail(self):
        
        for agent in self.ma_env.agents:
            
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_gen,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].gen_orig_ids,
                type_action = 'curtail',
                is_set = True,
                value = 0.42,
            )

                
    def test_local_action_to_global_set_storage(self):
        # We should have at least one storage unit
        assert self.ma_env._cent_env.n_storage > 0
        
        for agent in self.ma_env.agents:
            
            self.check_local2global(
                agent = agent,
                n = self.ma_env._subgrids_cls['action'][agent].n_storage,
                orig_ids = self.ma_env._subgrids_cls['action'][agent].storage_orig_ids,
                type_action = 'storage_p',
                is_set = True,
                value = 0.42,
            )

        
        
    #TODO other actions 
    # V0
    # change_bus, done
    # redispatch done
    # curtail done
    # change_line_status done 
    # set_line_status done
    # set_storage done
    
    # TODO v0.1
    # Topo actions on interconnections
    
    # TODO v0.2
    # change_interco_status
    # set_...

    def check_legal_actions(self):
        # We create a legal action and check
        # if _build_global_action gives the
        # correct global_action
        action = dict()
        local_id = 0
        for agent in self.ma_env.agents:
            action[agent] = self.ma_env.action_spaces[agent]({})
            action[agent].change_line_status = [local_id]
        
        self.ma_env._build_global_action(action, self.ma_env.agents)
        
        ref_action = self.ma_env._cent_env.action_space({})
        ref_action.change_line_status = [
            self.ma_env._subgrids_cls['action'][agent].line_orig_ids[local_id]
            for agent in self.ma_env.agents
        ]
        
        # We check if the global action is the wanted action
        assert ref_action == self.ma_env.global_action
    
    def check_illegal_actions(self):
        # We create an illegal action and check
        # if _build_global_action gives the
        # do nothing
        action = dict()
        local_id = 0
        for agent in self.ma_env.agents:
            action[agent] = self.ma_env.action_spaces[agent]({})
            action[agent].change_line_status = [local_id, local_id+1]
        
        self.ma_env._build_global_action(action, self.ma_env.agents)
        
        do_nothing = self.ma_env._cent_env.action_space({})
        
        # We check if the global action is do nothing
        assert do_nothing == self.ma_env.global_action
        # We check if info is updated
        assert (np.array([
            self.ma_env.info[a]['action_is_illegal']
            for a in self.ma_env.agents
        ])).all()
    
    def check_ambiguous_actions(self):
        # We create an ambiguous action and check
        # if _build_global_action gives the
        # do nothing
        action = dict()
        local_id = 0
        for agent in self.ma_env.agents:
            action[agent] = self.ma_env.action_spaces[agent]({})
            # This is an ambiguous action. The curtailment ratio
            # must be between 0 and 1
            action[agent].curtail = [(local_id, 2)]
        
        self.ma_env._build_global_action(action, self.ma_env.agents)
        
        do_nothing = self.ma_env._cent_env.action_space({})
        # We check if the global action is do nothing
        assert do_nothing == self.ma_env.global_action
        # We check if info is updated
        assert (np.array([
            self.ma_env.info[a]['is_ambiguous']
            for a in self.ma_env.agents
        ])).all()
        
    def test_build_global_action(self):
        self.check_legal_actions()
        self.check_illegal_actions()
        self.check_ambiguous_actions()
        
    
    def test_action_spaces(self):        
        action_domains = {
            'agent_0' : [0,1,2,3, 4],
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        space = "action"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ma_env = MultiAgentEnv(self.env,
                                   action_domains,
                                   _add_to_name="_test_action_spaces")
            
        for agent in ma_env.agents:
            # The action space must have the same dim_topo
            # as its subgrid class
            assert ma_env.action_spaces[agent].dim_topo == ma_env._subgrids_cls[space][agent].dim_topo, f"wrong dimension of action space for agent {agent}"
            do_nothing = ma_env.action_spaces[agent]({})
            # Same for any instance (action)
            assert do_nothing.dim_topo == ma_env._subgrids_cls[space][agent].dim_topo, f"wrong dimension of action for agent {agent}"
            
            # check name of classes are correct
            assert re.sub("^SubGridAction", "", type(do_nothing).__name__) == re.sub("^SubGridActionSpace", "", type(ma_env.action_spaces[agent]).__name__)
    
    def test_step(self):
        self.ma_env.seed(0)  # do not change the seed otherwise you might have some "action on interco" which are not fully implemented yet
        self.ma_env.reset()
        for _ in range(10):
            while True:
                actions = {
                    agent : _aux_sample_withtout_interco(self.ma_env.action_spaces[agent])
                    for agent in self.ma_env.agents
                }
                obs, rewards, dones, info = self.ma_env.step(actions)
                if dones[self.ma_env.agents[0]]:
                    self.ma_env.reset()
                    break
                
        
class TestAction(unittest.TestCase):
    def setUp(self) -> None:
        
        self.action_domains = {
            'agent_0' : [0, 1, 2, 3, 4],
            'agent_1' : [5, 6, 7, 8, 9, 10, 11, 12, 13]
        }
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    action_class=PlayableAction,
                                    _add_to_name=type(self).__name__)

        
            self.ma_env = MultiAgentEnv(self.env, self.action_domains)
            self.ma_env.seed(0)
            self.ma_env.reset()
            
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        self.ma_env._cent_env.close()
        return super().tearDown()
        
    def test_interco_set_bus(self):
        id_ = 0
        # 1) test it works when it should
        for agent_nm in ["agent_0", "agent_1"]:
            act = self.ma_env.action_spaces[agent_nm]()
            for id_ in range(type(act).n_interco):
                for bus_id in [1, 2]:
                    act = self.ma_env.action_spaces[agent_nm]()
                    act.interco_set_bus = [(id_, bus_id)]
                    # the correct position is changed
                    assert act._set_topo_vect[type(act).interco_pos_topo_vect[id_]] == bus_id
                    # only this position is affected
                    assert act._set_topo_vect[type(act).interco_pos_topo_vect].sum() == bus_id
                
        # 2) test it should NOT work when it shouldn't
        agent_nm = "agent_0"
        # wrong id
        id_ = -1
        act = self.ma_env.action_spaces[agent_nm]()
        with self.assertRaises(IllegalAction):
            act.interco_set_bus = [(id_, 1)]
            
        # wrong id
        id_ = type(act).n_interco
        act = self.ma_env.action_spaces[agent_nm]()
        with self.assertRaises(IllegalAction):
            act.interco_set_bus = [(id_, 1)]
            
        # wrong bus
        id_ = 0
        act = self.ma_env.action_spaces[agent_nm]()
        with self.assertRaises(IllegalAction):
            act.interco_set_bus = [(id_, 3)]
            
    def test_interco_change_bus(self):
        id_ = 0
        # 1) test it works when it should
        for agent_nm in ["agent_0", "agent_1"]:
            act = self.ma_env.action_spaces[agent_nm]()
            for id_ in range(type(act).n_interco):
                act = self.ma_env.action_spaces[agent_nm]()
                act.interco_change_bus = [id_]
                # the correct position is changed
                assert act._change_bus_vect[type(act).interco_pos_topo_vect[id_]]
                # only this position is affected
                assert act._change_bus_vect[type(act).interco_pos_topo_vect].sum() == 1
                
        # 2) test it should NOT work when it shouldn't
        agent_nm = "agent_0"
        # wrong id
        id_ = -1
        act = self.ma_env.action_spaces[agent_nm]()
        with self.assertRaises(IllegalAction):
            act.interco_change_bus = [id_]
            
        # wrong id
        id_ = type(act).n_interco
        act = self.ma_env.action_spaces[agent_nm]()
        with self.assertRaises(IllegalAction):
            act.interco_change_bus = [id_]

    def test_interco_set_bus_dict(self):
        id_ = 0
        # 1) test it works when it should
        for agent_nm in ["agent_0", "agent_1"]:
            act = self.ma_env.action_spaces[agent_nm]()
            for id_ in range(type(act).n_interco):
                for bus_id in [1, 2]:
                    act = self.ma_env.action_spaces[agent_nm]({"set_bus": {"intercos_id":  [(id_, bus_id)]}})
                    # the correct position is changed
                    assert act._set_topo_vect[type(act).interco_pos_topo_vect[id_]] == bus_id
                    # only this position is affected
                    assert act._set_topo_vect[type(act).interco_pos_topo_vect].sum() == bus_id
                
        # 2) test it should NOT work when it shouldn't
        agent_nm = "agent_0"
        # wrong id
        id_ = -1
        with self.assertRaises(IllegalAction):
            act = self.ma_env.action_spaces[agent_nm]({"set_bus": {"intercos_id":  [(id_, 1)]}})
            
        # wrong id
        id_ = type(act).n_interco
        act = self.ma_env.action_spaces[agent_nm]()
        with self.assertRaises(IllegalAction):
            act = self.ma_env.action_spaces[agent_nm]({"set_bus": {"intercos_id":  [(id_, 1)]}})
            
        # wrong bus
        id_ = 0
        with self.assertRaises(IllegalAction):
            act = self.ma_env.action_spaces[agent_nm]({"set_bus": {"intercos_id":  [(id_, 3)]}})
             
    def test_interco_change_bus_dict(self):
        id_ = 0
        # 1) test it works when it should
        for agent_nm in ["agent_0", "agent_1"]:
            act = self.ma_env.action_spaces[agent_nm]()
            for id_ in range(type(act).n_interco):
                act = self.ma_env.action_spaces[agent_nm]({"change_bus": {"intercos_id":  [id_]}})
                # the correct position is changed
                assert act._change_bus_vect[type(act).interco_pos_topo_vect[id_]]
                # only this position is affected
                assert act._change_bus_vect[type(act).interco_pos_topo_vect].sum() == 1
                
        # 2) test it should NOT work when it shouldn't
        agent_nm = "agent_0"
        # wrong id
        id_ = -1
        with self.assertRaises(IllegalAction):
            act = self.ma_env.action_spaces[agent_nm]({"change_bus": {"intercos_id":  [id_]}})
            
        # wrong id
        id_ = type(act).n_interco
        with self.assertRaises(IllegalAction):
            act = self.ma_env.action_spaces[agent_nm]({"change_bus": {"intercos_id":  [id_]}})

    def test_interco_set_status(self):
        id_ = 0
        # 1) test it works when it should
        for agent_nm in ["agent_0", "agent_1"]:
            act = self.ma_env.action_spaces[agent_nm]()
            for id_ in range(type(act).n_interco):
                for stat in [-1, 1]:
                    act = self.ma_env.action_spaces[agent_nm]()
                    act.interco_set_status = [(id_, stat)]
                    # the flag that we change is properly set
                    assert act._modif_interco_set_status 
                    # the correct position is changed
                    assert act._set_interco_status[id_] == stat
                    # only this position is affected
                    assert act._set_interco_status.sum() == stat
                
        # 2) test it should NOT work when it shouldn't
        agent_nm = "agent_0"
        # wrong id
        id_ = -1
        act = self.ma_env.action_spaces[agent_nm]()
        with self.assertRaises(IllegalAction):
            act.interco_set_status = [(id_, 1)]
            
        # wrong id
        id_ = type(act).n_interco
        act = self.ma_env.action_spaces[agent_nm]()
        with self.assertRaises(IllegalAction):
            act.interco_set_status = [(id_, 1)]
            
        # wrong status
        id_ = 0
        act = self.ma_env.action_spaces[agent_nm]()
        with self.assertRaises(IllegalAction):
            act.interco_set_status = [(id_, 2)]
        with self.assertRaises(IllegalAction):
            act.interco_set_status = [(id_, -2)]

    def test_interco_change_status(self):
        id_ = 0
        # 1) test it works when it should
        for agent_nm in ["agent_0", "agent_1"]:
            act = self.ma_env.action_spaces[agent_nm]()
            for id_ in range(type(act).n_interco):
                act = self.ma_env.action_spaces[agent_nm]()
                act.interco_change_status = [id_]
                # the flag that we change is properly set
                assert act._modif_interco_change_status 
                # the correct position is changed
                assert act._switch_interco_status[id_]
                # only this position is affected
                assert act._switch_interco_status.sum() == 1
                
        # 2) test it should NOT work when it shouldn't
        agent_nm = "agent_0"
        # wrong id (too low)
        id_ = -1
        act = self.ma_env.action_spaces[agent_nm]()
        with self.assertRaises(IllegalAction):
            act.interco_change_status = [id_]
            
        # wrong id (too high)
        id_ = type(act).n_interco
        act = self.ma_env.action_spaces[agent_nm]()
        with self.assertRaises(IllegalAction):
            act.interco_change_status = [id_]

    def test_interco_set_status_dict(self):
        id_ = 0
        # 1) test it works when it should
        for agent_nm in ["agent_0", "agent_1"]:
            act = self.ma_env.action_spaces[agent_nm]()
            for id_ in range(type(act).n_interco):
                for stat in [-1, 1]:
                    act = self.ma_env.action_spaces[agent_nm]({"set_interco_status": [(id_, stat)]})
                    # the flag that we change is properly set
                    assert act._modif_interco_set_status 
                    # the correct position is changed
                    assert act._set_interco_status[id_] == stat
                    # only this position is affected
                    assert act._set_interco_status.sum() == stat
                
        # 2) test it should NOT work when it shouldn't
        agent_nm = "agent_0"
        # wrong id
        id_ = -1
        with self.assertRaises(IllegalAction):
            act = self.ma_env.action_spaces[agent_nm]({"set_interco_status": [(id_, 1)]})
            
        # wrong id
        id_ = type(act).n_interco
        with self.assertRaises(IllegalAction):
            act = self.ma_env.action_spaces[agent_nm]({"set_interco_status": [(id_, 1)]})
            
        # wrong status
        id_ = 0
        with self.assertRaises(IllegalAction):
            act = self.ma_env.action_spaces[agent_nm]({"set_interco_status": [(id_, -2)]})
        with self.assertRaises(IllegalAction):
            act = self.ma_env.action_spaces[agent_nm]({"set_interco_status": [(id_, 2)]})
      
    def test_interco_change_status_dict(self):
        id_ = 0
        # 1) test it works when it should
        for agent_nm in ["agent_0", "agent_1"]:
            act = self.ma_env.action_spaces[agent_nm]()
            for id_ in range(type(act).n_interco):
                act = self.ma_env.action_spaces[agent_nm]({"change_interco_status":  [id_]})
                # the flag that we change is properly set
                assert act._modif_interco_change_status 
                # the correct position is changed
                assert act._switch_interco_status[id_]
                # only this position is affected
                assert act._switch_interco_status.sum() == 1
                
        # 2) test it should NOT work when it shouldn't
        agent_nm = "agent_0"
        # wrong id (too low)
        id_ = -1
        with self.assertRaises(IllegalAction):
            act = self.ma_env.action_spaces[agent_nm]({"change_interco_status":  [id_]})
            
        # wrong id (too high)
        id_ = type(act).n_interco
        with self.assertRaises(IllegalAction):
            act = self.ma_env.action_spaces[agent_nm]({"change_interco_status":  [id_]})
                    
    def test_to_local(self):
        # TODO this test is not extensive at all !
        # for example, it should test the interco too, and I remove them !
        # and its just "quick and dirty" tests using seed, better tests would be to create these actions
        # and think about 
        self.ma_env.seed(0)
        self.ma_env.reset()
        
        # test global -> locals -> global
        for i in range(100):
            global_act = _aux_sample_without_interco_from_global(self.ma_env._cent_env.action_space,
                                                                 self.ma_env.action_spaces
                                                                )
            local_act = {agent_nm: self.ma_env.action_spaces[agent_nm].from_global(global_act) 
                         for agent_nm in self.ma_env.agents}
            
            global_act_2 = (local_act["agent_0"].to_global(self.ma_env._cent_env.action_space) +
                            local_act["agent_1"].to_global(self.ma_env._cent_env.action_space)
                           )
            if i == 42:
                # this action does nothing, but unfortunately it's because it sampled a subtation
                # a change_bus, and "decided" not to change anything
                # this flag is "lost in the conversion"
                global_act._modif_change_bus = False
                # global_act._modif_set_bus = False
            # when I combine these actions, it should be true
            assert global_act_2 == global_act, f"error for iteration {i} with ref:\n{global_act}\nand rebuilt:\n{global_act_2}"
            
        # test locals -> global -> locals
        for i in range(100):
            local_act = {agent : _aux_sample_withtout_interco(self.ma_env.action_spaces[agent])
                         for agent in self.ma_env.agents}
            
            global_act = (local_act["agent_0"].to_global(self.ma_env._cent_env.action_space) +
                          local_act["agent_1"].to_global(self.ma_env._cent_env.action_space)
                         )
            
            local_act_2 = {agent_nm: self.ma_env.action_spaces[agent_nm].from_global(global_act) 
                           for agent_nm in self.ma_env.agents}

            # if i == 0:
                # local_act["agent_0"]._modif_alert = False
            if i == 78:
                # this action does nothing, but unfortunately it's because it sampled a subtation
                # a change_bus, and "decided" not to change anything
                # this flag is "lost in the conversion"
                local_act["agent_0"]._modif_change_bus = False

            elif i == 82 or i == 83 or i == 85:
                # this action does nothing, but unfortunately it's because it sampled a subtation
                # a change_bus, and "decided" not to change anything
                # this flag is "lost in the conversion"
                local_act["agent_1"]._modif_change_bus = False
                
            # when I combine these actions, it should be true
            for agent_nm in self.ma_env.agents:
                assert local_act[agent_nm] == local_act_2[agent_nm], f"error for iteration {i}, agent {agent_nm} with ref:\n{local_act[agent_nm]}\nand rebuilt:\n{local_act_2[agent_nm]}"
   
                                                    
class TestLocalObservation(unittest.TestCase):
    def setUp(self) -> None:
        
        self.action_domains = {
            'agent_0' : [0, 1, 2, 3, 4],
            'agent_1' : [5, 6, 7, 8, 9, 10, 11, 12, 13]
        }
        
        self.observation_domains = {
            'agent_0' : [0, 1, 2, 3, 4, 5, 6, 8],
            'agent_1' : [5, 6, 7, 8, 9, 10, 11, 12, 13, 4, 3]
        }
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    action_class=PlayableAction,
                                    _add_to_name=type(self).__name__)

        
            self.ma_env = MultiAgentEnv(self.env, self.action_domains, self.observation_domains)
            self.ma_env.seed(0)
            self.ma_env.reset()
            
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        self.ma_env._cent_env.close()
        return super().tearDown()

    def test_reset_env(self):
        obs = self.ma_env.reset()
        for ag_nm in ["agent_0", "agent_1"]:
            assert not obs[ag_nm]._is_complete_obs       
    
    def test_step(self):
        self.ma_env.seed(0)  # do not change the seed otherwise you might have some "action on interco" which are not fully implemented yet
        self.ma_env.reset()
        for _ in range(10):
            while True:
                actions = {
                    agent : _aux_sample_withtout_interco(self.ma_env.action_spaces[agent])
                    for agent in self.ma_env.agents
                }
                obs, rewards, dones, info = self.ma_env.step(actions)
                if dones[self.ma_env.agents[0]]:
                    self.ma_env.reset()
                    break
                
                # For now, it is not clear how to "simulate" with a partial observation
                with self.assertRaises(SimulateError):
                    obs["agent_0"].simulate(actions)
                with self.assertRaises(SimulateError):
                    obs["agent_1"].simulate(actions)
                 

class TestGlobalObservation(unittest.TestCase):
    def setUp(self) -> None:
        
        self.action_domains = {
            'agent_0' : [0, 1, 2, 3, 4],
            'agent_1' : [5, 6, 7, 8, 9, 10, 11, 12, 13]
        }
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    action_class=PlayableAction,
                                    _add_to_name=type(self).__name__)

        
            self.ma_env = MultiAgentEnv(self.env, self.action_domains)
            self.ma_env.seed(0)
            self.ma_env.reset()
            
        return super().setUp()

    def test_reset_env(self):
        obs = self.ma_env.reset()
        for ag_nm in ["agent_0", "agent_1"]:
            assert obs[ag_nm]._is_complete_obs       
    
    def test_step(self):
        self.ma_env.seed(0)  # do not change the seed otherwise you might have some "action on interco" which are not fully implemented yet
        self.ma_env.reset()
        for _ in range(10):
            while True:
                actions = {
                    agent : _aux_sample_withtout_interco(self.ma_env.action_spaces[agent])
                    for agent in self.ma_env.agents
                }
                obs, rewards, dones, info = self.ma_env.step(actions)
                if dones[self.ma_env.agents[0]]:
                    self.ma_env.reset()
                    break
                
    def test_simulate(self):
        self.ma_env.seed(0)
        obs = self.ma_env.reset()
        for nb_step in range(10):
            actions = {
                agent : _aux_sample_withtout_interco(self.ma_env.action_spaces[agent])
                for agent in self.ma_env.agents
            }
            
            for ag_nm in ["agent_0", "agent_1"]:
                sim_o, sim_r, sim_d, sim_i =  obs[ag_nm].simulate(actions)   
                assert not isinstance(sim_o, SubGridObservation)            
                assert isinstance(sim_o, CompleteObservation)  
                
                # now check the simulated observation is the same as the one from the global env
                global_obs = self.ma_env._cent_env.get_obs()
                global_act_sp = self.ma_env._cent_env.action_space
                global_act = global_act_sp()
                for agent_nm, local_act in actions.items():
                    global_act += local_act.to_global(global_act_sp)
                sim_o_g, sim_r_g, sim_d_g, sim_i_g = global_obs.simulate(global_act)
                assert [f"{el}" for el in sim_i_g["exception"]] == [f"{el}" for el in sim_i["exception"]]
                assert sim_o_g == sim_o
                assert sim_r_g == sim_r
            obss, rewards, dones, infos = self.ma_env.step(actions)
            if dones[self.ma_env.agents[0]]:
                obs = self.ma_env.reset()


if __name__ == "__main__":
    unittest.main()
