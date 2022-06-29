# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
from grid2op import make
from grid2op.multi_agent.multiAgentEnv import MultiAgentEnv
import pdb
import numpy as np
from grid2op.multi_agent.multi_agentExceptions import *



class MATesterGlobalObs(unittest.TestCase):
    def setUp(self) -> None:
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("l2rpn_case14_sandbox", test=True, _add_to_name="test_ma")

        self.action_domains = {
            'agent_0' : [0,1,2,3, 4],
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        self.ma_env = MultiAgentEnv(self.env, self.action_domains)
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
            MultiAgentEnv(self.env, action_domains, _add_to_name="test_verify_domains_0")
        
        action_domains = {
            'agent_0' : [0],
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        # above action domain should raise an error: substations are not fully allocated !
        with self.assertRaises(DomainException) as de:
            MultiAgentEnv(self.env, action_domains, _add_to_name="test_verify_domains_1")
            
        action_domains = {
            'agent_0' : [],
            'agent_1' : list(range(self.env.n_sub))
        }
        # above action domain should raise an error: agents must have at least one substation !
        with self.assertRaises(DomainException) as de:
            MultiAgentEnv(self.env, action_domains, _add_to_name="test_verify_domains_2")
            
        action_domains = {
            'agent_0' : [0,1,6,3, 4],
            'agent_1' : [5,2,7,8,9,10,11,12,13]
        }
        # this domain is valid even if it is not connected
        try:
            MultiAgentEnv(self.env, action_domains, _add_to_name="test_verify_domains_3")
        except DomainException:
            self.fail("action_domains raised Domain Exception unexpectedly!")
            
        action_domains = {
            'agent_0' : [0,1,6,3, 4, 5],
            'agent_1' : [5,2,7,8,9,10,11,12,13]
        }
        # this domain is not a partition ; it should raise an error
        with self.assertRaises(DomainException) as de:
            MultiAgentEnv(self.env, action_domains, _add_to_name="test_verify_domains_4")
        
    
    def test_build_subgrids_action_domains(self):
        # Simple test to verify if action domains are correctly
        # taken into accaount by the env
        self.ma_env = MultiAgentEnv(self.env, self.action_domains, _add_to_name="test_build_subgrids_action_domains")
        assert self.ma_env._action_domains['agent_0']['sub_id'] == self.action_domains['agent_0']
        assert self.ma_env._action_domains['agent_1']['sub_id'] == self.action_domains['agent_1']
    
    def test_masks(self):
        # We compare the masks with known values for every agent
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
        assert (self.ma_env._action_domains['agent_0']['mask_storage'] == []).all()
        assert (self.ma_env._action_domains['agent_1']['mask_storage'] == []).all()
        
        mask_line_ex_agent0 = np.array([ True,  True,  True,  True,  True,  True,  True, 
                                False, False,False, False, False, False, False, False, 
                                False, False, False,False, False])
        # We compare the line_ex masks with known values for every agent
        assert (self.ma_env._action_domains['agent_0']['mask_line_ex'] == mask_line_ex_agent0).all()
        mask_line_ex_agent1 = np.array([False, False, False, False, False, False, False,
                                        True,  True,  True, True,  True, True,  True,  True,
                                        False, False, False,  True,  True])
        assert (self.ma_env._action_domains['agent_1']['mask_line_ex'] == mask_line_ex_agent1).all()
        # We compare the line_or masks with known values for every agent
        assert (self.ma_env._action_domains['agent_0']['mask_line_or'] == mask_line_ex_agent0).all()
        assert (self.ma_env._action_domains['agent_1']['mask_line_or'] == mask_line_ex_agent1).all()
        
        mask_shunt_agent0 = np.array([False])
        # We compare the shunt masks with known values for every agent
        assert (self.ma_env._action_domains['agent_0']['mask_shunt'] == mask_shunt_agent0).all()
        assert (self.ma_env._action_domains['agent_1']['mask_shunt'] == ~mask_shunt_agent0).all()
        
    def test_interco(self):
        
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
            'agent_0' : [0,1,2,3, 4],
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        ma_env = MultiAgentEnv(self.env, action_domains, _add_to_name="test_build_subgrid_obj")
        
        # We compare the number of generators for every agents' subgrids with known values
        assert ma_env._subgrids_cls['action']['agent_0'].n_gen == 3
        assert ma_env._subgrids_cls['action']['agent_1'].n_gen == 3
        
        self.check_subgrid_consistency(ma_env, action_domains)
        
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
        
        
    def test_build_subgrid_obj2(self):    
        # 2
        # Test with 3 agents
        action_domains = {
            'test_2_agent_0' : [0,1, 2, 3, 4],
            'test_2_agent_1' : [5, 6, 7, 8, 9],
            'test_2_agent_2' : [10, 11, 12, 13],
            
        }
        ma_env = MultiAgentEnv(self.env, action_domains, _add_to_name="test_build_subgrid_obj2")
        
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
            # run redispatch agent on one scenario for 100 timesteps
            ma_env = MultiAgentEnv(self.env, action_domains, _add_to_name=f"_it_{it}")
            self.check_subgrid_consistency(ma_env, action_domains, add_msg=f"error for iter {it}")
            
            
    def check_subgrid_consistency(self, ma_env, action_domains, add_msg=""):
        # Regroups all the checks to be done
        self.check_orig_ids(ma_env, action_domains)
        self.check_n_objects(ma_env, action_domains, add_msg=add_msg)
        self.check_objects_to_subid(ma_env, action_domains)
        self.check_connections(ma_env, action_domains)
        self.check_shunt(ma_env)
    
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
                ma_env._subgrids_cls[space][agent].n_interco, add_msg
        
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
            mask_line_or = ma_env._subgrids_cls[space][agent].mask_line_or
            mask_line_ex = ma_env._subgrids_cls[space][agent].mask_line_ex
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
                    np.arange(ma_env._cent_env.n_line)[mask_line_or]).all()
            # We check that we have the correct generators original ids
            # Ids should be the same as those given by line_ex masks 
            assert (ma_env._subgrids_cls[space][agent].line_orig_ids\
                ==\
                    np.arange(ma_env._cent_env.n_line)[mask_line_ex]).all()
            
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
                        ma_env._subgrids_cls[space][agent].mask_line_ex
                    ]).all()
            # We check if a line_or is on a substation on the subgrid,
            # it is also on the original grid
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].line_or_to_subid]\
                ==\
                    ma_env._cent_env.line_or_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_line_or
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
    
    
    
    def test_local_action_to_global_set_bus(self):
        # TODO 
        np.random.seed(0)
        for agent in self.ma_env.agents:
            # Test for loads
            local_load_id = np.random.randint(0, self.ma_env._subgrids_cls['action'][agent].n_load)
            #local_act = self.ma_env.action_spaces[agent]({
            #    'set_bus' : 
            #        (self.ma_env.action_spaces[agent].load_pos_topo_vect[local_load_id], 2)
            #})
            local_act = self.ma_env.action_spaces[agent]({})
            local_act.load_set_bus = [(local_load_id, 2)]
            global_act = self.ma_env._local_action_to_global(agent, local_act)
            global_load_id = self.ma_env._subgrids_cls['action'][agent].load_orig_ids[local_load_id]
            ref_global_act = self.ma_env._cent_env.action_space({
                'set_bus' : 
                    (self.ma_env._cent_env.action_space.load_pos_topo_vect[global_load_id], 2)
            })
            assert (global_act.set_bus == ref_global_act.set_bus).all()
            assert global_act._modif_set_bus == ref_global_act._modif_set_bus
            
            # Test for gens
            local_gen_id = np.random.randint(0, self.ma_env._subgrids_cls['action'][agent].n_gen)
            #local_act = self.ma_env.action_spaces[agent]({
            #    'set_bus' : 
            #        (self.ma_env.action_spaces[agent].gen_pos_topo_vect[local_gen_id], 2)
            #})
            local_act = self.ma_env.action_spaces[agent]({})
            local_act.gen_set_bus = [(local_gen_id, 2)]
            global_act = self.ma_env._local_action_to_global(agent, local_act)
            global_gen_id = self.ma_env._subgrids_cls['action'][agent].gen_orig_ids[local_gen_id]
            ref_global_act = self.ma_env._cent_env.action_space({
                'set_bus' : 
                    (self.ma_env._cent_env.action_space.gen_pos_topo_vect[global_gen_id], 2)
            })
            assert (global_act.set_bus == ref_global_act.set_bus).all()
            assert global_act._modif_set_bus == ref_global_act._modif_set_bus
            
            # Test for line_ors
            local_line_or_id = np.random.randint(0, self.ma_env._subgrids_cls['action'][agent].n_line)
            #local_act = self.ma_env.action_spaces[agent]({
            #    'set_bus' : 
            #        (self.ma_env.action_spaces[agent].line_or_pos_topo_vect[local_line_or_id], 2)
            #})
            local_act = self.ma_env.action_spaces[agent]({})
            local_act.line_or_set_bus = [(local_line_or_id, 2)]
            global_act = self.ma_env._local_action_to_global(agent, local_act)
            global_line_or_id = self.ma_env._subgrids_cls['action'][agent].line_orig_ids[local_line_or_id]
            ref_global_act = self.ma_env._cent_env.action_space({
                'set_bus' : 
                    (self.ma_env._cent_env.action_space.line_or_pos_topo_vect[global_line_or_id], 2)
            })
            assert (global_act.set_bus == ref_global_act.set_bus).all()
            assert global_act._modif_set_bus == ref_global_act._modif_set_bus
            
            # Test for line_exs
            local_line_ex_id = np.random.randint(0, self.ma_env._subgrids_cls['action'][agent].n_line)
            #local_act = self.ma_env.action_spaces[agent]({
            #    'set_bus' : 
            #        (self.ma_env.action_spaces[agent].line_ex_pos_topo_vect[local_line_ex_id], 2)
            #})
            local_act = self.ma_env.action_spaces[agent]({})
            local_act.line_ex_set_bus = [(local_line_ex_id, 2)]
            global_act = self.ma_env._local_action_to_global(agent, local_act)
            global_line_ex_id = self.ma_env._subgrids_cls['action'][agent].line_orig_ids[local_line_ex_id]
            ref_global_act = self.ma_env._cent_env.action_space({
                'set_bus' : 
                    (self.ma_env._cent_env.action_space.line_ex_pos_topo_vect[global_line_ex_id], 2)
            })
            assert (global_act.set_bus == ref_global_act.set_bus).all()
            assert global_act._modif_set_bus == ref_global_act._modif_set_bus
            
            # Test for storages
            if self.ma_env._subgrids_cls['action'][agent].n_storage > 0:
                local_storage_id = np.random.randint(0, self.ma_env._subgrids_cls['action'][agent].n_storage)
                #local_act = self.ma_env.action_spaces[agent]({
                #    'set_bus' : 
                #        (self.ma_env.action_spaces[agent].storage_pos_topo_vect[local_storage_id], 2)
                #})
                local_act = self.ma_env.action_spaces[agent]({})
                local_act.storage_set_bus = [(local_storage_id, 2)]
                global_act = self.ma_env._local_action_to_global(agent, local_act)
                global_storage_id = self.ma_env._subgrids_cls['action'][agent].storage_orig_ids[local_storage_id]
                ref_global_act = self.ma_env._cent_env.action_space({
                    'set_bus' : 
                        (self.ma_env._cent_env.action_space.storage_pos_topo_vect[global_storage_id], 2)
                })
                assert (global_act.set_bus == ref_global_act.set_bus).all()
                assert global_act._modif_set_bus == ref_global_act._modif_set_bus
            
    def test_local_action_to_global_change_bus(self):
        # TODO 
        np.random.seed(0)
        for agent in self.ma_env.agents:
            # Test for loads
            local_load_id = np.random.randint(0, self.ma_env._subgrids_cls['action'][agent].n_load)
            #local_act = self.ma_env.action_spaces[agent]({
            #    'change_bus' : 
            #        (self.ma_env.action_spaces[agent].load_pos_topo_vect[local_load_id], 2)
            #})
            local_act = self.ma_env.action_spaces[agent]({})
            local_act.load_change_bus = [local_load_id]
            
            global_act = self.ma_env._local_action_to_global(agent, local_act)
            
            global_load_id = self.ma_env._subgrids_cls['action'][agent].load_orig_ids[local_load_id]
            ref_global_act = self.ma_env._cent_env.action_space({
                'change_bus' : 
                    self.ma_env._cent_env.action_space.load_pos_topo_vect[global_load_id]
            })
            assert (global_act.change_bus == ref_global_act.change_bus).all()
            assert global_act._modif_change_bus == ref_global_act._modif_change_bus
            
            # Test for gens
            local_gen_id = np.random.randint(0, self.ma_env._subgrids_cls['action'][agent].n_gen)
            #local_act = self.ma_env.action_spaces[agent]({
            #    'change_bus' : 
            #        (self.ma_env.action_spaces[agent].gen_pos_topo_vect[local_gen_id], 2)
            #})
            local_act = self.ma_env.action_spaces[agent]({})
            local_act.gen_change_bus = [local_gen_id]
            
            global_act = self.ma_env._local_action_to_global(agent, local_act)
            
            global_gen_id = self.ma_env._subgrids_cls['action'][agent].gen_orig_ids[local_gen_id]
            ref_global_act = self.ma_env._cent_env.action_space({
                'change_bus' : 
                    self.ma_env._cent_env.action_space.gen_pos_topo_vect[global_gen_id]
            })
            assert (global_act.change_bus == ref_global_act.change_bus).all()
            assert global_act._modif_change_bus == ref_global_act._modif_change_bus
            
            # Test for line_ors
            local_line_or_id = np.random.randint(0, self.ma_env._subgrids_cls['action'][agent].n_line)
            #local_act = self.ma_env.action_spaces[agent]({
            #    'change_bus' : 
            #        (self.ma_env.action_spaces[agent].line_or_pos_topo_vect[local_line_or_id], 2)
            #})
            local_act = self.ma_env.action_spaces[agent]({})
            local_act.line_or_change_bus = [local_line_or_id]
            
            global_act = self.ma_env._local_action_to_global(agent, local_act)
            
            global_line_or_id = self.ma_env._subgrids_cls['action'][agent].line_orig_ids[local_line_or_id]
            ref_global_act = self.ma_env._cent_env.action_space({
                'change_bus' : 
                    self.ma_env._cent_env.action_space.line_or_pos_topo_vect[global_line_or_id]
            })
            assert (global_act.change_bus == ref_global_act.change_bus).all()
            assert global_act._modif_change_bus == ref_global_act._modif_change_bus
            
            # Test for line_exs
            local_line_ex_id = np.random.randint(0, self.ma_env._subgrids_cls['action'][agent].n_line)
            #local_act = self.ma_env.action_spaces[agent]({
            #    'change_bus' : 
            #        (self.ma_env.action_spaces[agent].line_ex_pos_topo_vect[local_line_ex_id], 2)
            #})
            local_act = self.ma_env.action_spaces[agent]({})
            local_act.line_ex_change_bus = [local_line_ex_id]
            
            global_act = self.ma_env._local_action_to_global(agent, local_act)
            
            global_line_ex_id = self.ma_env._subgrids_cls['action'][agent].line_orig_ids[local_line_ex_id]
            ref_global_act = self.ma_env._cent_env.action_space({
                'change_bus' : 
                    self.ma_env._cent_env.action_space.line_ex_pos_topo_vect[global_line_ex_id]
            })
            assert (global_act.change_bus == ref_global_act.change_bus).all()
            assert global_act._modif_change_bus == ref_global_act._modif_change_bus
            
            # Test for storages
            if self.ma_env._subgrids_cls['action'][agent].n_storage > 0:
                local_storage_id = np.random.randint(0, self.ma_env._subgrids_cls['action'][agent].n_storage)
                #local_act = self.ma_env.action_spaces[agent]({
                #    'change_bus' : 
                #        (self.ma_env.action_spaces[agent].storage_pos_topo_vect[local_storage_id], 2)
                #})
                local_act = self.ma_env.action_spaces[agent]({})
                local_act.storage_change_bus = [local_storage_id]
                
                global_act = self.ma_env._local_action_to_global(agent, local_act)
                
                global_storage_id = self.ma_env._subgrids_cls['action'][agent].storage_orig_ids[local_storage_id]
                ref_global_act = self.ma_env._cent_env.action_space({
                    'change_bus' : 
                        self.ma_env._cent_env.action_space.storage_pos_topo_vect[global_storage_id]
                })
                assert (global_act.change_bus == ref_global_act.change_bus).all()
                assert global_act._modif_change_bus == ref_global_act._modif_change_bus

        
    #TODO other actions 
    # V0
    # change_bus,
    # redispatch
    # curtail
    # change_line_status
    # set_line_status
    # set_storage
    
    # TODO v0.1
    # Topo actions on interconnections
    
    # TODO v0.2
    # change_interco_status
    # set_...

    
if __name__ == "__main__":
    unittest.main()
