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
        # this domain is valid even if it is not connected
        with self.assertRaises(DomainException) as de:
            MultiAgentEnv(self.env, action_domains, _add_to_name="test_verify_domains_4")
        
    
    def test_build_subgrids_action_domains(self):
        """Tests that the action_domains are correctly defined 
            in MultiAgentEnv._build_subgrid_from_domain method
        """
        self.ma_env = MultiAgentEnv(self.env, self.action_domains, _add_to_name="test_build_subgrids_action_domains")
        assert self.ma_env._action_domains['agent_0']['sub_id'] == self.action_domains['agent_0']
        assert self.ma_env._action_domains['agent_1']['sub_id'] == self.action_domains['agent_1']
    
    def test_masks(self):
        # We compare the masks with known values
        self.ma_env = MultiAgentEnv(self.env, self.action_domains, _add_to_name="test_masks")
        assert (self.ma_env._action_domains['agent_0']['mask_load'] == [True,  True,  True,  True, False, False, False, 
                                                                        False, False, False, False]).all()
        assert (self.ma_env._action_domains['agent_1']['mask_load'] == np.invert([True,  True,  True,  True, False, False, 
                                                                                  False, False, False, False, False])).all()
        
        assert (self.ma_env._action_domains['agent_0']['mask_gen'] == [ True,  True, False, False, False,  True]).all()
        assert (self.ma_env._action_domains['agent_1']['mask_gen'] == np.invert([ True,  True, False, False, False,  True])).all()
        
        assert (self.ma_env._action_domains['agent_0']['mask_storage'] == []).all()
        assert (self.ma_env._action_domains['agent_1']['mask_storage'] == []).all()
        
        assert (self.ma_env._action_domains['agent_0']['mask_line_ex'] == [ True,  True,  True,  True,  True,  True,  True, 
                                                                           False, False,False, False, False, False, False, False, 
                                                                           False, False, False,False, False]).all()
        assert (self.ma_env._action_domains['agent_1']['mask_line_ex'] == np.invert([ True,  True,  True,  True,  True,  
                                                                                     True,  True, False, False,False,
                                                                                     False, False, False, False, False, 
                                                                                     True, True, True,False, False])).all()
        
        assert (self.ma_env._action_domains['agent_0']['mask_line_or'] == self.ma_env._action_domains['agent_0']['mask_line_ex']).all()
        assert (self.ma_env._action_domains['agent_1']['mask_line_or'] == self.ma_env._action_domains['agent_1']['mask_line_ex']).all()
        
        assert (self.ma_env._action_domains['agent_0']['mask_shunt'] == [False]).all()
        assert (self.ma_env._action_domains['agent_1']['mask_shunt'] == [True]).all()
        
    def test_interco(self):
        
        self.ma_env = MultiAgentEnv(self.env, self.action_domains, _add_to_name="test_interco")
        # Tests on interconnections with known values
        assert (self.ma_env._action_domains['agent_0']['mask_interco'] == [False, False, False, False, False, False, False, False, False,
                                                                            False, False, False, False, False, False,  True,  True,  True,
                                                                            False, False]).all()
        assert (self.ma_env._action_domains['agent_1']['mask_interco'] == [False, False, False, False, False, False, False, False, False,
                                                                            False, False, False, False, False, False,  True,  True,  True,
                                                                            False, False]).all()
        assert (self.ma_env._action_domains['agent_0']['interco_is_origin'] == [True, True, True]).all()
        assert (self.ma_env._action_domains['agent_1']['interco_is_origin'] == np.invert([True, True, True])).all()
        
        # In the two-agent case, they must have the same number of interconnections
        assert self.ma_env._subgrids_cls['action']['agent_0'].n_interco == self.ma_env._subgrids_cls['action']['agent_1'].n_interco
        
        # In the two-agent case, the total number of lines in the global env 
        # must be equal to the number of line in both local grids plus
        # the number of interconnections.
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
        # run redispatch agent on one scenario for 100 timesteps
        ma_env = MultiAgentEnv(self.env, action_domains, _add_to_name="test_build_subgrid_obj")
        
        
        assert ma_env._subgrids_cls['action']['agent_0'].n_gen == 3
        assert ma_env._subgrids_cls['action']['agent_1'].n_gen == 3
        
        self.checks(ma_env, action_domains)
        
        assert (ma_env._subgrids_cls['action']['agent_0'].interco_to_lineid == np.array([15,16,17])).all()
        assert (ma_env._subgrids_cls['action']['agent_1'].interco_to_lineid == np.array([15,16,17])).all()
        
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
        
        self.checks(ma_env, action_domains)

    
    def test_build_subgrid_obj3(self):    
        # 3 random sub ids
        np.random.seed(0)
        for it in range(100):
            sub_ids = list(range(14))
            np.random.shuffle(sub_ids)  # you should see it for reproducible results
            action_domains = {
                'agent_0' : sub_ids[:7],
                'agent_1' : sub_ids[7:],
            }
            # run redispatch agent on one scenario for 100 timesteps
            ma_env = MultiAgentEnv(self.env, action_domains, _add_to_name=f"_it_{it}")
            self.checks(ma_env, action_domains, add_msg=f"error for iter {it}")
            
            
    def checks(self, ma_env, action_domains, add_msg=""):
        # Regroups all the checks to be done
        self.check_orig_ids(ma_env, action_domains)
        self.check_n_objects(ma_env, action_domains, add_msg=add_msg)
        self.check_objects_to_subid(ma_env, action_domains)
        self.check_connections(ma_env, action_domains)
        self.check_shunt(ma_env)
    
    def check_n_objects(self, ma_env, domain, space = 'action', add_msg = ""):
        # Check the number of objects in subgrids. The sum must be equal 
        # to the number in the global grid
        
        # Tests if the sum of object numbers in subgrids is equal to the number of that 
        # object in the original grid.
        assert np.sum([ma_env._subgrids_cls[space][a].n_gen for a in domain.keys()]) == self.env.n_gen, add_msg
        assert np.sum([ma_env._subgrids_cls[space][a].n_load for a in domain.keys()]) == self.env.n_load, add_msg
        assert np.sum([ma_env._subgrids_cls[space][a].n_shunt for a in domain.keys()]) == self.env.n_shunt, add_msg
        assert np.sum([ma_env._subgrids_cls[space][a].n_storage for a in domain.keys()]) == self.env.n_storage, add_msg
        # For interconnections, we concatenate, then, we drop duplicates and we take the length
        assert np.sum([ma_env._subgrids_cls[space][a].n_line for a in domain.keys()])\
             + len(set(np.concatenate([ma_env._subgrids_cls[space][a].interco_to_lineid for a in domain.keys()])))\
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
            
            # Check if sub ids are smaller than the number of substations
            assert (ma_env._subgrids_cls[space][agent].load_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            assert (ma_env._subgrids_cls[space][agent].line_or_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            assert (ma_env._subgrids_cls[space][agent].line_ex_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            assert (ma_env._subgrids_cls[space][agent].storage_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            assert (ma_env._subgrids_cls[space][agent].gen_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            assert (ma_env._subgrids_cls[space][agent].interco_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            if ma_env._subgrids_cls[space][agent].n_shunt:
                assert (ma_env._subgrids_cls[space][agent].shunt_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
                
            # Check if objects are correctly connected
            for subid in range(ma_env._subgrids_cls[space][agent].n_sub):
                dict_connected_objects = ma_env._subgrids_cls[space][agent].get_obj_connect_to(substation_id=subid)
                
                assert (np.where(ma_env._subgrids_cls[space][agent].load_to_subid == subid)\
                    ==\
                    dict_connected_objects["loads_id"]).all()
                
                assert (np.where(ma_env._subgrids_cls[space][agent].gen_to_subid == subid)\
                    ==\
                    dict_connected_objects["generators_id"]).all()
                
                assert (np.where(ma_env._subgrids_cls[space][agent].line_or_to_subid == subid)\
                    ==\
                    dict_connected_objects["lines_or_id"]).all()
                
                assert (np.where(ma_env._subgrids_cls[space][agent].line_ex_to_subid == subid)\
                    ==\
                    dict_connected_objects["lines_ex_id"]).all()
                
                assert (np.where(ma_env._subgrids_cls[space][agent].storage_to_subid == subid)\
                    ==\
                    dict_connected_objects["storages_id"]).all()
            
    def check_orig_ids(self, ma_env, domain : dict, space = 'action'):
        # It tests if the origin ids are correct and
        # we have the same with masks.
        
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
            
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids == np.sort(domain[agent])).all()
            # We check that we have the correct object original ids
            # Ids should be the same as those given by masks 
            assert (ma_env._subgrids_cls[space][agent].gen_orig_ids\
                ==\
                    np.arange(ma_env._cent_env.n_gen)[mask_gen]).all()
            
            assert (ma_env._subgrids_cls[space][agent].load_orig_ids\
                ==\
                    np.arange(ma_env._cent_env.n_load)[mask_load]).all()
            
            assert (ma_env._subgrids_cls[space][agent].storage_orig_ids\
                ==\
                    np.arange(ma_env._cent_env.n_storage)[mask_storage]).all()
            
            assert (ma_env._subgrids_cls[space][agent].line_orig_ids\
                ==\
                    np.arange(ma_env._cent_env.n_line)[mask_line_or]).all()
            
            assert (ma_env._subgrids_cls[space][agent].line_orig_ids\
                ==\
                    np.arange(ma_env._cent_env.n_line)[mask_line_ex]).all()
            
            if ma_env._subgrids_cls[space][agent].n_shunt > 0:
                assert (ma_env._subgrids_cls[space][agent].shunt_orig_ids\
                    ==\
                        np.arange(ma_env._cent_env.n_shunt)[mask_shunt]).all()
                
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
            
            # First, we check if 2 elements are on the same substation 
            # on the original grid, they also are on the same substation 
            # on the subgrid.
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].load_to_subid]\
                ==\
                    ma_env._cent_env.load_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_load
                    ]).all()
            
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].gen_to_subid]\
                ==\
                    ma_env._cent_env.gen_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_gen
                    ]).all()
            
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].storage_to_subid]\
                ==\
                    ma_env._cent_env.storage_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_storage
                    ]).all()
            
            # This test didn't pass with the previous version
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].shunt_to_subid]\
                ==\
                    ma_env._cent_env.shunt_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_shunt
                    ]).all()
            
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].line_ex_to_subid]\
                ==\
                    ma_env._cent_env.line_ex_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_line_ex
                    ]).all()
            
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].line_or_to_subid]\
                ==\
                    ma_env._cent_env.line_or_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_line_or
                    ]).all()
            
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].interco_to_subid][
                            ma_env._subgrids_cls[space][agent].interco_is_origin]\
                ==\
                    ma_env._cent_env.line_or_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_interco
                    ][
                        ma_env._subgrids_cls[space][agent].interco_is_origin
                ]).all()
            
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids[
                        ma_env._subgrids_cls[space][agent].interco_to_subid][
                            ~ma_env._subgrids_cls[space][agent].interco_is_origin]\
                ==\
                    ma_env._cent_env.line_ex_to_subid[
                        ma_env._subgrids_cls[space][agent].mask_interco
                    ][
                        ~ma_env._subgrids_cls[space][agent].interco_is_origin
                ]).all()
            
        # We check that if 2 elements are on the same substation 
        # in the subgrid, they are on the same substation 
        # in the main grid (and vice-versa).
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
            # We check if the original ids of elements in the local dict
            # are the same as the elements in the global dict 
            assert dict_local['nb_elements'] == dict_global['nb_elements']
            assert (dict_global['loads_id']\
                ==\
                    ma_env._subgrids_cls[space][agent].load_orig_ids[dict_local['loads_id']]
            ).all()
            assert (dict_global['generators_id']\
                ==\
                    ma_env._subgrids_cls[space][agent].gen_orig_ids[dict_local['generators_id']]
            ).all()
            
            assert (dict_global['storages_id']\
                ==\
                    ma_env._subgrids_cls[space][agent].storage_orig_ids[dict_local['storages_id']]
            ).all()
            # Interconnections are tricky
            # If there's an interconnection (or more), we take the original line ids 
            # for line_or, line_ex and interco, then, we concatenate and sort them to
            # compare with the original concatenated and sorted line_ex and line_or ids
            if len(dict_local['intercos_id']):
                assert (np.sort(np.concatenate((dict_global['lines_or_id'], dict_global['lines_ex_id'])))\
                    ==\
                        np.sort(np.concatenate((
                            ma_env._subgrids_cls[space][agent].line_orig_ids[dict_local['lines_or_id']],
                            ma_env._subgrids_cls[space][agent].line_orig_ids[dict_local['lines_ex_id']],
                            ma_env._subgrids_cls[space][agent].interco_to_lineid[dict_local['intercos_id']]
                        )))
                ).all()
            else:
                # Otherwise, we simply compare line_ex and line_or original ids
                assert (dict_global['lines_or_id']\
                    ==\
                        ma_env._subgrids_cls[space][agent].line_orig_ids[dict_local['lines_or_id']]
                ).all()
                assert (dict_global['lines_ex_id']\
                    ==\
                        ma_env._subgrids_cls[space][agent].line_orig_ids[dict_local['lines_ex_id']]
                ).all()
                
    def check_shunt(self, ma_env, space = "action"):
        # Test for shunts
        for agent in ma_env.agents:
            assert ma_env._subgrids_cls[space][agent].shunts_data_available \
                ==\
                    self.env.shunts_data_available 
                    
            if ma_env._subgrids_cls[space][agent].n_shunt > 0:
                assert (ma_env._subgrids_cls[space][agent].shunt_orig_ids\
                    ==\
                        np.arange(ma_env._cent_env.n_shunt)[
                            ma_env._subgrids_cls[space][agent].mask_shunt
                        ]
                ).all()
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
    
    
    
if __name__ == "__main__":
    unittest.main()
