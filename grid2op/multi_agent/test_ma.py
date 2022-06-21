# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import sys
import pathlib
filepath = pathlib.Path(__file__).resolve().parent.parent.parent
print(filepath)
print()
sys.path.insert(0, str(filepath))
print(sys.path)

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
            self.env = make("l2rpn_case14_sandbox", test=False)

        self.action_domains = {
            'agent_0' : [0,1,2,3, 4],
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        #self.observation_domains = {
        #    'agent_0' : self.action_domains['agent_1'],
        #    'agent_1' : self.action_domains['agent_0']
        #}
        # run redispatch agent on one scenario for 100 timesteps
        # self.ma_env = MultiAgentEnv(self.env, self.action_domains)
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
        
    
    def test_build_subgrids_action_domains(self):
        """Tests that the action_domains are correctly defined 
            in MultiAgentEnv._build_subgrid_from_domain method
        """
        self.ma_env = MultiAgentEnv(self.env, self.action_domains, _add_to_name="test_build_subgrids_action_domains")
        assert self.ma_env._action_domains['agent_0']['sub_id'] == self.action_domains['agent_0']
        assert self.ma_env._action_domains['agent_1']['sub_id'] == self.action_domains['agent_1']
    
    def test_masks(self):
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
        assert (self.ma_env._action_domains['agent_0']['mask_interco'] == [False, False, False, False, False, False, False, False, False,
                                                                            False, False, False, False, False, False,  True,  True,  True,
                                                                            False, False]).all()
        assert (self.ma_env._action_domains['agent_1']['mask_interco'] == [False, False, False, False, False, False, False, False, False,
                                                                            False, False, False, False, False, False,  True,  True,  True,
                                                                            False, False]).all()
        
        assert (self.ma_env._action_domains['agent_0']['interco_is_origin'] == [True, True, True]).all()
        assert (self.ma_env._action_domains['agent_1']['interco_is_origin'] == np.invert([True, True, True])).all()
                
        
    #def test_build_subgrids_observation_domains(self):
    #    """Tests that the observation_domains are correctly defined 
    #        in MultiAgentEnv._build_subgrid_from_domain method
    #    """
    #    assert self.ma_env._observation_domains['agent_1']['sub_id'] == self.action_domains['agent_0']
    #    assert self.ma_env._observation_domains['agent_0']['sub_id'] == self.action_domains['agent_1']
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['mask_load'] == [True,  True,  True,  True, False, False, False, False, False, False, False]).all()
    #    assert (self.ma_env._observation_domains['agent_0']['mask_load'] == np.invert([True,  True,  True,  True, False, False, False, False, False, False, False])).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['mask_gen'] == [True,  True, False, False, False,  True]).all()
    #    assert (self.ma_env._observation_domains['agent_0']['mask_gen'] == np.invert([ True,  True, False, False, False,  True])).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['mask_storage'] == []).all()
    #    assert (self.ma_env._observation_domains['agent_0']['mask_storage'] == []).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['mask_line_ex'] == [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,False, False]).all()
    #    assert (self.ma_env._observation_domains['agent_0']['mask_line_ex'] == np.invert([ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,False, False])).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['mask_line_or'] == self.ma_env._observation_domains['agent_1']['mask_line_ex'] ).all()
    #    assert (self.ma_env._observation_domains['agent_0']['mask_line_or'] == np.invert(self.ma_env._observation_domains['agent_1']['mask_line_ex'])).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['mask_shunt'] == [False]).all()
    #    assert (self.ma_env._observation_domains['agent_0']['mask_shunt'] == [True]).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_0']['mask_interco'] == [False, False, False, False, False, False, False, False, False,
    #                                                                        False, False, False, False, False, False,  True,  True,  True,
    #                                                                        False, False]).all()
    #    assert (self.ma_env._observation_domains['agent_1']['mask_interco'] == [False, False, False, False, False, False, False, False, False,
    #                                                                        False, False, False, False, False, False,  True,  True,  True,
    #                                                                        False, False]).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['interco_is_origin'] == [True, True, True]).all()
    #    assert (self.ma_env._observation_domains['agent_0']['interco_is_origin'] == np.invert([True, True, True])).all()
    

    def test_build_subgrid_obj(self):
        """test the MultiAgentEnv._build_subgrid_obj_from_domain"""
        # TODO test that this function creates an object with the right
        # attributes and the right values from the action / observation
        # domain
        
        # 1
        action_domains = {
            'agent_0' : [0,1,2,3, 4],
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        #observation_domains = {
        #    'agent_0' : self.action_domains['agent_1'],
        #    'agent_1' : self.action_domains['agent_0']
        #}
        # run redispatch agent on one scenario for 100 timesteps
        ma_env = MultiAgentEnv(self.env, action_domains, _add_to_name="test_build_subgrid_obj")
        
        
        assert ma_env._subgrids_cls['action']['agent_0'].n_gen == 3
        assert ma_env._subgrids_cls['action']['agent_1'].n_gen == 3
        
        self.check_orig_ids(ma_env, action_domains)
        self.check_n_objects(ma_env, a0 = 'agent_0', a1 = 'agent_1')
        self.check_objects_to_subid(ma_env, action_domains)

        self.run_in_env(ma_env)
        
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
        
        assert (ma_env._subgrids_cls['action']['agent_0'].grid_objects_types == ref).all()
        
        
    def test_build_subgrid_obj2(self):    
        # 2
        action_domains = {
            'test_2_agent_0' : [0,1,4,5,10,11,12],
            'test_2_agent_1' : [2,3,6,7,8,9,13]
        }
        # TODO run redispatch agent on one scenario for 100 timesteps
        ma_env = MultiAgentEnv(self.env, action_domains, _add_to_name="test_build_subgrid_obj2")
        
        assert ma_env._subgrids_cls['action']['test_2_agent_0'].n_gen == 4
        assert ma_env._subgrids_cls['action']['test_2_agent_1'].n_gen == 2
        
        self.check_orig_ids(ma_env, action_domains)
        self.check_n_objects(ma_env, a0 = 'test_2_agent_0', a1 = 'test_2_agent_1')
        self.check_objects_to_subid(ma_env, action_domains)

        self.run_in_env(ma_env)
        
    
    def test_build_subgrid_obj3(self):    
        # 3 random sub ids
        np.random.seed(0)
        for it in range(100):
            sub_ids = list(range(14))
            np.random.shuffle(sub_ids)  # you should see it for reproducible results
            action_domains = {
                'agent_0' : sub_ids[:7],
                'agent_1' : sub_ids[7:]
            }
            #observation_domains = {
            #    'agent_0' : action_domains['agent_1'],
            #    'agent_1' : action_domains['agent_0']
            #}
            # run redispatch agent on one scenario for 100 timesteps
            ma_env = MultiAgentEnv(self.env, action_domains, _add_to_name=f"_it_{it}")
            
            self.check_connections(ma_env, action_domains)
            self.check_orig_ids(ma_env, action_domains)
            self.check_n_objects(ma_env, a0 = 'agent_0', a1 = 'agent_1', add_msg=f"error for iter {it}")
            self.check_objects_to_subid(ma_env, action_domains)

            # TODO BEN: and check more carefully the things, it's not enough at the moment
        
                    
    # def test_action_space(self):
    #     """test for the action spaces created for agents
    #     """
        
    #     #assert self.ma_env.action_spaces['agent_0'].n_interco == self.ma_env._subgrids_cls['action']['agent_0'].n_interco
    #     for agent in self.action_domains.keys():
    #         assert self.ma_env.action_spaces[agent].n_line == self.ma_env._subgrids_cls['action'][agent].n_line
    #         assert self.ma_env.action_spaces[agent].n_gen == self.ma_env._subgrids_cls['action'][agent].n_gen
    #         assert self.ma_env.action_spaces[agent].n_line == self.ma_env._subgrids_cls['action'][agent].n_line
    #         assert self.ma_env.action_spaces[agent].n_sub == self.ma_env._subgrids_cls['action'][agent].n_sub
            
    #         assert self.ma_env.action_spaces[agent].n_line is type(self.ma_env.action_spaces[agent]).n_line
    #         assert self.ma_env.action_spaces[agent].n_gen is  type(self.ma_env.action_spaces[agent]).n_gen
    #         assert self.ma_env.action_spaces[agent].n_line is type(self.ma_env.action_spaces[agent]).n_line
    #         assert self.ma_env.action_spaces[agent].n_sub is  type(self.ma_env.action_spaces[agent]).n_sub
            
    #         try:
    #             #Simple do nothing action
    #             print(self.ma_env.action_spaces[agent]({}))
    #             #assert True
    #         except Exception as e:
    #             print(f"Exception occured in test_action_space : {e}")

    #         try:
    #             #action on a line
    #             print(self.ma_env.action_spaces[agent]({
    #                 'change_bus' : self.ma_env.action_spaces[agent].line_or_pos_topo_vect[0]
    #             }))
    #             print(self.ma_env.action_spaces['agent_1']({
    #                 'change_bus' : self.ma_env.action_spaces[agent].line_ex_pos_topo_vect[0]
    #             }))
    #             #assert True
    #         except Exception as e:
    #             #assert False
    #             print(f"Exception occured in test_action_space : {e}")

    #         try:
    #             #action on a gen
    #             print(self.ma_env.action_spaces[agent]({
    #                 'change_bus' : self.ma_env.action_spaces[agent].gen_pos_topo_vect[0]
    #             }))
    #             #assert True
    #         except Exception as e:
    #             #assert False
    #             print(f"Exception occured in test_action_space : {e}")

    #         try:
    #             #action on a load
    #             print(self.ma_env.action_spaces[agent]({
    #                 'change_bus' : self.ma_env.action_spaces[agent].load_pos_topo_vect[0]
    #             }))
    #             #assert True
    #         except Exception as e:
    #             #assert False
    #             print(f"Exception occured in test_action_space : {e}")

    #         #try:
    #         #    #action on an interconnection
    #         #    print(self.ma_env.action_spaces[agent]({
    #         #        'change_bus' : self.ma_env.action_spaces[agent].interco_pos_topo_vect[0]
    #         #    }))
    #         #    assert True
    #         #except Exception as e:
    #         #    assert False
    
    
    def check_n_objects(self, ma_env, a0, a1, space = 'action', add_msg = ""):
        assert ma_env._subgrids_cls[space][a0].n_gen +  ma_env._subgrids_cls[space][a1].n_gen == self.env.n_gen, add_msg
        assert ma_env._subgrids_cls[space][a0].n_load +  ma_env._subgrids_cls[space][a1].n_load == self.env.n_load, add_msg
        assert ma_env._subgrids_cls[space][a0].n_interco == ma_env._subgrids_cls[space][a1].n_interco, add_msg
        assert ma_env._subgrids_cls[space][a0].n_line \
             + ma_env._subgrids_cls[space][a1].n_line \
             + ma_env._subgrids_cls[space][a0].n_interco == self.env.n_line, add_msg
             
        assert np.sum(ma_env._subgrids_cls[space][a0].sub_info)\
            ==\
            ma_env._subgrids_cls[space][a0].n_gen+\
            ma_env._subgrids_cls[space][a0].n_load+\
            ma_env._subgrids_cls[space][a0].n_line*2+\
            ma_env._subgrids_cls[space][a0].n_interco, add_msg
            
        assert np.sum(ma_env._subgrids_cls[space][a1].sub_info)\
            ==\
            ma_env._subgrids_cls[space][a1].n_gen+\
            ma_env._subgrids_cls[space][a1].n_load+\
            ma_env._subgrids_cls[space][a1].n_line*2+\
            ma_env._subgrids_cls[space][a1].n_interco, add_msg
            
        
        assert len(ma_env._subgrids_cls[space][a0].line_ex_to_subid) == ma_env._subgrids_cls[space][a0].n_line, add_msg
        assert len(ma_env._subgrids_cls[space][a0].line_or_to_subid) == ma_env._subgrids_cls[space][a0].n_line, add_msg
        
        assert len(ma_env._subgrids_cls[space][a1].line_ex_to_subid) == ma_env._subgrids_cls[space][a1].n_line, add_msg
        assert len(ma_env._subgrids_cls[space][a1].line_or_to_subid) == ma_env._subgrids_cls[space][a1].n_line, add_msg
        
    def check_objects_to_subid(self, ma_env, domain, space = 'action'):
        
        # Verifies if sub ids are correct   
        for agent in domain.keys():
         
            assert (ma_env._subgrids_cls[space][agent].load_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            assert (ma_env._subgrids_cls[space][agent].line_or_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            assert (ma_env._subgrids_cls[space][agent].line_ex_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            assert (ma_env._subgrids_cls[space][agent].storage_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            assert (ma_env._subgrids_cls[space][agent].gen_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            assert (ma_env._subgrids_cls[space][agent].interco_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
            if ma_env._subgrids_cls[space][agent].n_shunt:
                assert (ma_env._subgrids_cls[space][agent].shunt_to_subid < ma_env._subgrids_cls[space][agent].n_sub).all()
                
            
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
        
        for agent in domain.keys():
            assert (ma_env._subgrids_cls[space][agent].sub_orig_ids == np.sort(domain[agent])).all()
            
    def check_connections(self, ma_env, action_domains, space = 'action'):
        pass
            
    def run_in_env(self, ma_env):
        #TODO
        pass
    
    
    
    
    # TODO BEN: use a function to check all that more generally (eg the one you coded for test_build_subgrid_obj4)
    # TODO BEN and this generic function you test EVERYTHING, not half of it.
    # TODO BEN by everything I mean really every class attributes
        
if __name__ == "__main__":
    unittest.main()
