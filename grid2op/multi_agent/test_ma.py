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



class MATester(unittest.TestCase):
    def setUp(self) -> None:
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("l2rpn_case14_sandbox", test = True)

        self.action_domains = {
            'agent_0' : [0,1,2,3, 4],
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        #self.observation_domains = {
        #    'agent_0' : self.action_domains['agent_1'],
        #    'agent_1' : self.action_domains['agent_0']
        #}
        # run redispatch agent on one scenario for 100 timesteps
        self.ma_env = MultiAgentEnv(self.env, self.action_domains)
        return super().setUp()
    
    def test_verify_domains(self):
        action_domains = {
            'agent_0' : 0,
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        #observation_domains = action_domains
        #try:
        #    MultiAgentEnv(self.env, observation_domains, action_domains)
        #    assert False
        #    
        #except DomainException :
        #    assert True
            
        with self.assertRaises(DomainException) as de:
            MultiAgentEnv(self.env, action_domains)
        
        action_domains = {
            'agent_0' : [0],
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        #observation_domains = action_domains
        with self.assertRaises(DomainException) as de:
            MultiAgentEnv(self.env, action_domains)
            
        action_domains = {
            'agent_0' : [],
            'agent_1' : list(range(self.env.n_sub))
        }
        #observation_domains = action_domains
        with self.assertRaises(DomainException) as de:
            MultiAgentEnv(self.env, action_domains)
        
    
    # TODO Test in case subs are not connex

    # TODO 
    
    def test_build_subgrids_action_domains(self):
        """Tests that the action_domains are correctly defined 
            in MultiAgentEnv._build_subgrid_from_domain method
        """
        assert self.ma_env._action_domains['agent_0']['sub_id'] == self.action_domains['agent_0']
        assert self.ma_env._action_domains['agent_1']['sub_id'] == self.action_domains['agent_1']
    
    def test_masks(self):
        
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
    
    # TODO
    def test_validate_action_domain(self):
        """test the MultiAgentEnv._verify_domains method """
        # TODO it should test that :
        # 1) the function works (does not throw an error) when the input domains are correct
        # 2) the function throws an error when the input domains are wrong
        # (the more "wrong" cases tested the better)
        pass

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
        env = make("l2rpn_case14_sandbox", test = False, _add_to_name="test1")
        ma_env = MultiAgentEnv(env, action_domains)
        
        #print(ma_env._subgrids_cls['action']['agent_0'].n_line)
        #print(ma_env._subgrids_cls['action']['agent_1'].n_line)
        #print(ma_env._subgrids_cls['action']['agent_0'].n_interco)
        #print(self.env.n_line)
        #print(ma_env._subgrids_cls['action']['agent_0'])
        assert ma_env._subgrids_cls['action']['agent_0'].n_gen == 3
        assert ma_env._subgrids_cls['action']['agent_1'].n_gen == 3
        #assert ma_env._subgrids_cls['action']['agent_0'].n_gen is not ma_env._subgrids_cls['action']['agent_1'].n_gen
        #assert ma_env._subgrids_cls['action']['agent_0'].n_load is not ma_env._subgrids_cls['action']['agent_1'].n_load
        #assert ma_env._subgrids_cls['action']['agent_0'].n_line is not ma_env._subgrids_cls['action']['agent_1'].n_line
        #assert ma_env._subgrids_cls['action']['agent_0'].n_interco is not ma_env._subgrids_cls['action']['agent_1'].n_interco
        
        assert ma_env._subgrids_cls['action']['agent_0'].n_gen +  ma_env._subgrids_cls['action']['agent_1'].n_gen == self.env.n_gen
        assert ma_env._subgrids_cls['action']['agent_0'].n_load +  ma_env._subgrids_cls['action']['agent_1'].n_load == self.env.n_load
        assert ma_env._subgrids_cls['action']['agent_0'].n_interco == ma_env._subgrids_cls['action']['agent_1'].n_interco
        assert ma_env._subgrids_cls['action']['agent_0'].n_line \
             + ma_env._subgrids_cls['action']['agent_1'].n_line \
             + ma_env._subgrids_cls['action']['agent_0'].n_interco == self.env.n_line
            
        assert self.env.n_gen is type(self.env).n_gen
        #assert ma_env._subgrids_cls['action']['agent_0'].n_gen is type(ma_env._subgrids_cls['action']['agent_0']).n_gen 
        #assert ma_env._subgrids_cls['action']['agent_1'].n_gen is type(ma_env._subgrids_cls['action']['agent_1']).n_gen
        assert ma_env._subgrids_cls['action']['agent_0'].n_load +  ma_env._subgrids_cls['action']['agent_1'].n_load == self.env.n_load
        assert ma_env._subgrids_cls['action']['agent_0'].n_interco == ma_env._subgrids_cls['action']['agent_1'].n_interco
        assert ma_env._subgrids_cls['action']['agent_0'].n_line \
             + ma_env._subgrids_cls['action']['agent_1'].n_line \
             + ma_env._subgrids_cls['action']['agent_0'].n_interco == self.env.n_line
        
        assert (ma_env._subgrids_cls['action']['agent_0'].interco_to_lineid == np.array([15,16,17])).all()
        assert (ma_env._subgrids_cls['action']['agent_1'].interco_to_lineid == np.array([15,16,17])).all()
        
        assert (ma_env._subgrids_cls['action']['agent_0'].grid_objects_types == np.array([[ 0., -1., -1.,  0., -1., -1.],
                                                                                          [ 0., -1., -1.,  1., -1., -1.],
                                                                                          [ 0., -1.,  2., -1., -1., -1.],
                                                                                          [ 1., -1., -1., -1.,  0., -1.],
                                                                                          [ 1., -1., -1.,  2., -1., -1.],
                                                                                          [ 1., -1., -1.,  3., -1., -1.],
                                                                                          [ 1., -1., -1.,  4., -1., -1.],
                                                                                          [ 1., -1.,  0., -1., -1., -1.],
                                                                                          [ 1.,  0., -1., -1., -1., -1.],
                                                                                          [ 2., -1., -1., -1.,  2., -1.],
                                                                                          [ 2., -1., -1.,  5., -1., -1.],
                                                                                          [ 2., -1.,  1., -1., -1., -1.],
                                                                                          [ 2.,  1., -1., -1., -1., -1.],
                                                                                          [ 3., -1., -1., -1.,  3., -1.],
                                                                                          [ 3., -1., -1., -1.,  5., -1.],
                                                                                          [ 3., -1., -1.,  6., -1., -1.],
                                                                                          [ 3., -1., -1., -1., -1.,  0.],
                                                                                          [ 3., -1., -1., -1., -1.,  1.],
                                                                                          [ 3.,  2., -1., -1., -1., -1.],
                                                                                          [ 4., -1., -1., -1.,  1., -1.],
                                                                                          [ 4., -1., -1., -1.,  4., -1.],
                                                                                          [ 4., -1., -1., -1.,  6., -1.],
                                                                                          [ 4., -1., -1., -1., -1.,  2.],
                                                                                          [ 4.,  3., -1., -1., -1., -1.]])).all()
        
        
    def test_build_subgrid_obj2(self):    
        # 2
        action_domains = {
            'test_2_agent_0' : [0,1,4,5,10,11,12],
            'test_2_agent_1' : [2,3,6,7,8,9,13]
        }
        #observation_domains = {
        #    'agent_0' : action_domains['agent_1'],
        #    'agent_1' : action_domains['agent_0']
        #}
        # run redispatch agent on one scenario for 100 timesteps
        ma_env = MultiAgentEnv(self.env, action_domains)
        
        assert ma_env._subgrids_cls['action']['test_2_agent_0'].n_gen == 4
        assert ma_env._subgrids_cls['action']['test_2_agent_1'].n_gen == 2
        assert ma_env._subgrids_cls['action']['test_2_agent_0'].n_gen + ma_env._subgrids_cls['action']['test_2_agent_1'].n_gen == self.env.n_gen
        assert ma_env._subgrids_cls['action']['test_2_agent_0'].n_load + ma_env._subgrids_cls['action']['test_2_agent_1'].n_load == self.env.n_load
        assert ma_env._subgrids_cls['action']['test_2_agent_0'].n_interco == ma_env._subgrids_cls['action']['test_2_agent_1'].n_interco
        assert ma_env._subgrids_cls['action']['test_2_agent_0'].n_line \
             + ma_env._subgrids_cls['action']['test_2_agent_1'].n_line \
             + ma_env._subgrids_cls['action']['test_2_agent_0'].n_interco == self.env.n_line
    
    def test_build_subgrid_obj4(self):    
        # 4 random sub ids
        for i in range(1000):
            sub_ids = list(range(14))
            np.random.shuffle(sub_ids)
            action_domains = {
                'agent_0' : sub_ids[:7],
                'agent_1' : sub_ids[7:]
            }
            #observation_domains = {
            #    'agent_0' : action_domains['agent_1'],
            #    'agent_1' : action_domains['agent_0']
            #}
            # run redispatch agent on one scenario for 100 timesteps
            env = make("l2rpn_case14_sandbox", test = False, _add_to_name=f"test_random_{i}")
            print(i)
            ma_env = MultiAgentEnv(env, action_domains)

            assert ma_env._subgrids_cls['action']['agent_0'].n_gen + ma_env._subgrids_cls['action']['agent_1'].n_gen == self.env.n_gen
            assert ma_env._subgrids_cls['action']['agent_0'].n_load + ma_env._subgrids_cls['action']['agent_1'].n_load == self.env.n_load
            assert ma_env._subgrids_cls['action']['agent_0'].n_shunt + ma_env._subgrids_cls['action']['agent_1'].n_shunt == self.env.n_shunt
            assert ma_env._subgrids_cls['action']['agent_0'].n_interco == ma_env._subgrids_cls['action']['agent_1'].n_interco
            assert ma_env._subgrids_cls['action']['agent_0'].n_line \
                 + ma_env._subgrids_cls['action']['agent_1'].n_line \
                 + ma_env._subgrids_cls['action']['agent_0'].n_interco == self.env.n_line
                
            assert len(ma_env._subgrids_cls['action']['agent_0'].line_ex_to_subid) == ma_env._subgrids_cls['action']['agent_0'].n_line
            assert len(ma_env._subgrids_cls['action']['agent_0'].line_or_to_subid) == ma_env._subgrids_cls['action']['agent_0'].n_line
            
            assert len(ma_env._subgrids_cls['action']['agent_1'].line_ex_to_subid) == ma_env._subgrids_cls['action']['agent_1'].n_line
            assert len(ma_env._subgrids_cls['action']['agent_1'].line_or_to_subid) == ma_env._subgrids_cls['action']['agent_1'].n_line
            
            assert np.sum(ma_env._subgrids_cls['action']['agent_0'].sub_info)\
                ==\
                ma_env._subgrids_cls['action']['agent_0'].n_gen+\
                ma_env._subgrids_cls['action']['agent_0'].n_load+\
                ma_env._subgrids_cls['action']['agent_0'].n_line*2+\
                ma_env._subgrids_cls['action']['agent_0'].n_interco
                
            assert np.sum(ma_env._subgrids_cls['action']['agent_1'].sub_info)\
                ==\
                ma_env._subgrids_cls['action']['agent_1'].n_gen+\
                ma_env._subgrids_cls['action']['agent_1'].n_load+\
                ma_env._subgrids_cls['action']['agent_1'].n_line*2+\
                ma_env._subgrids_cls['action']['agent_1'].n_interco
            
            
            # Verifies if sub ids are correct    
            assert (ma_env._subgrids_cls['action']['agent_0'].load_to_subid < ma_env._subgrids_cls['action']['agent_0'].n_sub).all()
            assert (ma_env._subgrids_cls['action']['agent_0'].line_or_to_subid < ma_env._subgrids_cls['action']['agent_0'].n_sub).all()
            assert (ma_env._subgrids_cls['action']['agent_0'].line_ex_to_subid < ma_env._subgrids_cls['action']['agent_0'].n_sub).all()
            assert (ma_env._subgrids_cls['action']['agent_0'].storage_to_subid < ma_env._subgrids_cls['action']['agent_0'].n_sub).all()
            assert (ma_env._subgrids_cls['action']['agent_0'].gen_to_subid < ma_env._subgrids_cls['action']['agent_0'].n_sub).all()
            assert (ma_env._subgrids_cls['action']['agent_0'].interco_to_subid < ma_env._subgrids_cls['action']['agent_0'].n_sub).all()
            if ma_env._subgrids_cls['action']['agent_0'].n_shunt:
                assert (ma_env._subgrids_cls['action']['agent_0'].shunt_to_subid < ma_env._subgrids_cls['action']['agent_0'].n_sub).all()
            
            assert (ma_env._subgrids_cls['action']['agent_1'].load_to_subid < ma_env._subgrids_cls['action']['agent_1'].n_sub).all()
            assert (ma_env._subgrids_cls['action']['agent_1'].line_or_to_subid < ma_env._subgrids_cls['action']['agent_1'].n_sub).all()
            assert (ma_env._subgrids_cls['action']['agent_1'].line_ex_to_subid < ma_env._subgrids_cls['action']['agent_1'].n_sub).all()
            assert (ma_env._subgrids_cls['action']['agent_1'].storage_to_subid < ma_env._subgrids_cls['action']['agent_1'].n_sub).all()
            assert (ma_env._subgrids_cls['action']['agent_1'].gen_to_subid < ma_env._subgrids_cls['action']['agent_1'].n_sub).all()
            assert (ma_env._subgrids_cls['action']['agent_1'].interco_to_subid < ma_env._subgrids_cls['action']['agent_1'].n_sub).all()
            if ma_env._subgrids_cls['action']['agent_1'].n_shunt:
                assert (ma_env._subgrids_cls['action']['agent_1'].shunt_to_subid < ma_env._subgrids_cls['action']['agent_1'].n_sub).all()
           
            
    def test_build_subgrid_obj3(self):    
        # 3
        action_domains = {
            'test_3_agent_0' : [0,1,2,3,4,5,6,7],
            'test_3_agent_1' : [8,9,10,11,12,13]
        }
        #observation_domains = {
        #    'agent_0' : action_domains['agent_1'],
        #    'agent_1' : action_domains['agent_0']
        #}
        # run redispatch agent on one scenario for 100 timesteps
        ma_env = MultiAgentEnv(self.env, action_domains)
        assert ma_env._subgrids_cls['action']['test_3_agent_0'].n_gen + ma_env._subgrids_cls['action']['test_3_agent_1'].n_gen == self.env.n_gen
        assert ma_env._subgrids_cls['action']['test_3_agent_0'].n_load + ma_env._subgrids_cls['action']['test_3_agent_1'].n_load == self.env.n_load
        assert ma_env._subgrids_cls['action']['test_3_agent_0'].n_shunt + ma_env._subgrids_cls['action']['test_3_agent_1'].n_shunt == self.env.n_shunt
        assert ma_env._subgrids_cls['action']['test_3_agent_0'].n_interco == ma_env._subgrids_cls['action']['test_3_agent_1'].n_interco
        assert ma_env._subgrids_cls['action']['test_3_agent_0'].n_line \
             + ma_env._subgrids_cls['action']['test_3_agent_1'].n_line \
             + ma_env._subgrids_cls['action']['test_3_agent_0'].n_interco == self.env.n_line
            
        assert len(ma_env._subgrids_cls['action']['test_3_agent_0'].line_ex_to_subid) == ma_env._subgrids_cls['action']['test_3_agent_0'].n_line
        assert len(ma_env._subgrids_cls['action']['test_3_agent_0'].line_or_to_subid) == ma_env._subgrids_cls['action']['test_3_agent_0'].n_line
        
        assert len(ma_env._subgrids_cls['action']['test_3_agent_1'].line_ex_to_subid) == ma_env._subgrids_cls['action']['test_3_agent_1'].n_line
        assert len(ma_env._subgrids_cls['action']['test_3_agent_1'].line_or_to_subid) == ma_env._subgrids_cls['action']['test_3_agent_1'].n_line
        
        assert np.sum(ma_env._subgrids_cls['action']['test_3_agent_0'].sub_info)\
            ==\
            ma_env._subgrids_cls['action']['test_3_agent_0'].n_gen+\
            ma_env._subgrids_cls['action']['test_3_agent_0'].n_load+\
            ma_env._subgrids_cls['action']['test_3_agent_0'].n_line*2+\
            ma_env._subgrids_cls['action']['test_3_agent_0'].n_interco
            
        assert np.sum(ma_env._subgrids_cls['action']['test_3_agent_1'].sub_info)\
            ==\
            ma_env._subgrids_cls['action']['test_3_agent_1'].n_gen+\
            ma_env._subgrids_cls['action']['test_3_agent_1'].n_load+\
            ma_env._subgrids_cls['action']['test_3_agent_1'].n_line*2+\
            ma_env._subgrids_cls['action']['test_3_agent_1'].n_interco
        
        
        # Verifies if sub ids are correct    
        assert (ma_env._subgrids_cls['action']['test_3_agent_0'].load_to_subid < ma_env._subgrids_cls['action']['test_3_agent_0'].n_sub).all()
        assert (ma_env._subgrids_cls['action']['test_3_agent_0'].line_or_to_subid < ma_env._subgrids_cls['action']['test_3_agent_0'].n_sub).all()
        assert (ma_env._subgrids_cls['action']['test_3_agent_0'].line_ex_to_subid < ma_env._subgrids_cls['action']['test_3_agent_0'].n_sub).all()
        assert (ma_env._subgrids_cls['action']['test_3_agent_0'].storage_to_subid < ma_env._subgrids_cls['action']['test_3_agent_0'].n_sub).all()
        assert (ma_env._subgrids_cls['action']['test_3_agent_0'].gen_to_subid < ma_env._subgrids_cls['action']['test_3_agent_0'].n_sub).all()
        assert (ma_env._subgrids_cls['action']['test_3_agent_0'].interco_to_subid < ma_env._subgrids_cls['action']['test_3_agent_0'].n_sub).all()
        if ma_env._subgrids_cls['action']['test_3_agent_0'].n_shunt:
            assert (ma_env._subgrids_cls['action']['test_3_agent_0'].shunt_to_subid < ma_env._subgrids_cls['action']['test_3_agent_0'].n_sub).all()
        
        assert (ma_env._subgrids_cls['action']['test_3_agent_1'].load_to_subid < ma_env._subgrids_cls['action']['test_3_agent_1'].n_sub).all()
        assert (ma_env._subgrids_cls['action']['test_3_agent_1'].line_or_to_subid < ma_env._subgrids_cls['action']['test_3_agent_1'].n_sub).all()
        assert (ma_env._subgrids_cls['action']['test_3_agent_1'].line_ex_to_subid < ma_env._subgrids_cls['action']['test_3_agent_1'].n_sub).all()
        assert (ma_env._subgrids_cls['action']['test_3_agent_1'].storage_to_subid < ma_env._subgrids_cls['action']['test_3_agent_1'].n_sub).all()
        assert (ma_env._subgrids_cls['action']['test_3_agent_1'].gen_to_subid < ma_env._subgrids_cls['action']['test_3_agent_1'].n_sub).all()
        assert (ma_env._subgrids_cls['action']['test_3_agent_1'].interco_to_subid < ma_env._subgrids_cls['action']['test_3_agent_1'].n_sub).all()
        if ma_env._subgrids_cls['action']['test_3_agent_1'].n_shunt:
            assert (ma_env._subgrids_cls['action']['test_3_agent_1'].shunt_to_subid < ma_env._subgrids_cls['action']['test_3_agent_1'].n_sub).all()


            
            
    def test_action_space(self):
        """test for the action spaces created for agents
        """
        
        #assert self.ma_env.action_spaces['agent_0'].n_interco == self.ma_env._subgrids_cls['action']['agent_0'].n_interco
        for agent in self.action_domains.keys():
            assert self.ma_env.action_spaces[agent].n_line == self.ma_env._subgrids_cls['action'][agent].n_line
            assert self.ma_env.action_spaces[agent].n_gen == self.ma_env._subgrids_cls['action'][agent].n_gen
            assert self.ma_env.action_spaces[agent].n_line == self.ma_env._subgrids_cls['action'][agent].n_line
            assert self.ma_env.action_spaces[agent].n_sub == self.ma_env._subgrids_cls['action'][agent].n_sub
            
            assert self.ma_env.action_spaces[agent].n_line is type(self.ma_env.action_spaces[agent]).n_line
            assert self.ma_env.action_spaces[agent].n_gen is  type(self.ma_env.action_spaces[agent]).n_gen
            assert self.ma_env.action_spaces[agent].n_line is type(self.ma_env.action_spaces[agent]).n_line
            assert self.ma_env.action_spaces[agent].n_sub is  type(self.ma_env.action_spaces[agent]).n_sub
            
            try:
                #Simple do nothing action
                print(self.ma_env.action_spaces[agent]({}))
                #assert True
            except Exception as e:
                print(f"Exception occured in test_action_space : {e}")

            try:
                #action on a line
                print(self.ma_env.action_spaces[agent]({
                    'change_bus' : self.ma_env.action_spaces[agent].line_or_pos_topo_vect[0]
                }))
                print(self.ma_env.action_spaces['agent_1']({
                    'change_bus' : self.ma_env.action_spaces[agent].line_ex_pos_topo_vect[0]
                }))
                #assert True
            except Exception as e:
                #assert False
                print(f"Exception occured in test_action_space : {e}")

            try:
                #action on a gen
                print(self.ma_env.action_spaces[agent]({
                    'change_bus' : self.ma_env.action_spaces[agent].gen_pos_topo_vect[0]
                }))
                #assert True
            except Exception as e:
                #assert False
                print(f"Exception occured in test_action_space : {e}")

            try:
                #action on a load
                print(self.ma_env.action_spaces[agent]({
                    'change_bus' : self.ma_env.action_spaces[agent].load_pos_topo_vect[0]
                }))
                #assert True
            except Exception as e:
                #assert False
                print(f"Exception occured in test_action_space : {e}")

            #try:
            #    #action on an interconnection
            #    print(self.ma_env.action_spaces[agent]({
            #        'change_bus' : self.ma_env.action_spaces[agent].interco_pos_topo_vect[0]
            #    }))
            #    assert True
            #except Exception as e:
            #    assert False
    
if __name__ == "__main__":
    unittest.main()
