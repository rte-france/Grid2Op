# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings
import numpy as np
import pdb
import os

from grid2op.tests.helper_path_test import *
from grid2op.Exceptions import NoForecastAvailable
from grid2op.Chronics import MultifolderWithCache
    
import grid2op
import numpy as np


class MultiStepsForcaTester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(os.path.join(PATH_DATA_TEST, "5bus_example_forecasts"), test=True,
                                    _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
    
    def aux_test_for_consistent(self, obs):
        tmp_o_1, *_ = obs.simulate(self.env.action_space(),
                                   time_step=1)
        assert (obs.load_p + 1. == tmp_o_1.load_p).all()  # that's how I generated the forecast for this "env"
        tmp_o_2, *_ = obs.simulate(self.env.action_space(),
                                   time_step=2)
        assert (obs.load_p + 2. == tmp_o_2.load_p).all()  # that's how I generated the forecast for this "env"
        tmp_o_3, *_ = obs.simulate(self.env.action_space(),
                                   time_step=3)
        assert (obs.load_p + 3. == tmp_o_3.load_p).all()
        tmp_o_12, *_ = obs.simulate(self.env.action_space(),
                                    time_step=12)
        assert (obs.load_p + 12. == tmp_o_12.load_p).all()
        
    def test_can_do(self):
        obs = self.env.reset()
        self.aux_test_for_consistent(obs)
        
        # should raise because there is no "13 steps ahead forecasts"
        with self.assertRaises(NoForecastAvailable):
            obs.simulate(self.env.action_space(),
                         time_step=13)
        
        # check it's still consistent
        obs, *_ = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        # check it's still consistent
        obs, *_ = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
            
    def test_chunk_size(self):
        self.env.set_chunk_size(1)
        obs = self.env.reset()
        self.aux_test_for_consistent(obs)
        
        # check it's still consistent
        obs, *_ = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
            
        # check it's still consistent
        obs, *_ = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        # check it's still consistent
        obs, *_ = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
    
    def test_max_iter(self):
        max_iter = 4
        self.env.chronics_handler.set_max_iter(max_iter)
        
        obs = self.env.reset()
        self.aux_test_for_consistent(obs)
        
        # check it's still consistent
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        assert done        

    def test_cache(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(os.path.join(PATH_DATA_TEST, "5bus_example_forecasts"),
                               test=True,
                               chronics_class=MultifolderWithCache,
                               _add_to_name=type(self).__name__)
            
        env.seed(0)
        env.set_id(0)
        env.chronics_handler.reset()
        
        obs = self.env.reset()
        self.aux_test_for_consistent(obs)
        
        # check it's still consistent
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)

    def test_cooldowns(self):
        obs = self.env.reset()
        dn = self.env.action_space()
        act = self.env.action_space({"set_bus": {"substations_id": [(2, (2, 1, 2, 1))]}})
        
        # check it properly applies in "simulate"
        tmp_o_1, *_ = obs.simulate(act, time_step=1)
        assert tmp_o_1.time_before_cooldown_sub[2] == 3
        tmp_o_2, *_ = obs.simulate(act, time_step=2)
        assert tmp_o_2.time_before_cooldown_sub[2] == 3
        tmp_o_3, *_ = obs.simulate(act, time_step=3)
        assert tmp_o_3.time_before_cooldown_sub[2] == 3
        tmp_o_12, *_ = obs.simulate(act, time_step=12)
        assert tmp_o_12.time_before_cooldown_sub[2] == 3 
        
        # check if a cooldown exists it is properly changed in simulate
        obs2, reward, done, info = self.env.step(act)
        tmp_o_1_2, *_ = obs2.simulate(dn, time_step=1)
        assert tmp_o_1_2.time_before_cooldown_sub[2] == 2      
        tmp_o_2_2, *_ = obs2.simulate(dn, time_step=2)
        assert tmp_o_2_2.time_before_cooldown_sub[2] == 1        
        tmp_o_3_2, *_ = obs2.simulate(dn, time_step=3)
        assert tmp_o_3_2.time_before_cooldown_sub[2] == 0        
        tmp_o_12_2, *_ = obs2.simulate(dn, time_step=12)
        assert tmp_o_12_2.time_before_cooldown_sub[2] == 0        
        
        # check if a cooldown exists it is properly changed in simulate
        obs3, reward, done, info = self.env.step(dn)
        tmp_o_1_3, *_ = obs3.simulate(dn, time_step=1)
        assert tmp_o_1_3.time_before_cooldown_sub[2] == 1      
        tmp_o_2_3, *_ = obs3.simulate(dn, time_step=2)
        assert tmp_o_2_3.time_before_cooldown_sub[2] == 0        
        tmp_o_3_3, *_ = obs3.simulate(dn, time_step=3)
        assert tmp_o_3_3.time_before_cooldown_sub[2] == 0        
        tmp_o_12_3, *_ = obs3.simulate(dn, time_step=12)
        assert tmp_o_12_3.time_before_cooldown_sub[2] == 0        
    
    def test_maintenance(self):        
        obs = self.env.reset()   # no maintenance
        obs = self.env.reset()  # maintenance
        dn = self.env.action_space()
        assert obs.time_next_maintenance[5] == 6        
        assert obs.duration_next_maintenance[5] == 4    
        
        # check it properly applies in "simulate"
        obs_1, *_ = self.env.step(dn)
        assert obs_1.time_next_maintenance[5] == 5
        assert obs.time_next_maintenance[5] == 6  
        tmp_o_1, reward, done, info = obs.simulate(dn, time_step=1)
        assert not done
        assert tmp_o_1.time_next_maintenance[5] == 5
        
        obs_2, *_ = self.env.step(dn)
        assert obs_2.time_next_maintenance[5] == 4
        tmp_o_2, *_ = obs.simulate(dn, time_step=2)
        assert tmp_o_2.time_next_maintenance[5] == 4
        
        obs_3, *_ = self.env.step(dn)
        assert obs_3.time_next_maintenance[5] == 3        
        tmp_o_3, *_ = obs.simulate(dn, time_step=3)
        assert tmp_o_3.time_next_maintenance[5] == 3

        obs_4, *_ = self.env.step(dn)
        assert obs_4.time_next_maintenance[5] == 2           
        tmp_o_4, *_ = obs.simulate(dn, time_step=4)
        assert tmp_o_4.time_next_maintenance[5] == 2
        
        obs_5, *_ = self.env.step(dn)
        assert obs_5.time_next_maintenance[5] == 1          
        tmp_o_5, *_ = obs.simulate(dn, time_step=5)
        assert tmp_o_5.time_next_maintenance[5] == 1
        
        # first corner case: line should be disconnected (first step)
        obs_6, *_ = self.env.step(dn)
        assert obs_6.time_next_maintenance[5] == 0 
        assert obs_6.duration_next_maintenance[5] == 4
        tmp_o_6, *_ = obs.simulate(dn, time_step=6)
        assert tmp_o_6.time_next_maintenance[5] == 0
        assert tmp_o_6.duration_next_maintenance[5] == 4
        assert not tmp_o_6.line_status[5] 
        
        # now the "duration next maintenance" should decrease of 1
        tmp_o_7, *_ = obs.simulate(dn, time_step=7)
        assert tmp_o_7.time_next_maintenance[5] == 0
        assert tmp_o_7.duration_next_maintenance[5] == 3
        assert not tmp_o_7.line_status[5] 
        
        tmp_o_8, *_ = obs.simulate(dn, time_step=8)
        assert tmp_o_8.time_next_maintenance[5] == 0
        assert tmp_o_8.duration_next_maintenance[5] == 2
        assert not tmp_o_8.line_status[5] 
        
        tmp_o_9, *_ = obs.simulate(dn, time_step=9)
        assert tmp_o_9.time_next_maintenance[5] == 0
        assert tmp_o_9.duration_next_maintenance[5] == 1
        assert not tmp_o_9.line_status[5] 
        
        # second corner case: line should not be modified 
        # maintenance is totally 'skiped' : forecast horizon is after maintenance
        # occured
        tmp_o_10, *_ = obs.simulate(dn, time_step=10)
        assert tmp_o_10.time_next_maintenance[5] == -1
        assert tmp_o_10.duration_next_maintenance[5] == 0
        assert tmp_o_10.line_status[5] 

        tmp_o_12, *_ = obs.simulate(dn, time_step=12)
        assert tmp_o_12.time_next_maintenance[5] == -1
        assert tmp_o_12.duration_next_maintenance[5] == 0
        assert tmp_o_12.line_status[5] 
        
        
class ChainSimulateTester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(os.path.join(PATH_DATA_TEST, "5bus_example_forecasts"), test=True,
                                    _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
    
    def aux_test_for_consistent_independant(self, obs, tmp_o, h):
        assert (obs.load_p + 1. * h == tmp_o.load_p).all()
        tmp_o_1, *_ = obs.simulate(self.env.action_space(),
                                   time_step=h)
        assert (obs.load_p + 1. * h == tmp_o_1.load_p).all()
         
    def test_can_chain_independant(self):
        obs = self.env.reset()
        tmp_o_1, *_ = obs.simulate(self.env.action_space(),
                                   time_step=1)
        self.aux_test_for_consistent_independant(obs, tmp_o_1, 1)
        
        tmp_o_2, *_ = tmp_o_1.simulate(self.env.action_space(),
                                       time_step=1)
        self.aux_test_for_consistent_independant(obs, tmp_o_2, 2)
        _ = tmp_o_1.simulate(self.env.action_space(),
                             time_step=11)
        with self.assertRaises(NoForecastAvailable):
            # not available
            tmp_o_2, *_ = tmp_o_1.simulate(self.env.action_space(),
                                           time_step=12)
        
        tmp_o_3, *_ = tmp_o_2.simulate(self.env.action_space(),
                                       time_step=1)
        self.aux_test_for_consistent_independant(obs, tmp_o_3, 3)
        _ = tmp_o_2.simulate(self.env.action_space(),
                             time_step=10)
        with self.assertRaises(NoForecastAvailable):
            # not available
            tmp_o_2, *_ = tmp_o_2.simulate(self.env.action_space(),
                                           time_step=11)
    
    def test_can_chain_dependant(self):
        obs = self.env.reset()
        dn = self.env.action_space()
        # if I do nothing it's like it's independant
        tmp_o_1, *_ = obs.simulate(dn, time_step=1)
        self.aux_test_for_consistent_independant(obs, tmp_o_1, 1)
        
        # check that it's not independant
        act = self.env.action_space({"set_bus": {"substations_id": [(2, (2, 1, 2, 1))]}})
        tmp_o_1_1, *_ = obs.simulate(act, time_step=1)
        assert (tmp_o_1_1.topo_vect[[9, 11]] == [2, 2]).all()
        tmp_o_2_1, *_ = tmp_o_1_1.simulate(dn, time_step=1)
        assert (tmp_o_2_1.topo_vect[[9, 11]] == [2, 2]).all()
        # check that the original simulate is not "broken"
        tmp_o_1_base, *_ = obs.simulate(dn, time_step=1)
        assert (tmp_o_1_base.topo_vect[[9, 11]] == [1, 1]).all()
        
        # and to be sure, check that it's independant if i put the flag
        # it's surprising that it works TODO !
        act = self.env.action_space({"set_bus": {"substations_id": [(2, (2, 1, 2, 1))]}})
        tmp_o_1_2, *_ = obs.simulate(act, time_step=1)
        assert (tmp_o_1_2.topo_vect[[9, 11]] == [2, 2]).all()
        # check that the original simulate is not "broken"
        tmp_o_1_base, *_ = obs.simulate(dn, time_step=1)
        assert (tmp_o_1_base.topo_vect[[9, 11]] == [1, 1]).all()
        # check 2nd simulate is indpendant of first one
        tmp_o_2_2, *_ = tmp_o_1_2.simulate(dn, time_step=1)
        assert (tmp_o_2_2.topo_vect[[9, 11]] == [2, 2]).all()
        # check that the original simulate is not "broken"
        tmp_o_1_base, *_ = obs.simulate(dn, time_step=1)
        assert (tmp_o_1_base.topo_vect[[9, 11]] == [1, 1]).all()
    
    def test_cooldown_when_chained(self):
        obs = self.env.reset()
        dn = self.env.action_space()
                
        act = self.env.action_space({"set_bus": {"substations_id": [(2, (2, 1, 2, 1))]}})
        tmp_o_1_1, *_ = obs.simulate(act, time_step=1)
        assert (tmp_o_1_1.topo_vect[[9, 11]] == [2, 2]).all()
        assert tmp_o_1_1.time_before_cooldown_sub[2] == 3
        
        tmp_o_2, *_ = tmp_o_1_1.simulate(dn, time_step=1)
        assert tmp_o_2.time_before_cooldown_sub[2] == 2
        
        tmp_o_2_2, reward, done, info = tmp_o_1_1.simulate(act, time_step=1)
        assert info["is_illegal"]
        assert tmp_o_2_2.time_before_cooldown_sub[2] == 2
        
        tmp_o_3, r, done, info = tmp_o_2.simulate(dn, time_step=1)
        assert not done
        assert tmp_o_3.time_before_cooldown_sub[2] == 1
        
        tmp_o_4, r, done, info = tmp_o_3.simulate(dn, time_step=1)
        assert not done
        assert tmp_o_4.time_before_cooldown_sub[2] == 0
        
        tmp_o_4_2, r, done, info = tmp_o_3.simulate(act, time_step=1)
        assert not done
        assert info["is_illegal"]  # because cooldown > 0 for tmp_o_3
        assert tmp_o_4_2.time_before_cooldown_sub[2] == 0
        
        tmp_o_5, reward, done, info = tmp_o_4.simulate(act, time_step=1)
        assert not done
        assert not info["is_illegal"]
        assert tmp_o_5.time_before_cooldown_sub[2] == 3
        
    def test_maintenance(self):        
        obs = self.env.reset()   # no maintenance
        obs = self.env.reset()  # maintenance
        dn = self.env.action_space()
        assert obs.time_next_maintenance[5] == 6        
        assert obs.duration_next_maintenance[5] == 4    
        
        # check it properly applies in "simulate"
        obs_1, *_ = self.env.step(dn)
        assert obs_1.time_next_maintenance[5] == 5
        assert obs.time_next_maintenance[5] == 6  
        tmp_o_1, reward, done, info = obs.simulate(dn, time_step=1)
        assert not done
        assert tmp_o_1.time_next_maintenance[5] == 5
        
        obs_2, *_ = self.env.step(dn)
        assert obs_2.time_next_maintenance[5] == 4
        tmp_o_2, *_ = tmp_o_1.simulate(dn, time_step=1)
        assert tmp_o_2.time_next_maintenance[5] == 4
        
        obs_3, *_ = self.env.step(dn)
        assert obs_3.time_next_maintenance[5] == 3        
        tmp_o_3, *_ = tmp_o_2.simulate(dn, time_step=1)
        assert tmp_o_3.time_next_maintenance[5] == 3

        obs_4, *_ = self.env.step(dn)
        assert obs_4.time_next_maintenance[5] == 2           
        tmp_o_4, *_ = tmp_o_3.simulate(dn, time_step=1)
        assert tmp_o_4.time_next_maintenance[5] == 2
        
        obs_5, *_ = self.env.step(dn)
        assert obs_5.time_next_maintenance[5] == 1          
        tmp_o_5, *_ = tmp_o_4.simulate(dn, time_step=1)
        assert tmp_o_5.time_next_maintenance[5] == 1
        
        # first corner case: line should be disconnected (first step)
        obs_6, *_ = self.env.step(dn)
        assert obs_6.time_next_maintenance[5] == 0 
        assert obs_6.duration_next_maintenance[5] == 4
        tmp_o_6, *_ = tmp_o_5.simulate(dn, time_step=1)
        assert tmp_o_6.time_next_maintenance[5] == 0
        assert tmp_o_6.duration_next_maintenance[5] == 4
        assert not tmp_o_6.line_status[5] 
        
        # now the "duration next maintenance" should decrease of 1
        tmp_o_7, *_ = tmp_o_6.simulate(dn, time_step=1)
        assert tmp_o_7.time_next_maintenance[5] == 0
        assert tmp_o_7.duration_next_maintenance[5] == 3
        assert not tmp_o_7.line_status[5] 
        
        tmp_o_8, *_ = tmp_o_7.simulate(dn, time_step=1)
        assert tmp_o_8.time_next_maintenance[5] == 0
        assert tmp_o_8.duration_next_maintenance[5] == 2
        assert not tmp_o_8.line_status[5] 
        
        tmp_o_9, *_ = tmp_o_8.simulate(dn, time_step=1)
        assert tmp_o_9.time_next_maintenance[5] == 0
        assert tmp_o_9.duration_next_maintenance[5] == 1
        assert not tmp_o_9.line_status[5] 
        
        # second corner case: line should not be modified 
        # maintenance is totally 'skiped' : forecast horizon is after maintenance
        # occured
        tmp_o_10, *_ = tmp_o_9.simulate(dn, time_step=1)
        assert tmp_o_10.time_next_maintenance[5] == -1
        assert tmp_o_10.duration_next_maintenance[5] == 0
        # no reason to reconnect the line automatically
        assert not tmp_o_10.line_status[5] 

        tmp_o_12, *_ = tmp_o_10.simulate(dn, time_step=2)
        assert tmp_o_12.time_next_maintenance[5] == -1
        assert tmp_o_12.duration_next_maintenance[5] == 0
        # no reason to reconnect the line automatically
        assert not tmp_o_12.line_status[5] 


class SoftOverflowTester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(os.path.join(PATH_DATA_TEST, "5bus_example_forecasts"), test=True,
                                    _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
        a_or_first = np.array([442.308, 198.55365, 116.50534,  93.63006,
                               442.2703 , 110.96754, 110.96754,  92.05039])
        th_lim = a_or_first * 2.
        th_lim[5] /= 2.5
        self.env.set_thermal_limit(th_lim)
        param = self.env.parameters
        param.NO_OVERFLOW_DISCONNECTION = False
        param.NB_TIMESTEP_RECONNECTION = 3
        self.env.change_parameters(param)
        self.env.change_forecast_parameters(param)
        
    def test_my_env_is_correct(self):
        obs = self.env.reset()
        assert obs.timestep_overflow[5] == 0
        
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.timestep_overflow[5] == 1
        assert obs.line_status[5]
        
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.timestep_overflow[5] == 2
        assert obs.line_status[5]
        
        # now it should be disconnected
        obs, *_ = self.env.step(self.env.action_space())
        assert not obs.line_status[5]
        assert obs.timestep_overflow[5] == 0
    
    def test_simulate_multi_h(self):
        reco = self.env.action_space({"set_line_status": [(5, +1)]})
        obs = self.env.reset()
        assert obs.timestep_overflow[5] == 0
        
        sim_o1, *_ = obs.simulate(self.env.action_space(), 1)
        assert sim_o1.line_status[5]
        assert sim_o1.timestep_overflow[5] == 1
        
        sim_o2, *_ = obs.simulate(self.env.action_space(), 2)
        assert sim_o2.line_status[5]
        assert sim_o2.timestep_overflow[5] == 2
        
        # now it should be disconnected
        sim_o3, *_ = obs.simulate(self.env.action_space(), 3)
        assert not sim_o3.line_status[5]
        assert sim_o3.timestep_overflow[5] == 0
        assert sim_o3.time_before_cooldown_line[5] == 3  # because it's like that in the parameters
        
        # still disconnected
        sim_o4, *_ = obs.simulate(self.env.action_space(), 4)
        assert not sim_o4.line_status[5]
        assert sim_o4.timestep_overflow[5] == 0
        # assert sim_o4.time_before_cooldown_line[5] == 2

        # still disconnected
        sim_o5, *_ = obs.simulate(self.env.action_space(), 5)
        assert not sim_o5.line_status[5]
        assert sim_o5.timestep_overflow[5] == 0
        # assert sim_o5.time_before_cooldown_line[5] == 1
        
        # still disconnected
        sim_o6, *_ = obs.simulate(self.env.action_space(), 6)
        assert not sim_o6.line_status[5]
        assert sim_o6.timestep_overflow[5] == 0
        # assert sim_o6.time_before_cooldown_line[5] == 0
        
        # still disconnected
        sim_o7, *_ = obs.simulate(self.env.action_space(), 7)
        assert not sim_o7.line_status[5]
        assert sim_o7.timestep_overflow[5] == 0
        # assert sim_o7.time_before_cooldown_line[5] == 0
                
        # still disconnected
        sim_o12, *_ = obs.simulate(self.env.action_space(), 12)
        assert not sim_o12.line_status[5]
        # assert sim_o12.time_before_cooldown_line[5] == 0
        
    def test_simulate_chained(self):
        reco = self.env.action_space({"set_line_status": [(5, +1)]})
        obs = self.env.reset()
        assert obs.timestep_overflow[5] == 0
        
        sim_o1, *_ = obs.simulate(self.env.action_space(), 1)
        assert sim_o1.line_status[5]
        assert sim_o1.timestep_overflow[5] == 1
        
        sim_o2, *_ = sim_o1.simulate(self.env.action_space(), 1)
        assert sim_o2.line_status[5]
        assert sim_o2.timestep_overflow[5] == 2
        
        # now it should be disconnected
        sim_o3, *_ = sim_o2.simulate(self.env.action_space(), 1)
        assert not sim_o3.line_status[5]
        assert sim_o3.timestep_overflow[5] == 0
        assert sim_o3.time_before_cooldown_line[5] == 3  # because it's like that in the parameters
        
        # still disconnected
        sim_o4, *_ = sim_o3.simulate(self.env.action_space(), 1)
        assert not sim_o4.line_status[5]
        assert sim_o4.timestep_overflow[5] == 0
        # assert sim_o4.time_before_cooldown_line[5] == 2

        # still disconnected
        sim_o5, *_ = sim_o4.simulate(self.env.action_space(), 1)
        assert not sim_o5.line_status[5]
        assert sim_o5.timestep_overflow[5] == 0
        # assert sim_o5.time_before_cooldown_line[5] == 1
        
        # still disconnected
        sim_o6, *_ = sim_o5.simulate(self.env.action_space(), 1)
        assert not sim_o6.line_status[5]
        assert sim_o6.timestep_overflow[5] == 0
        # assert sim_o6.time_before_cooldown_line[5] == 0
        
        # still disconnected, but I can reconnect it if I want
        sim_o7, *_ = sim_o6.simulate(self.env.action_space(), 1)
        assert not sim_o7.line_status[5]
        assert sim_o7.timestep_overflow[5] == 0
        # assert sim_o7.time_before_cooldown_line[5] == 0
        
                
if __name__ == "__main__":
    unittest.main()
