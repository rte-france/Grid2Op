# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/rte-france/Grid2Op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict
from gymnasium.vector import AsyncVectorEnv
import warnings
import numpy as np
from multiprocessing import set_start_method

import grid2op
from grid2op.Action import PlayableAction
from grid2op.gym_compat import GymEnv, BoxGymActSpace, BoxGymObsSpace, DiscreteActSpace, MultiDiscreteActSpace


class AsyncGymEnvTester_Fork(unittest.TestCase):
    def _aux_start_method(self):
        return "fork"
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    _add_to_name=type(self).__name__,
                                    action_class=PlayableAction,
                                    experimental_read_from_local_dir=True)
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        return super().setUp()
    
    def test_default_space_obs_act(self):
        template_env = GymEnv(self.env)
        template_env.action_space.seed(0)
        obs = template_env.reset(seed=0, options={"time serie id": 0})
        async_vect_env = AsyncVectorEnv((lambda: GymEnv(self.env), lambda: GymEnv(self.env)),
                                        context=self._aux_start_method())
        assert isinstance(async_vect_env.action_space, Dict)
        assert isinstance(async_vect_env.observation_space, Dict)
        obs, infos = async_vect_env.reset(seed=[0, 1],
                                          options={"time serie id": 0})

        dn_act_single = template_env.action_space.sample()
        for k, v in dn_act_single.items():
            v[:] = 0
        dn_acts = {k: np.tile(v, reps=[2, 1]) for k, v in dn_act_single.items()}       
        obs2 = async_vect_env.step(dn_acts)
        
        rnd_acts_li = [template_env.action_space.sample(), template_env.action_space.sample()]
        rnd_acts = {k: np.concatenate((rnd_acts_li[0][k], rnd_acts_li[1][k])) for k in rnd_acts_li[0].keys()} 
        obs3 = async_vect_env.step(rnd_acts)
        
        obs, infos = async_vect_env.reset(seed=[2, 3],
                                          options={"time serie id": 0},
                                          )
    
    def _aux_obs_act_vect(self, ts_id=0):
        gym_env = GymEnv(self.env)
        gym_env.action_space.close()
        gym_env.action_space = BoxGymActSpace(self.env.action_space, attr_to_keep=["redispatch", "curtail"])
        gym_env.observation_space.close()
        gym_env.observation_space = BoxGymObsSpace(self.env.observation_space, attr_to_keep=["rho"])
        gym_env.action_space.seed(0)
        _ = gym_env.reset(seed=0, options={"time serie id": ts_id})
        return gym_env
        
    def test_space_obs_act_vect(self):
        template_env = self._aux_obs_act_vect(0)
        async_vect_env = AsyncVectorEnv((lambda: self._aux_obs_act_vect(1),
                                         lambda: self._aux_obs_act_vect(2)),
                                        context=self._aux_start_method())
        try:
            assert isinstance(async_vect_env.action_space, Box)
            assert isinstance(async_vect_env.observation_space, Box)
            obs, infos = async_vect_env.reset(seed=[0, 1],
                                            options={"time serie id": 0})

            dn_act_single = template_env.action_space.sample()
            dn_act_single[:] = 0
            dn_acts = np.tile(dn_act_single, reps=[2, 1])    
            obs2 = async_vect_env.step(dn_acts)
            
            rnd_acts_li = [template_env.action_space.sample().reshape(1,-1), template_env.action_space.sample().reshape(1,-1)]
            rnd_acts = np.concatenate(rnd_acts_li)  
            obs3 = async_vect_env.step(rnd_acts)
            
            obs, infos = async_vect_env.reset(seed=[2, 3],
                                            options={"time serie id": 0},
                                            )
        finally:
            async_vect_env.close()
            template_env.close()
    
    def _aux_obs_vect_act_discrete(self, ts_id=0):
        gym_env = GymEnv(self.env)
        gym_env.observation_space.close()
        gym_env.observation_space = BoxGymObsSpace(self.env.observation_space, attr_to_keep=["rho"])
        gym_env.action_space.close()
        gym_env.action_space = DiscreteActSpace(self.env.action_space, attr_to_keep=["set_bus"])
        gym_env.action_space.seed(0)
        _ = gym_env.reset(seed=0, options={"time serie id": ts_id})
        return gym_env
        
    def test_space_obs_vect_act_discrete(self):
        template_env = self._aux_obs_vect_act_discrete(0)
        assert isinstance(template_env.action_space, Discrete)
        async_vect_env = AsyncVectorEnv((lambda: self._aux_obs_vect_act_discrete(1),
                                         lambda: self._aux_obs_vect_act_discrete(2)),
                                        context=self._aux_start_method())
        try:
            assert isinstance(async_vect_env.action_space, MultiDiscrete)  # converted to MultiDiscrete by gymnasium
            assert isinstance(async_vect_env.observation_space, Box)
            obs, infos = async_vect_env.reset(seed=[0, 1],
                                            options={"time serie id": 0})

            dn_act_single = 0
            dn_acts = np.tile(dn_act_single, reps=[2, 1])    
            obs2 = async_vect_env.step(dn_acts)
            
            rnd_acts_li = [template_env.action_space.sample().reshape(1,-1), template_env.action_space.sample().reshape(1,-1)]
            rnd_acts = np.concatenate(rnd_acts_li)  
            obs3 = async_vect_env.step(rnd_acts)
            
            obs, infos = async_vect_env.reset(seed=[2, 3],
                                            options={"time serie id": 0},
                                            )
        finally:
            async_vect_env.close()
            template_env.close()
    
    def _aux_obs_vect_act_multidiscrete(self, ts_id=0):
        gym_env = GymEnv(self.env)
        gym_env.observation_space.close()
        gym_env.observation_space = BoxGymObsSpace(self.env.observation_space, attr_to_keep=["rho"])
        gym_env.action_space.close()
        gym_env.action_space = MultiDiscreteActSpace(self.env.action_space, attr_to_keep=["one_sub_set", "one_line_set"])
        gym_env.action_space.seed(0)
        _ = gym_env.reset(seed=0, options={"time serie id": ts_id})
        return gym_env
        
    def test_space_obs_vect_act_multidiscrete(self):
        template_env = self._aux_obs_vect_act_multidiscrete(0)
        assert isinstance(template_env.action_space, MultiDiscrete)
        async_vect_env = AsyncVectorEnv((lambda: self._aux_obs_vect_act_multidiscrete(1),
                                         lambda: self._aux_obs_vect_act_multidiscrete(2)),
                                        context=self._aux_start_method())
        try:
            assert isinstance(async_vect_env.action_space, Box)  # converted to Box by gymnasium
            assert isinstance(async_vect_env.observation_space, Box)
            obs, infos = async_vect_env.reset(seed=[0, 1],
                                            options={"time serie id": 0})

            dn_act_single = template_env.action_space.sample()
            dn_act_single[:] = 0
            dn_acts = np.tile(dn_act_single, reps=[2, 1])    
            obs2 = async_vect_env.step(dn_acts)
            
            rnd_acts_li = [template_env.action_space.sample().reshape(1,-1), template_env.action_space.sample().reshape(1,-1)]
            rnd_acts = np.concatenate(rnd_acts_li)  
            obs3 = async_vect_env.step(rnd_acts)
            
            obs, infos = async_vect_env.reset(seed=[2, 3],
                                            options={"time serie id": 0},
                                            )
        finally:
            async_vect_env.close()
            template_env.close()
            
            
# class AsyncGymEnvTester_Spawn(AsyncGymEnvTester_Fork):
# Will be working when branch class_in_files will be merged
#     def _aux_start_method(self):
#         return "spawn"
    
    
if __name__ == "__main__":
    unittest.main()
