# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import grid2op
from grid2op.Agent import BaseAgent
from grid2op.Backend import PandaPowerBackend
from grid2op.Runner import Runner
import unittest
import warnings
import numpy as np
import pdb


class ModifPPBackend(PandaPowerBackend):
    pass
        
        
class RememberRX(BaseAgent):
    def act(self, observation, reward, done=False):
        self._x = 1.0 * observation._obs_env.backend._grid.line["x_ohm_per_km"]
        self._r = 1.0 * observation._obs_env.backend._grid.line["r_ohm_per_km"]
        self._detailed_infos_for_cascading_failures = observation._obs_env.backend.detailed_infos_for_cascading_failures
        self._lightsim2grid = observation._obs_env.backend._lightsim2grid
        self._max_iter = observation._obs_env.backend._max_iter
        self._bk = observation._obs_env.backend
        return self.action_space()
    
    def __copy__(self):
        # prevent copy
        raise copy.Error
    
    
class Case14DiffGridTester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_case14_sandbox_diff_grid", test=True)
        self.env.seed(0)
        self.env.set_id(0)
    
    def test_backend_different(self):
        assert  self.env.backend._grid is not self.env.observation_space._backend_obs
        x_orig = self.env.backend._grid.line["x_ohm_per_km"]
        x_modif = self.env.observation_space._backend_obs._grid.line["x_ohm_per_km"]
        assert np.all(x_orig != x_modif)
        r_orig = self.env.backend._grid.line["r_ohm_per_km"]
        r_modif = self.env.observation_space._backend_obs._grid.line["r_ohm_per_km"]
        assert np.all(r_orig != r_modif)
    
    def test_simulate(self):
        obs_init = self.env.reset()
        loadp_next = [22.1, 89. , 45.9,  6.9, 12.3, 28.4,  8.8,  3.4,  5.5, 12.9, 15.2]
        loadq_next = [15.5, 62.1, 31.2,  4.9,  8.4, 19.8,  6.1,  2.4,  3.8,  8.9, 10.6]
        genp_next = [73.3   , 72.6   , 36.6   ,  0.    ,  0.    , 71.5158]
        act = self.env.action_space()
        # set the observation to the right values (for the forecast)
        obs_init._forecasted_inj[1][1]["injection"]["load_p"] = np.array(loadp_next).astype(np.float32)
        obs_init._forecasted_inj[1][1]["injection"]["load_q"] = np.array(loadq_next).astype(np.float32)
        obs_init._forecasted_inj[1][1]["injection"]["prod_p"] = np.array(genp_next).astype(np.float32)
        if 1 in obs_init._forecasted_grid_act:
            del obs_init._forecasted_grid_act[1]
        sim_obs, sim_r, simd, sim_i = obs_init.simulate(act)
        obs, reward, done, info = self.env.step(act)
        # inputs are the same
        assert np.all(sim_obs.load_p == obs.load_p)
        assert np.all(sim_obs.load_q == obs.load_q)
        assert np.all(sim_obs.gen_p[:-1] == obs.gen_p[:-1])  # all equals except the slack
        # and now check the outputs of the backend are different
        assert sim_obs.gen_p[-1] != obs.gen_p[-1]  # slack different
        assert np.all(sim_obs.p_or != obs.p_or)
        assert np.all(sim_obs.q_or != obs.q_or)
        assert np.all(sim_obs.a_or != obs.a_or)
    
    def test_simulator(self):
        obs = self.env.reset()
        sim = obs.get_simulator()
        assert  self.env.backend._grid is not sim.backend
        x_orig = self.env.backend._grid.line["x_ohm_per_km"]
        x_modif = sim.backend._grid.line["x_ohm_per_km"]
        assert np.all(x_orig != x_modif)
        r_orig = self.env.backend._grid.line["r_ohm_per_km"]
        r_modif = sim.backend._grid.line["r_ohm_per_km"]
        assert np.all(r_orig != r_modif)
    
    def test_forecasted_env(self):
        obs = self.env.reset()
        act = self.env.action_space()
        
        for_env = obs.get_forecast_env()
        assert  self.env.backend._grid is not for_env.backend
        x_orig = self.env.backend._grid.line["x_ohm_per_km"]
        x_modif = for_env.backend._grid.line["x_ohm_per_km"]
        assert np.all(x_orig != x_modif)
        r_orig = self.env.backend._grid.line["r_ohm_per_km"]
        r_modif = for_env.backend._grid.line["r_ohm_per_km"]
        assert np.all(r_orig != r_modif)
        
        sim_obs, sim_r, simd, sim_i = obs.simulate(act)
        for_obs, for_r, for_d, for_i = for_env.step(act)
        assert np.all(sim_obs.a_or == for_obs.a_or)
    
    def test_thermal_limit(self):
        obs = self.env.reset()
        sim = obs.get_simulator()
        for_env = obs.get_forecast_env()
        assert np.all(sim.backend.get_thermal_limit() == self.env.get_thermal_limit())
        assert np.all(for_env.get_thermal_limit() == self.env.get_thermal_limit())
        assert np.all(obs._obs_env.get_thermal_limit() == self.env.get_thermal_limit())
        new_th_lim = 2.0 * self.env.get_thermal_limit()
        
        self.env.set_thermal_limit(new_th_lim)
        obs = self.env.reset()
        sim = obs.get_simulator()
        for_env = obs.get_forecast_env()
        assert np.all(sim.backend.get_thermal_limit() == new_th_lim)
        assert np.all(for_env.get_thermal_limit() == new_th_lim)
        assert np.all(obs._obs_env.get_thermal_limit() == new_th_lim)
        
    def test_runner(self):
        agent = RememberRX(self.env.action_space)
        runner = Runner(**self.env.get_params_for_runner(), agentInstance=agent, agentClass=None)
        _ = runner.run(nb_episode=1, max_iter=1)
        obs = self.env.reset()
        assert hasattr(agent, "_r")
        assert np.all(agent._r == obs._obs_env.backend._grid.line["r_ohm_per_km"])
        assert np.all(agent._r != self.env.backend._grid.line["r_ohm_per_km"])
        assert hasattr(agent, "_x")
        assert np.all(agent._x == obs._obs_env.backend._grid.line["x_ohm_per_km"])
        assert np.all(agent._x != self.env.backend._grid.line["x_ohm_per_km"])
        
        
class Case14DiffGridCopyTester(Case14DiffGridTester):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.aux_env = grid2op.make("l2rpn_case14_sandbox_diff_grid", test=True)
            
        self.env = self.aux_env.copy()
        self.env.seed(0)
        self.env.set_id(0)


class DiffGridMakeTester(unittest.TestCase):      
    
    def _aux_check_bk_kwargs(self, bk):
        assert bk._lightsim2grid
        assert bk._max_iter == 15
    
    def _aux_check_different_stuff(self, env, fun_bk):        
        obs = env.reset()
        fun_bk(obs._obs_env.backend)
        
        # copy
        env_cpy = env.copy()
        obs_cpy = env_cpy.reset()
        fun_bk(obs_cpy._obs_env.backend)
        
        # runner
        agent = RememberRX(env.action_space)
        runner = Runner(**env.get_params_for_runner(), agentInstance=agent, agentClass=None)
        _ = runner.run(nb_episode=1, max_iter=1)
        fun_bk(agent)
        
        # forecasted_env
        for_env = obs.get_forecast_env()
        fun_bk(for_env.backend)
        
        # simulator
        sim = obs.get_simulator()
        fun_bk(sim.backend)
              
    def test_bk_kwargs(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            env = grid2op.make("l2rpn_case14_sandbox_diff_grid", test=True,
                               observation_backend_kwargs={"max_iter": 15,
                                                           "lightsim2grid": True})
        self._aux_check_different_stuff(env, self._aux_check_bk_kwargs)
    
    def _aux_bk_class(self, bk):
        if isinstance(bk, PandaPowerBackend):
            # subtlety: it can be called with an agent for the test of runner...
            assert isinstance(bk, ModifPPBackend)
        else:
            # in this case "bk" is in fact an agent...
            assert isinstance(bk._bk, ModifPPBackend)
        
    def test_bk_class(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            env = grid2op.make("l2rpn_case14_sandbox_diff_grid", test=True,
                               observation_backend_class=ModifPPBackend)
        self._aux_check_different_stuff(env, self._aux_bk_class)


if __name__ == "__main__":
    unittest.main()
