# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import grid2op
import numpy as np
from grid2op.Action import PlayableAction

from grid2op.simulator import Simulator
from grid2op.Exceptions import SimulatorError, BaseObservationError

import pdb


class TestSimulator(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True)
        self.env.seed(0)
        self.obs = self.env.reset()
    
    def tearDown(self) -> None:
        self.env.close()
        
    def test_create(self):
        """test i can create them"""
        simulator = Simulator(backend=self.env.backend)
        assert simulator.backend is not self.env.backend
        
        simulator = Simulator(backend=None, env=self.env)
        assert simulator.backend is not self.env.backend
        
        with self.assertRaises(SimulatorError):
            # backend should be a backend
            simulator = Simulator(backend=self.env)
        with self.assertRaises(SimulatorError):
            # backend is not None
            simulator = Simulator(backend=self.env.backend, env=self.env)
        with self.assertRaises(SimulatorError):
            # env is not a BaseEnv
            simulator = Simulator(backend=self.env.backend, env=self.env.backend)

    def test_change_backend(self):
        simulator = Simulator(backend=self.env.backend)
        with self.assertRaises(SimulatorError):
            # not initialized
            simulator.change_backend(self.env.backend.copy())
        
        simulator.set_state(self.obs)
        simulator.change_backend(self.env.backend.copy())
        
        with self.assertRaises(SimulatorError):
            # env is not a BaseEnv
            simulator.change_backend(self.env)
    
    def test_change_backend_type(self):
        simulator = Simulator(backend=self.env.backend)
        with self.assertRaises(SimulatorError):
            # not initialized
            simulator.change_backend_type(self.env.backend.copy(), grid_path=self.env._init_grid_path)
        
        simulator.set_state(self.obs)
        simulator.change_backend_type(self.env._raw_backend_class, grid_path=self.env._init_grid_path)
        
        with self.assertRaises(SimulatorError):
            # self.env.backend is not a type
            simulator.change_backend_type(self.env.backend, grid_path=self.env._init_grid_path) 
        
        with self.assertRaises(SimulatorError):
            # wrong type
            simulator.change_backend_type(type(self.env), grid_path=self.env._init_grid_path) 
    
    def test_predict(self):
        env = self.env
        simulator = Simulator(backend=self.env.backend)
        
        act1 = env.action_space({"set_line_status": [(1, -1)]})
        act2 = env.action_space({"set_bus": {"substations_id": [(5, (2, 1, 2, 1, 2, 1, 2))]}})

        with self.assertRaises(SimulatorError):
            # not initialized
            sim1 = simulator.predict(act1)
        
        simulator.set_state(self.obs)
        
        sim1 = simulator.predict(act1)
        assert sim1 is not simulator
        assert sim1.current_obs.rho[1] == 0.
        
        sim2 = simulator.predict(act2)
        assert sim2 is not simulator
        assert abs(sim2.current_obs.rho[1] -0.35845447) <= 1e-6
        
        sim3 = simulator.predict(act1).predict(act2, do_copy=False)
        assert abs(sim3.current_obs.rho[1]) <= 1e-6
        assert np.any(sim3.current_obs.rho != sim1.current_obs.rho)
        assert np.any(sim3.current_obs.rho != sim2.current_obs.rho)
        assert np.any(sim3.current_obs.rho != simulator.current_obs.rho)
        
        sim4 = simulator.predict(act1,
                                 new_gen_p=env.chronics_handler.real_data.data.prod_p[1],
                                 new_gen_v=env.chronics_handler.real_data.data.prod_v[1],
                                 new_load_p=env.chronics_handler.real_data.data.load_p[1],
                                 new_load_q=env.chronics_handler.real_data.data.load_q[1],
                                 )
        assert sim4 is not simulator
        assert sim4.current_obs.rho[1] == 0.
        assert np.any(sim4.current_obs.rho != sim1.current_obs.rho)
        
        sim5 = sim1.predict(act2)
        assert abs(sim5.current_obs.rho[1]) <= 1e-6
        assert np.max(np.abs(sim5.current_obs.rho - sim3.current_obs.rho)) <= 1e-6
        
        sim6 = simulator.predict(act1, do_copy=False)
        assert sim6 is simulator
        assert abs(sim6.current_obs.rho[1]) <= 1e-6
        assert np.max(np.abs(sim6.current_obs.rho - sim1.current_obs.rho)) <= 1e-6

    def test_copy(self):
        simulator = Simulator(backend=self.env.backend)
        with self.assertRaises(SimulatorError):
            # not initialized
            sim1 = simulator.copy()
            
        simulator.set_state(self.obs)
        sim1 = simulator.copy()
        assert sim1 is not simulator
        assert np.max(np.abs(sim1.current_obs.rho - simulator.current_obs.rho)) <= 1e-6
    
    def test_obs(self):
        simulator = self.obs.get_simulator()
        assert np.max(np.abs(simulator.current_obs.rho - self.obs.rho)) <= 1e-6
        
        with self.assertRaises(BaseObservationError):
            sim2 = simulator.current_obs.get_simulator()

class TestComplexActions(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    action_class=PlayableAction)
        self.env.seed(0)
        self.obs = self.env.reset()
        self.simulator = Simulator(backend=self.env.backend)            
        self.simulator.set_state(self.obs)
    
    def tearDown(self) -> None:
        self.env.close()
        self.simulator.close()
        
    def test_redisp_action(self):
        act = self.env.action_space({"redispatch": [(0, 5.)]})
        obs, *_ = self.env.step(act)
        res = self.simulator.predict(act,
                                     new_gen_p=obs.gen_p - obs.actual_dispatch,
                                     new_load_p=obs.load_p,
                                     new_load_q=obs.load_q)
        assert np.max(np.abs(res.current_obs.target_dispatch - obs.target_dispatch)) <= 1e-5
        assert np.max(np.abs(res.current_obs.actual_dispatch - obs.actual_dispatch)) <= 1e-5
        assert np.max(np.abs(res.current_obs.gen_p - obs.gen_p)) <= 1e-5
        
        act2 = self.env.action_space({"redispatch": [(0, 5.), (1, 4.)]})
        obs2, *_ = self.env.step(act2)
        res2 = res.predict(act2,
                           new_gen_p=obs2.gen_p - obs.actual_dispatch,
                           new_load_p=obs2.load_p,
                           new_load_q=obs2.load_q)
        assert np.max(np.abs(res2.current_obs.target_dispatch - obs2.target_dispatch)) <= 1e-5
        # ultimately the redispatch should match (but not necessarily at this step)
        for _ in range(2):
            obsn, *_ = self.env.step(self.env.action_space())
        assert np.max(np.abs(res2.current_obs.actual_dispatch - obsn.actual_dispatch)) <= 1e-1
        
        act3 = self.env.action_space({"redispatch": [(5, 3.)]})
        obs3, *_ = self.env.step(act3)
        res3 = res2.predict(act3,
                            new_gen_p=obs3.gen_p - obs3.actual_dispatch,
                            new_load_p=obs3.load_p,
                            new_load_q=obs3.load_q)
        assert np.max(np.abs(res3.current_obs.target_dispatch - obs3.target_dispatch)) <= 1e-1
        assert np.max(np.abs(res3.current_obs.actual_dispatch - obs3.actual_dispatch)) <= 0.2
        assert np.max(np.abs(res3.current_obs.gen_p - obs3.gen_p)) <= 0.2
        
    def test_storage(self):
        # TODO
        pass
        
    def test_curtailment(self):
        gen_id = 2
        act = self.env.action_space()  # should curtail 3.4 MW
        act.curtail_mw = [(gen_id, 5.)]
        obs, *_ = self.env.step(act)
        new_gen_p = obs.gen_p - obs.actual_dispatch
        new_gen_p[gen_id] = obs.gen_p_before_curtail[gen_id]
        res = self.simulator.predict(act,
                                     new_gen_p=new_gen_p,
                                     new_load_p=obs.load_p,
                                     new_load_q=obs.load_q)
        assert np.max(np.abs(res.current_obs.target_dispatch - obs.target_dispatch)) <= 1e-5
        assert np.max(np.abs(res.current_obs.actual_dispatch - obs.actual_dispatch)) <= 1e-2
        assert np.max(np.abs(res.current_obs.gen_p - obs.gen_p)) <= 1e-2
        
        # TODO continue that !
        
if __name__ == "__main__":
    unittest.main()
