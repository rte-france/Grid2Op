# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import warnings

import grid2op
import unittest
from grid2op.Chronics import FromHandlers
from grid2op.Chronics.handlers import PerfectForecastHandler, CSVHandler


class Issue598Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                        "gen_p_handler": CSVHandler("prod_p"),
                                        "load_p_handler": CSVHandler("load_p"),
                                        "gen_v_handler": CSVHandler("prod_v"),
                                        "load_q_handler": CSVHandler("load_q"),
                                        "gen_p_for_handler": PerfectForecastHandler("prod_p_forecasted"),
                                        "load_p_for_handler": PerfectForecastHandler("load_p_forecasted"),
                                        "load_q_for_handler": PerfectForecastHandler("load_q_forecasted"),
                                        "h_forecast": (5, 10, 15, 20, 25, 30),
                                        }
                    )
        params = self.env.parameters
        params.ACTIVATE_STORAGE_LOSS = False
        self.env.change_parameters(params)
        self.env.change_forecast_parameters(params)
        self.env.reset(seed=0, options={"time serie id": 0})
        self.dn = self.env.action_space()
        return super().setUp()
    
    def test_issue_598_dn_same_res(self):
        """no redisp: simu = step"""
        obs, *_ = self.env.step(self.dn)
        obs_simulate, *_ = obs.simulate(self.dn, time_step=0)
        #no redispatch action yet, the productions are the same after simulation on the same state
        assert (np.abs(obs_simulate.prod_p - obs.prod_p) <= 1e-6).all()
        
    def test_issue_598_dn_redisp(self, redisp_amout=2., storage_amount=None):
        """one small redispatch action before simulation and then a single simulation of do nothing
        on current step (default args)"""
        self.skipTest("Not sure it should be equal, waiting for the model in the issue 598")
        act = self.env.action_space()
        if redisp_amout is not None:
            act.redispatch = {"gen_2_1": redisp_amout}
        if storage_amount is not None:
            act.storage_p = [(1, 10.)]

        obs, reward, done, info = self.env.step(act)
        obs, reward, done, info = self.env.step(self.dn)
        assert not done
        assert not info["is_ambiguous"]
        assert not info["is_illegal"]
        print("here here here")
        obs_simulate, *_ = obs.simulate(self.dn, time_step=0)
        # obs.gen_p.sum() no redisp: 262.86395
        assert (np.abs(obs_simulate.prod_p - obs.prod_p) <= 1e-6).all(), f"{obs_simulate.prod_p} vs {obs.prod_p}"
        assert (np.abs(obs_simulate.storage_charge - obs.storage_charge) <= 1e-6).all()
        assert (np.abs(obs_simulate.storage_power - obs.storage_power) <= 1e-6).all()
        assert (np.abs(obs_simulate.storage_power_target - obs.storage_power_target) <= 1e-6).all()

    def test_simulate_ok_current_step_redisp_large(self):
        self.skipTest("Does not pass: redisp is not 'limited' by the simulate in this case")
        self.test_issue_598_dn_redisp(redisp_amout=-10.)
        
    def test_simulate_ok_current_step_storage(self):
        """doing a storage action. Then a simulate on current step lead to the same result"""
        self.skipTest("Does not pass: storage is not the same in this case (nothing in simulate, but something in the env)")
        # self.skipTest("Does not pass: curtailment is not 'limited' by the simulate in this case")
        self.test_issue_598_dn_redisp(redisp_amout=None, storage_amount=10.)  
                
    def test_simulate_step_redisp_before(self, redisp_amout=2., storage_amount=None):
        """one small redispatch before simulation and then lots of simulation of do nothing (default args)"""
        act = self.env.action_space()
        if redisp_amout is not None:
            act.redispatch = {"gen_2_1": redisp_amout}
        if storage_amount is not None:
            act.storage_p = [(1, 10.)]
        obs, *_ = self.env.step(act)
        next_obs = obs
        for time_step in [1, 2, 3, 4, 5, 6]:
            obs_simulate, *_ = next_obs.simulate(self.dn, time_step=1)
            next_obs, *_ = self.env.step(self.dn)
            assert (np.abs(obs_simulate.prod_p - next_obs.prod_p) <= 1e-6).all(), f"for h={time_step}: {obs_simulate.prod_p} vs {next_obs.prod_p}"
            assert (np.abs(obs_simulate.storage_charge - next_obs.storage_charge) <= 1e-6).all()
            assert (np.abs(obs_simulate.storage_power - next_obs.storage_power) <= 1e-6).all()
            assert (np.abs(obs_simulate.storage_power_target - next_obs.storage_power_target) <= 1e-6).all()
        
    def test_simulate_step_redisp_before_from_init(self, redisp_amout=2., storage_amount=None):
        """one small redispatch and then lots of simulation of do nothing (increasing the forecast horizon)
        (default args)
        """
        act = self.env.action_space()
        if redisp_amout is not None:
            act.redispatch = {"gen_2_1": redisp_amout}
        if storage_amount is not None:
            act.storage_p = [(1, 10.)]
        obs, *_ = self.env.step(act)
        for time_step in [1, 2, 3, 4, 5, 6]:
            obs_simulate, *_ = obs.simulate(self.dn, time_step=time_step)
            next_obs, *_ = self.env.step(self.dn)
            assert (np.abs(obs_simulate.prod_p - next_obs.prod_p) <= 1e-6).all(), f"for h={time_step}: {obs_simulate.prod_p} vs {next_obs.prod_p}"
            assert (np.abs(obs_simulate.storage_charge - next_obs.storage_charge) <= 1e-6).all()
            assert (np.abs(obs_simulate.storage_power - next_obs.storage_power) <= 1e-6).all()
            assert (np.abs(obs_simulate.storage_power_target - next_obs.storage_power_target) <= 1e-6).all()
                        
    def test_simulate_step_redisp_before_chain(self, redisp_amout=2., storage_amount=None):
        """one small redispatch and then lots of simulation of do nothing (chaining simulate on the forecasts)
        (default args)"""
        act = self.env.action_space()
        if redisp_amout is not None:
            act.redispatch = {"gen_2_1": redisp_amout}
        if storage_amount is not None:
            act.storage_p = [(1, 10.)]
        obs, *_ = self.env.step(act)
        obs_simulate = obs
        for time_step in [1, 2, 3, 4, 5, 6]:
            obs_simulate, *_ = obs_simulate.simulate(self.dn, time_step=1)
            next_obs, *_ = self.env.step(self.dn)
            assert (np.abs(obs_simulate.prod_p - next_obs.prod_p) <= 1e-6).all(), f"for h={time_step}: {obs_simulate.prod_p} vs {next_obs.prod_p}"
            assert (np.abs(obs_simulate.storage_charge - next_obs.storage_charge) <= 1e-6).all()
            assert (np.abs(obs_simulate.storage_power - next_obs.storage_power) <= 1e-6).all()
            assert (np.abs(obs_simulate.storage_power_target - next_obs.storage_power_target) <= 1e-6).all()
                        
    def test_simulate_step_redisp_before_large(self):
        """one large redispatch before simulation and then lots of simulation of do nothing"""
        self.test_simulate_step_redisp_before(redisp_amout=-10.)  # -10. so that redisp cannot be satisfied at first
        
    def test_simulate_step_redisp_before_from_init_large(self):
        """one large redispatch and then lots of simulation of do nothing (increasing the forecast horizon)"""
        self.test_simulate_step_redisp_before_from_init(redisp_amout=-10.)  # -10. so that redisp cannot be satisfied at first
        
    def test_simulate_step_redisp_before_chain_large(self):
        """one large redispatch and then lots of simulation of do nothing (increasing the forecast horizon)"""
        self.test_simulate_step_redisp_before_chain(redisp_amout=-10.)  # -10. so that redisp cannot be satisfied at first      

    def test_simulate_step_storage_before_large(self):
        """one action on storage unit before simulation and then lots of simulation of do nothing"""
        self.test_simulate_step_redisp_before(redisp_amout=None, storage_amount=10.)
        
    def test_simulate_step_redisp_storage_from_init_large(self):
        """one action on storage unit and then lots of simulation of do nothing (increasing the forecast horizon)"""
        self.test_simulate_step_redisp_before_from_init(redisp_amout=None, storage_amount=10.)
        
    def test_simulate_step_redisp_storage_chain_large(self):
        """one action on storage unit and then lots of simulation of do nothing (increasing the forecast horizon)"""
        self.test_simulate_step_redisp_before_chain(redisp_amout=None, storage_amount=10.)
        
                
# TODO when a redisp is done in the forecast
# TODO with curtailment