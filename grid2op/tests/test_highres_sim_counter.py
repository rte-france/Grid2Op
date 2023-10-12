# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import grid2op
from grid2op.Chronics import FromHandlers
from grid2op.Chronics.handlers import PerfectForecastHandler, CSVHandler, DoNothingHandler
from grid2op.Observation import BaseObservation
from grid2op.Runner import Runner
from grid2op.Agent import BaseAgent

import pdb

import warnings


class OneSimulateAgent(BaseAgent):
    def act(self, observation: BaseObservation, reward: float, done: bool = False):
        observation.simulate(self.action_space())
        if observation.prod_p.sum() <= 250:
            # just to make different number of "simulate" per episode
            # this never triggers for ep id 0 but do it 3 times for ep_id 1 (provided that max_ts is 5)
            observation.simulate(self.action_space())
        return super().act(observation, reward, done)
            
            
class HighreSimTester(unittest.TestCase):
    def _make_env(self):
        forecasts_horizons = [5, 10, 15, 20, 25, 30]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            env = grid2op.make("l2rpn_case14_sandbox",
                               test=True,
                               _add_to_name=type(self).__name__,
                               data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                   "gen_p_handler": CSVHandler("prod_p"),
                                   "load_p_handler": CSVHandler("load_p"),
                                   "gen_v_handler": DoNothingHandler("prod_v"),
                                   "load_q_handler": CSVHandler("load_q"),
                                   "h_forecast": forecasts_horizons,
                                   "gen_p_for_handler": PerfectForecastHandler("prod_p_forecasted"),
                                   "load_p_for_handler": PerfectForecastHandler("load_p_forecasted"),
                                   "load_q_for_handler": PerfectForecastHandler("load_q_forecasted"),
                                  })
        env.seed(0)
        env.set_id(0)
        return env
        
    def setUp(self) -> None:
        self.env = self._make_env()
        
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_simple_setting(self):
        """test when no copy, runner, chain to simulate etc., just simulate and reset"""
        obs0 = self.env.reset()
        assert self.env.nb_highres_called == 0, f"{self.env.nb_highres_called} vs 0"
        obs0.simulate(self.env.action_space())
        assert self.env.nb_highres_called == 1, f"{self.env.nb_highres_called} vs 1"
        obs0.simulate(self.env.action_space())
        assert self.env.nb_highres_called == 2, f"{self.env.nb_highres_called} vs 2"
        obs0.simulate(self.env.action_space(), 5)
        assert self.env.nb_highres_called == 3, f"{self.env.nb_highres_called} vs 3"
        obs1 = self.env.reset()
        assert self.env.nb_highres_called == 3, f"{self.env.nb_highres_called} vs 3"
        obs1.simulate(self.env.action_space())
        assert self.env.nb_highres_called == 4, f"{self.env.nb_highres_called} vs 4"
        obs0.simulate(self.env.action_space(), 3)
        assert self.env.nb_highres_called == 5, f"{self.env.nb_highres_called} vs 5"
        obs2, *_ = self.env.step(self.env.action_space())
        assert self.env.nb_highres_called == 5, f"{self.env.nb_highres_called} vs 5"
        obs2.simulate(self.env.action_space(), 3)
        assert self.env.nb_highres_called == 6, f"{self.env.nb_highres_called} vs 6"
        
    def test_forecast_env(self):
        """test the correct behaviour even when there is a forecast env"""
        obs0 = self.env.reset()
        assert self.env.nb_highres_called == 0, f"{self.env.nb_highres_called} vs 0"
        
        for_env = obs0.get_forecast_env()
        for_env.step(self.env.action_space())
        assert self.env.nb_highres_called == 1, f"{self.env.nb_highres_called} vs 1"
        for_env.step(self.env.action_space())
        assert self.env.nb_highres_called == 2, f"{self.env.nb_highres_called} vs 2"
        for_env.step(self.env.action_space())
        assert self.env.nb_highres_called == 3, f"{self.env.nb_highres_called} vs 3"
        for_env1 = obs0.get_forecast_env()
        assert self.env.nb_highres_called == 3, f"{self.env.nb_highres_called} vs 3"
        obs1, *_ = self.env.step(self.env.action_space())
        assert self.env.nb_highres_called == 3, f"{self.env.nb_highres_called} vs 3"
        for_env1.step(self.env.action_space())
        assert self.env.nb_highres_called == 4, f"{self.env.nb_highres_called} vs 4"
        for_env.step(self.env.action_space())
        assert self.env.nb_highres_called == 5, f"{self.env.nb_highres_called} vs 5"
        obs1.simulate(self.env.action_space(), 3)
        assert self.env.nb_highres_called == 6, f"{self.env.nb_highres_called} vs 6"
        
        for_env2 = for_env1.copy()
        assert self.env.nb_highres_called == 6, f"{self.env.nb_highres_called} vs 6"
        
        for_env2.step(self.env.action_space())
        assert self.env.nb_highres_called == 7, f"{self.env.nb_highres_called} vs 7"
        
    def test_env_copied(self):
        """test the nb_highres is kept when env is copied"""
        obs0 = self.env.reset()
        assert self.env.nb_highres_called == 0, f"{self.env.nb_highres_called} vs 0"
        
        env_cpy = self.env.copy()
        obs1 = env_cpy.reset()
        assert self.env.nb_highres_called == 0, f"{self.env.nb_highres_called} vs 0"
        assert env_cpy.nb_highres_called == 0, f"{env_cpy.nb_highres_called} vs 0"
        
        obs1.simulate(self.env.action_space(), 3)
        assert env_cpy.nb_highres_called == 1, f"{env_cpy.nb_highres_called} vs 1"
        assert self.env.nb_highres_called == 1, f"{self.env.nb_highres_called} vs 1"
        
        obs0.simulate(self.env.action_space())
        assert self.env.nb_highres_called == 2, f"{self.env.nb_highres_called} vs 2"
        assert env_cpy.nb_highres_called == 2, f"{env_cpy.nb_highres_called} vs 2"
        
        for_env = obs0.get_forecast_env()
        assert self.env.nb_highres_called == 2, f"{self.env.nb_highres_called} vs 2"
        assert env_cpy.nb_highres_called == 2, f"{env_cpy.nb_highres_called} vs 2"
        
        for_env.step(self.env.action_space())
        assert self.env.nb_highres_called == 3, f"{self.env.nb_highres_called} vs 3"
        assert env_cpy.nb_highres_called == 3, f"{env_cpy.nb_highres_called} vs 3"
        
        for_env1 = obs1.get_forecast_env()
        assert self.env.nb_highres_called == 3, f"{self.env.nb_highres_called} vs 3"
        assert env_cpy.nb_highres_called == 3, f"{env_cpy.nb_highres_called} vs 3"
        
        for_env1.step(self.env.action_space())
        assert self.env.nb_highres_called == 4, f"{self.env.nb_highres_called} vs 4"
        assert env_cpy.nb_highres_called == 4, f"{env_cpy.nb_highres_called} vs 4"
    
    def test_chain_simulate(self):
        """test it counts properly when the calls to obs.simulate are chained"""
        obs0 = self.env.reset()
        assert self.env.nb_highres_called == 0, f"{self.env.nb_highres_called} vs 0"
        obs1, *_ = obs0.simulate(self.env.action_space())
        assert self.env.nb_highres_called == 1, f"{self.env.nb_highres_called} vs 1"
        obs2, *_ = obs1.simulate(self.env.action_space())
        assert self.env.nb_highres_called == 2, f"{self.env.nb_highres_called} vs 2"
        obs3, *_ = obs2.simulate(self.env.action_space())
        assert self.env.nb_highres_called == 3, f"{self.env.nb_highres_called} vs 3"

    def test_simulator(self):
        """test it works when I use simulator"""
        obs0 = self.env.reset()
        sim0 = obs0.get_simulator()
        assert self.env.nb_highres_called == 0, f"{self.env.nb_highres_called} vs 0"
        
        simulator_stressed0 = sim0.predict(act=self.env.action_space(), do_copy=False)
        assert self.env.nb_highres_called == 1, f"{self.env.nb_highres_called} vs 1"
        
        simulator_stressed1 = simulator_stressed0.predict(act=self.env.action_space())
        assert self.env.nb_highres_called == 2, f"{self.env.nb_highres_called} vs 2"
        
        simulator_stressed2 = simulator_stressed1.predict(act=self.env.action_space())
        assert self.env.nb_highres_called == 3, f"{self.env.nb_highres_called} vs 3"
        
    def test_noshare_between_makes(self):
        """test that if call twice the 'make' function then the highres_sim_counter is not shared"""
        env1 = self._make_env()
        obs0 = self.env.reset()
        obs0.simulate(self.env.action_space())
        assert self.env.nb_highres_called == 1, f"{self.env.nb_highres_called} vs 1"
        assert env1.nb_highres_called == 0, f"{self.env.nb_highres_called} vs 0"
        
    def test_runner_seq(self):
        """test it behaves normally with the runner"""
        runner = Runner(**self.env.get_params_for_runner())
        
        # normal processing, with do nothing
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = runner.run(nb_episode=1, max_iter=5, add_nb_highres_sim=True)
        assert len(res) == 1
        assert len(res[0]) == 6
        assert res[0][5] == 0
        
        # normal processing with an agent that uses simulate
        runner = Runner(**self.env.get_params_for_runner(), agentClass=OneSimulateAgent)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = runner.run(nb_episode=1, max_iter=5, add_nb_highres_sim=True)
        assert len(res) == 1
        assert len(res[0]) == 6
        assert res[0][5] == 5
        
        # normal processing with an agent that uses forecasted_env
        class OneForEnvAgent(BaseAgent):
            def act(self, observation: BaseObservation, reward: float, done: bool = False):
                for_env = observation.get_forecast_env()
                for_env.step(self.action_space())
                return super().act(observation, reward, done)
        runner = Runner(**self.env.get_params_for_runner(), agentClass=OneForEnvAgent)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = runner.run(nb_episode=1, max_iter=5, add_nb_highres_sim=True)
        assert len(res) == 1
        assert len(res[0]) == 6
        assert res[0][5] == 5
        
        # 2 episodes sequential
        runner = Runner(**self.env.get_params_for_runner(), agentClass=OneSimulateAgent)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = runner.run(nb_episode=2, max_iter=5, add_nb_highres_sim=True)
        assert len(res) == 2
        assert len(res[0]) == 6
        assert res[0][5] == 5
        assert len(res[1]) == 6
        assert res[1][5] == 8
    
    def test_runner_par(self):
        # 2 episodes parrallel
        runner = Runner(**self.env.get_params_for_runner(), agentClass=OneSimulateAgent)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = runner.run(nb_episode=2, max_iter=5, add_nb_highres_sim=True, nb_process=2)
        assert len(res) == 2
        assert len(res[0]) == 6
        assert res[0][5] == 5
        assert len(res[1]) == 6
        assert res[1][5] == 8
        
        
if __name__ == '__main__':
    unittest.main()
