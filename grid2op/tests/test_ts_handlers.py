# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import pdb
import time
import warnings

from grid2op.tests.helper_path_test import *

import grid2op
from grid2op.Chronics.time_series_from_handlers import FromHandlers
from grid2op.Chronics.handlers import CSVHandler, DoNothingHandler, CSVHandlerForecast
from grid2op.Runner import Runner
from grid2op.Exceptions import HandlerError

import warnings

# TODO check when there is also redispatching


def _load_next_chunk_in_memory_hack(self):
    self._nb_call += 1
    # i load the next chunk as dataframes
    array = self._get_next_chunk()  # array: load_p
    # i put these dataframes in the right order (columns)
    self._init_attrs(array)  # TODO
    # i don't forget to reset the reading index to 0
    self.current_index = 0
        
class TestCSVHandlerEnv(HelperTests):
    """test the env part of the storage functionality"""
    def _aux_reproducibility(self):
        for env in [self.env1, self.env2]:
            env.set_id(0)
            env.seed(0)
            env.reset()
        
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env1 = grid2op.make("l2rpn_case14_sandbox", test=True)  # regular env
            self.env2 = grid2op.make("l2rpn_case14_sandbox",
                                     data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                                          "gen_p_handler": CSVHandler("prod_p"),
                                                          "load_p_handler": CSVHandler("load_p"),
                                                          "gen_v_handler": CSVHandler("prod_v"),
                                                          "load_q_handler": CSVHandler("load_q"),
                                                          },
                                     _add_to_name="_TestCSVHandlerEnv",
                                     test=True)  # regular env
        self._aux_reproducibility()
        return super().setUp()

    def tearDown(self) -> None:
        self.env1.close()
        self.env2.close()
        return super().tearDown()

    def _aux_compare_one(self, it_nm, obs1, obs2, descr=""):
        for attr_nm in ["load_p", "load_q", "gen_p", "gen_v", "rho"]:
            assert np.all(getattr(obs1, attr_nm) == getattr(obs2, attr_nm)), f"error for {attr_nm}{descr} at iteration {it_nm}"

                
    def _run_then_compare(self, nb_iter=10, env1=None, env2=None):
        if env1 is None:
            env1 = self.env1
        if env2 is None:
            env2 = self.env2
            
        for k in range(nb_iter):
            obs1, reward1, done1, info1 = env1.step(env1.action_space())
            obs2, reward2, done2, info2 = env2.step(env2.action_space())
            assert done2 == done1, f"error at iteration {k}"
            assert reward1 == reward2, f"error at iteration {k}"
            self._aux_compare_one(k, obs1, obs2)
                
    def test_step_equal(self):
        self._run_then_compare()
        self.env1.reset()
        self.env2.reset()
        self._run_then_compare()
        self.env1.reset()
        self.env2.reset()
        self._run_then_compare()
    
    def test_max_iter(self):
        self.env1.set_max_iter(5)
        self.env2.set_max_iter(5)
        
        self.env1.reset()
        self.env2.reset()
        self._run_then_compare(nb_iter=4)
        
        obs1, reward1, done1, info1 = self.env1.step(self.env1.action_space())
        obs2, reward2, done2, info2 = self.env2.step(self.env2.action_space())
        assert done1
        assert done2
        
    def test_chunk(self):
        self.env1.set_chunk_size(1)
        self.env2.set_chunk_size(1)
        
        # hugly copy paste from above otherwise the hack do not work...
        # because of the reset
        self.env1.set_max_iter(5)
        self.env2.set_max_iter(5)
        
        self.env1.reset()
        self.env2.reset()
        
        ###### hack to count the number this is called
        self.env2.chronics_handler.data.gen_p_handler._nb_call = 0
        self.env2.chronics_handler.data.gen_p_handler._load_next_chunk_in_memory = lambda : _load_next_chunk_in_memory_hack(self.env2.chronics_handler.data.gen_p_handler)
        ######
        
        self._run_then_compare(nb_iter=4)
        
        obs1, reward1, done1, info1 = self.env1.step(self.env1.action_space())
        obs2, reward2, done2, info2 = self.env2.step(self.env2.action_space())
        assert done1
        assert done2
        
        # now check the "load_next_chunk has been called the right number of time"
        assert self.env2.chronics_handler.data.gen_p_handler._nb_call == 5
        
    def test_copy(self):
        env2 = self.env2.copy()
        self._run_then_compare(env2=env2)
        self.env1.reset()
        env2.reset()
        self._run_then_compare(env2=env2)
        self.env1.reset()
        env2.reset()
        self._run_then_compare(env2=env2)   

    def test_runner(self):
        runner1 = Runner(**self.env1.get_params_for_runner())
        runner2 = Runner(**self.env2.get_params_for_runner())
        
        res1 = runner1.run(nb_episode=2, max_iter=5, env_seeds=[0, 1], episode_id=[0, 1])
        res2 = runner2.run(nb_episode=2, max_iter=5, env_seeds=[0, 1], episode_id=[0, 1])
        assert res1 == res2
    
    def test_if_file_absent(self):
        # do it only once
        if type(self) != TestCSVHandlerEnv:
            self.skipTest("This test should be done only in the TestCSVHandlerEnv class")
        with self.assertRaises(HandlerError):
            grid2op.make(os.path.join(PATH_DATA_TEST, "5bus_example_some_missing"),
                         data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                              "gen_p_handler": CSVHandler("prod_p"),
                                              "load_p_handler": CSVHandler("load_p"),
                                              "gen_v_handler": DoNothingHandler(),
                                              "load_q_handler": CSVHandler("load_q"), # crash because this file does not exist
                                              },
                         _add_to_name="_TestDNHandlerEnv")  # regular env
    
# TODO:
# test when "names_chronics_to_backend"
# test runner X
# test env copy X
# test when max_iter `env.set_max_iter` X
# test when "set_chunk" X
# test with forecasts
# test with maintenance


class TestDNHandlerEnv(TestCSVHandlerEnv):
    """test the env part of the storage functionality"""
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env1 = grid2op.make(os.path.join(PATH_DATA_TEST, "5bus_example_some_missing"))  # regular env
            self.env2 = grid2op.make(os.path.join(PATH_DATA_TEST, "5bus_example_some_missing"),
                                     data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                                          "gen_p_handler": CSVHandler("prod_p"),
                                                          "load_p_handler": CSVHandler("load_p"),
                                                          "gen_v_handler": DoNothingHandler(),
                                                          "load_q_handler": DoNothingHandler(),
                                                          },
                                     _add_to_name="_TestDNHandlerEnv")
        self._aux_reproducibility()
            
            
class TestForecastHandler14(TestCSVHandlerEnv):
    """test the env part of the storage functionality"""
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env1 = grid2op.make("l2rpn_case14_sandbox", test=True)  # regular env
            self.env2 = grid2op.make("l2rpn_case14_sandbox",
                                     data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                                          "gen_p_handler": CSVHandler("prod_p"),
                                                          "load_p_handler": CSVHandler("load_p"),
                                                          "gen_v_handler": CSVHandler("prod_v"),
                                                          "load_q_handler": CSVHandler("load_q"),
                                                          "gen_p_for_handler": CSVHandlerForecast("prod_p_forecasted"),
                                                          "load_p_for_handler": CSVHandlerForecast("load_p_forecasted"),
                                                          "gen_v_for_handler": CSVHandlerForecast("prod_v_forecasted"),
                                                          "load_q_for_handler": CSVHandlerForecast("load_q_forecasted"),
                                                          },
                                     _add_to_name="TestForecastHandlerEnv",
                                     test=True)
        self._aux_reproducibility()
        assert np.all(self.env1.chronics_handler.real_data.data.load_p_forecast == 
                      self.env2.chronics_handler.real_data.data.load_p_for_handler.array)
        
            
    def _aux_compare_one(self, it_nm, obs1, obs2):
        super()._aux_compare_one(it_nm, obs1, obs2)
        sim_obs1, *_ = obs1.simulate(self.env1.action_space())
        sim_obs2, *_ = obs2.simulate(self.env1.action_space())
        super()._aux_compare_one(it_nm, sim_obs1, sim_obs2, " forecast")
            
            
if __name__ == "__main__":
    unittest.main()
