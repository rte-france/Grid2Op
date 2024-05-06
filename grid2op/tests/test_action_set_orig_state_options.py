# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import warnings
import unittest
    
import grid2op
from grid2op.Runner import Runner
from grid2op.tests.helper_path_test import *


# TODO test with redispatching, curtailment or storage
# TODO in the runner too


class TestSetActOptionDefault(unittest.TestCase):        
    def _env_path(self):
        return os.path.join(
            PATH_DATA_TEST, "5bus_example_act_topo_set_init"
        )
    
    def setUp(self) -> None:
        self.env_nm = self._env_path()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_nm,
                                    test=True
                                    )
        
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def _aux_reset_env(self, seed, ep_id, init_state):
        obs = self.env.reset(seed=seed, options={"time serie id": ep_id,
                                                 "init state": init_state})
        return obs
    
    def _aux_make_step(self, act=None):
        if act is None:
            act = self.env.action_space()
        return self.env.step(act)
        
    def _aux_get_init_act(self):
        return self.env.chronics_handler.get_init_action()
    
    def test_combine_ts_set_bus_opt_setbus_nopb(self):
        # ts id 0 => set_bus (in the time series)
        self.obs = self._aux_reset_env(seed=0, ep_id=0, init_state={"set_bus": {"lines_or_id": [(0, 2)]}})
        # in the time series
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 2
        assert self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 2
        # in the action
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[0]] == 2
    
    def test_combine_ts_set_bus_opt_setbus_collision(self):
        # ts id 0 => set_bus (in the time series)
        self.obs = self._aux_reset_env(seed=0, ep_id=0, init_state={"set_bus": {"lines_or_id": [(1, 1)],
                                                                                "loads_id": [(0, 1)]}})
        
        # in the option (totally erase the time series)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 1
    
    def test_combine_ts_set_bus_opt_setstat_nopb(self):
        # ts id 0 => set_bus (in the time series)
        self.obs = self._aux_reset_env(seed=0, ep_id=0,
                                       init_state={"set_line_status": [(5, -1)]})
        
        # in the time series
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 2
        assert self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 2
        # in the action
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[5]] == -1
    
    def test_combine_ts_set_bus_opt_setstat_collision(self):
        # ts id 0 => set_bus (in the time series)
        self.obs = self._aux_reset_env(seed=0, ep_id=0,
                                       init_state={"set_bus": {"loads_id": [(0, 1)]},
                                                   "set_line_status": [(1, -1)]})
        # in the act
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == -1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == -1
        assert not self.obs.line_status[1]
        # in the time series
        assert self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 1
        
    def test_combine_ts_set_status_opt_setbus_nopb(self):
        # ts id 1 => set_status
        self.obs = self._aux_reset_env(seed=0, ep_id=1, init_state={"set_bus": {"lines_or_id": [(5, 2)]}})
        
        # in the time series
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == -1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == -1
        assert not self.obs.line_status[1]
        # in the action
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[5]] == 2
        
    def test_combine_ts_set_status_opt_setbus_collision(self):
        # ts id 1 => set_status
        self.obs = self._aux_reset_env(seed=0, ep_id=1, init_state={"set_bus": {"lines_or_id": [(1, 1)]}})
        # in the time series (erased by the action, or side)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 1
        assert self.obs.line_status[1]
        
        # ts id 1 => set_status
        self.obs = self._aux_reset_env(seed=0, ep_id=1, init_state={"set_bus": {"lines_ex_id": [(1, 2)]}})
        # in the time series (erased by the action, ex side)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 2
        assert self.obs.line_status[1]

    def test_combine_ts_set_status_opt_setstat_nopb(self):
        # ts id 1 => set_status
        self.obs = self._aux_reset_env(seed=0, ep_id=1, init_state={"set_line_status": [(5, -1)]})
        
        # in the time series
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == -1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == -1
        assert not self.obs.line_status[1]
        # in the action
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[5]] == -1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[5]] == -1
        assert not self.obs.line_status[5]    
        
    def test_combine_ts_set_status_opt_setstat_collision(self):
        # ts id 1 => set_status
        self.obs = self._aux_reset_env(seed=0, ep_id=1, init_state={"set_line_status": [(1, 1)]})
        
        # in the time series (bus overriden by the action)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 1
        assert self.obs.line_status[1]
        return self.env.chronics_handler.get_init_action()  
    
    def test_ignore_ts_set_bus_opt_setbus_nopb(self):
        # ts id 0 => set_bus (in the time series)
        self.obs = self._aux_reset_env(seed=0, ep_id=0, init_state={"set_bus": {"lines_or_id": [(5, 2)]}, "method": "ignore"})
        
        # in the time series
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 1
        # in the action
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[5]] == 2
    
    def test_ignore_ts_set_bus_opt_setbus_collision(self):
        # TODO not tested for method = ignore (because action here totally erased action in ts)
        # ts id 0 => set_bus (in the time series)
        self.obs = self._aux_reset_env(seed=0, ep_id=0, init_state={"set_bus": {"lines_or_id": [(1, 1)],
                                                                                "loads_id": [(0, 1)]},
                                                                    "method": "ignore"})
        
        # in the option (totally erase the time series)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 1
    
    def test_ignore_ts_set_bus_opt_setstat_nopb(self):
        # ts id 0 => set_bus (in the time series)
        self.obs = self._aux_reset_env(seed=0, ep_id=0,
                                       init_state={"set_line_status": [(5, -1)],
                                                   "method": "ignore"})
        
        # in the time series
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 1
        # in the action
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[5]] == -1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[5]] == -1
        assert not self.obs.line_status[5]
    
    def test_ignore_ts_set_bus_opt_setstat_collision(self):
        # ts id 0 => set_bus (in the time series)
        self.obs = self._aux_reset_env(seed=0, ep_id=0,
                                       init_state={"set_bus": {"loads_id": [(0, 1)]},
                                                   "set_line_status": [(1, -1)],
                                                   "method": "ignore"})
        # in the act
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == -1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == -1
        assert not self.obs.line_status[1]
        # in the time series (ignored)
        assert self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 1
        
    def test_ignore_ts_set_status_opt_setbus_nopb(self):
        # ts id 1 => set_status
        self.obs = self._aux_reset_env(seed=0, ep_id=1, init_state={"set_bus": {"lines_or_id": [(5, 2)]},
                                                                    "method": "ignore"})
        
        # in the time series (ignored)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 1
        assert self.obs.line_status[1]
        # in the action
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[5]] == 2
        
    def test_ignore_ts_set_status_opt_setbus_collision(self):
        # TODO not tested for method = ignore (because action here totally erased action in ts)
        # ts id 1 => set_status
        self.obs = self._aux_reset_env(seed=0, ep_id=1, init_state={"set_bus": {"lines_or_id": [(1, 1)]},
                                                                    "method": "ignore"})
        # in the time series (erased by the action, or side)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 1
        assert self.obs.line_status[1]
        
        # ts id 1 => set_status
        self.obs = self._aux_reset_env(seed=0, ep_id=1, init_state={"set_bus": {"lines_ex_id": [(1, 2)]},
                                                                    "method": "ignore"})
        # in the time series (erased by the action, ex side)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 2
        assert self.obs.line_status[1]

    def test_ignore_ts_set_status_opt_setstat_nopb(self):
        # ts id 1 => set_status
        self.obs = self._aux_reset_env(seed=0, ep_id=1, init_state={"set_line_status": [(5, -1)],
                                                                    "method": "ignore"})
        
        # in the time series (ignored)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 1
        assert self.obs.line_status[1]
        # in the action
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[5]] == -1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[5]] == -1
        assert not self.obs.line_status[5]    
        
    def test_ignore_ts_set_status_opt_setstat_collision(self):
        # TODO not tested for method = ignore (because action here totally erased action in ts)
        
        # ts id 1 => set_status
        self.obs = self._aux_reset_env(seed=0, ep_id=1, init_state={"set_line_status": [(1, 1)],
                                                                    "method": "ignore"})
        
        # in the time series (bus overriden by the action)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 1
        assert self.obs.line_status[1]
        
    def test_byact(self):
        # ts id 1 => set_status
        act = self.env.action_space({"set_line_status": [(1, 1)]})
        self.obs = self._aux_reset_env(seed=0, ep_id=1, init_state=act)
        
        # in the time series (bus overriden by the action)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 1
        assert self.obs.line_status[1]
        return self.env.chronics_handler.get_init_action()


class TestSetInitRunner(unittest.TestCase):
    def _env_path(self):
        return os.path.join(
            PATH_DATA_TEST, "5bus_example_act_topo_set_init"
        )
    
    def setUp(self) -> None:
        self.env_nm = self._env_path()
        self.max_iter = 5
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_nm,
                                    test=True
                                    )
        self.runner = Runner(**self.env.get_params_for_runner())
        
    def tearDown(self) -> None:
        self.env.close()
        self.runner._clean_up()
        return super().tearDown()
    
    def test_run_one_episode(self):
        res = self.runner.run_one_episode(init_state={"set_line_status": [(1, 1)], "method": "ignore"},
                                          episode_id=1,
                                          max_iter=self.max_iter,
                                          detailed_output=True
                                         )
        ep_data = res[-1]
        init_obs = ep_data.observations[0]
        assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == 1
        assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[1]] == 1
        assert init_obs.line_status[1]
        
    def test_run_onesingle_ep_onesingle_act(self):
        # one action
        res = self.runner.run(nb_episode=1,
                              init_states={"set_line_status": [(1, 1)], "method": "ignore"},
                              episode_id=[1],
                              max_iter=self.max_iter,
                              add_detailed_output=True
                              )
        ep_data = res[0][-1]
        init_obs = ep_data.observations[0]
        assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == 1
        assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[1]] == 1
        assert init_obs.line_status[1]
        
        # one list (of one element here)
        res = self.runner.run(nb_episode=1,
                              init_states=[{"set_line_status": [(1, 1)], "method": "ignore"}],
                              episode_id=[1],
                              max_iter=self.max_iter,
                              add_detailed_output=True
                              )
        ep_data = res[0][-1]
        init_obs = ep_data.observations[0]
        assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == 1
        assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[1]] == 1
        assert init_obs.line_status[1]
        
        # one tuple (of one element here)
        res = self.runner.run(nb_episode=1,
                              init_states=({"set_line_status": [(1, 1)], "method": "ignore"}, ),
                              episode_id=[1],
                              max_iter=self.max_iter,
                              add_detailed_output=True
                              )
        ep_data = res[0][-1]
        init_obs = ep_data.observations[0]
        assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == 1
        assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[1]] == 1
        assert init_obs.line_status[1]
        
    def test_run_two_eps_seq_onesingle_act(self, nb_process=1):
        # one action
        res = self.runner.run(nb_episode=2,
                              init_states={"set_line_status": [(1, 1)], "method": "ignore"},
                              episode_id=[1, 1],
                              max_iter=self.max_iter,
                              add_detailed_output=True,
                              nb_process=nb_process
                              )
        for el in res:
            ep_data = el[-1]
            init_obs = ep_data.observations[0]
            assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == 1
            assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[1]] == 1
            assert init_obs.line_status[1]
        
        # one list
        res = self.runner.run(nb_episode=2,
                              init_states=[{"set_line_status": [(1, 1)], "method": "ignore"},
                                           {"set_line_status": [(1, 1)], "method": "ignore"}],
                              episode_id=[1, 1],
                              max_iter=self.max_iter,
                              add_detailed_output=True,
                              nb_process=nb_process
                              )
        for el in res:
            ep_data = el[-1]
            init_obs = ep_data.observations[0]
            assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == 1
            assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[1]] == 1
            assert init_obs.line_status[1]
        
        # one tuple
        res = self.runner.run(nb_episode=2,
                              init_states=({"set_line_status": [(1, 1)], "method": "ignore"},
                                           {"set_line_status": [(1, 1)], "method": "ignore"}),
                              episode_id=[1, 1],
                              max_iter=self.max_iter,
                              add_detailed_output=True,
                              nb_process=nb_process
                              )
        for el in res:
            ep_data = el[-1]
            init_obs = ep_data.observations[0]
            assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == 1
            assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[1]] == 1
            assert init_obs.line_status[1]
        
    def test_run_two_eps_seq_two_acts(self, nb_process=1):  
        # given as list
        res = self.runner.run(nb_episode=2,
                              init_states=[{"set_bus": {"loads_id": [(0, 1)]}, "set_line_status": [(1, -1)], "method": "ignore"},
                                           {"set_line_status": [(1, 1)], "method": "ignore"}],
                              episode_id=[0, 1],
                              max_iter=self.max_iter,
                              add_detailed_output=True,
                              nb_process=nb_process
                              )
        
        # check for ep 0
        ep_data = res[0][-1]
        init_obs = ep_data.observations[0]
        assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == -1
        assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[1]] == -1
        assert not init_obs.line_status[1]
        assert init_obs.topo_vect[init_obs.load_pos_topo_vect[0]] == 1
        # check for ep 1
        ep_data = res[1][-1]
        init_obs = ep_data.observations[0]
        assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == 1
        assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[1]] == 1
        assert init_obs.line_status[1]
        
        # one tuple
        res = self.runner.run(nb_episode=2,
                              init_states=({"set_bus": {"loads_id": [(0, 1)]}, "set_line_status": [(1, -1)], "method": "ignore"},
                                           {"set_line_status": [(1, 1)], "method": "ignore"}),
                              episode_id=[0, 1],
                              max_iter=self.max_iter,
                              add_detailed_output=True,
                              nb_process=nb_process
                              )
        # check for ep 0
        ep_data = res[0][-1]
        init_obs = ep_data.observations[0]
        assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == -1
        assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[1]] == -1
        assert not init_obs.line_status[1]
        assert init_obs.topo_vect[init_obs.load_pos_topo_vect[0]] == 1
        # check for ep 1
        ep_data = res[1][-1]
        init_obs = ep_data.observations[0]
        assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == 1
        assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[1]] == 1
        assert init_obs.line_status[1]
        
    def test_run_two_eps_par_onesingle_act(self):
        self.test_run_two_eps_seq_onesingle_act(nb_process=2)
        
    def test_run_two_eps_par_two_acts(self):
        self.test_run_two_eps_seq_two_acts(nb_process=2)
    
    def test_fail_when_needed(self):
        # wrong type
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                  init_states=1,
                                  episode_id=[0, 1],
                                  max_iter=self.max_iter,
                                  add_detailed_output=True,
                                  )
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                init_states=[1, {"set_line_status": [(1, 1)], "method": "ignore"}],
                                episode_id=[0, 1],
                                max_iter=self.max_iter,
                                add_detailed_output=True,
                                )
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                init_states=[{"set_line_status": [(1, 1)], "method": "ignore"}, 1],
                                episode_id=[0, 1],
                                max_iter=self.max_iter,
                                add_detailed_output=True,
                                )
            
        # wrong size (too big)
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                init_states=[{"set_line_status": [(1, 1)], "method": "ignore"},
                                             {"set_line_status": [(1, 1)], "method": "ignore"},
                                             {"set_line_status": [(1, 1)], "method": "ignore"}],
                                episode_id=[0, 1],
                                max_iter=self.max_iter,
                                add_detailed_output=True,
                                )
        # wrong size (too small)
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                init_states=[{"set_line_status": [(1, 1)], "method": "ignore"}],
                                episode_id=[0, 1],
                                max_iter=self.max_iter,
                                add_detailed_output=True,
                                )
        

class TestSetActOptionDefaultComplexAction(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage", test=True, _add_to_name=type(self).__name__)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def test_storage(self):
        obs = self.env.reset(seed=0,
                             options={"time serie id": 0,
                                      "init state": {"set_storage": [(0, 5.)]}})
        assert abs(obs.storage_power[0] - 5.) <= 1e-6
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert abs(obs.storage_power[0] - 0.) <= 1e-6
    
    def test_curtail(self):
        obs = self.env.reset(seed=0,
                             options={"time serie id": 0,
                                      "init state": {"curtail": [(3, 0.1)]}})
        assert abs(obs.curtailment_limit[3] - 0.1) <= 1e-6
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert abs(obs.curtailment_limit[3] - 0.1) <= 1e-6
        
    def test_redispatching(self):
        obs = self.env.reset(seed=0,
                             options={"time serie id": 0,
                                      "init state": {"redispatch": [(0, -1)]}})
        assert abs(obs.target_dispatch[0] - -1.) <= 1e-6
        assert abs(obs.actual_dispatch[0] - -1.) <= 1e-6
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert abs(obs.target_dispatch[0] - -1.) <= 1e-6
        assert abs(obs.actual_dispatch[0] - -1.) <= 1e-6
        
        
if __name__ == "__main__":
    unittest.main()

