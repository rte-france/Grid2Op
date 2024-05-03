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
        self.obs = self._aux_reset_env(seed=0, ep_id=0, init_state={"set_bus": {"lines_or_id": [(5, 2)]}})
        
        # in the time series
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 2
        assert self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 2
        # in the action
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[5]] == 2
    
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
