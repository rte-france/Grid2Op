# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/rte-france/Grid2Op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.Exceptions import AmbiguousAction

import unittest
import warnings
import pdb


class RemoveLineStatusFromTopoTester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True)
            
        param = self.env.parameters
        param.NB_TIMESTEP_COOLDOWN_SUB = 3
        param.NB_TIMESTEP_COOLDOWN_LINE = 3
        param.NO_OVERFLOW_DISCONNECTION = True
        self.env.change_parameters(param)
        _ = self.env.reset()
            
    def test_limit_reco(self):
        """test that it limit the action when it reconnects"""
        act = self.env.action_space({"set_bus": {"substations_id": [(1, [1, 2, 2, -1, 1, 1])]}})
        obs, reward, done, info = self.env.step(act)

        # limit reco when  set
        act_sub4_clean = self.env.action_space({"set_bus": {"substations_id": [(4, [2, 2, 2, 1, 1])]}})
        assert act_sub4_clean._set_topo_vect[20] == 2
        act_sub4_clean.remove_line_status_from_topo(obs)       
        assert act_sub4_clean._set_topo_vect[20] == 0
        
        # limit reco when change
        act_sub4_clean = self.env.action_space({"change_bus": {"substations_id": [(4, [True, True, True, False, False])]}})
        assert act_sub4_clean._change_bus_vect[20]
        act_sub4_clean.remove_line_status_from_topo(obs)
        assert not act_sub4_clean._change_bus_vect[20]
    
    def test_limit_disco(self):
        """test that it limit the action when it disconnects"""
        dn = self.env.action_space()
        act = self.env.action_space({"set_bus": {"substations_id": [(1, [1, 2, 2, -1, 1, 1])]}})
        obs, reward, done, info = self.env.step(act)
        assert obs.time_before_cooldown_line[4] == 3
        obs, reward, done, info = self.env.step(dn)
        obs, reward, done, info = self.env.step(dn)
        obs, reward, done, info = self.env.step(dn)
        assert obs.time_before_cooldown_line[4] == 0
        
        # reconnect it
        act_reco = self.env.action_space({"set_bus": {"substations_id": [(4, [2, 2, 2, 1, 1])]}})
        obs, reward, done, info = self.env.step(act_reco)
        assert obs.time_before_cooldown_line[4] == 3
        
        # limit disco when  set
        act_deco = self.env.action_space({"set_bus": {"substations_id": [(1, [1, 2, 2, -1, 1, 1])]}})
        assert act_deco._set_topo_vect[6] == -1
        act_deco.remove_line_status_from_topo(obs)       
        assert act_deco._set_topo_vect[6] == 0
        
    def test_nothing_when_cooldown(self):
        """test it does nothing when there is no cooldown"""
        dn = self.env.action_space()
        act = self.env.action_space({"set_bus": {"substations_id": [(1, [1, 2, 2, -1, 1, 1])]}})
        obs, reward, done, info = self.env.step(act)
        assert obs.time_before_cooldown_line[4] == 3
        obs, reward, done, info = self.env.step(dn)
        obs, reward, done, info = self.env.step(dn)
        obs, reward, done, info = self.env.step(dn)
        assert obs.time_before_cooldown_line[4] == 0
        
        # action should not be modified because there is a cooldown
        act_sub4_clean = self.env.action_space({"set_bus": {"substations_id": [(4, [2, 2, 2, 1, 1])]}})
        assert act_sub4_clean._set_topo_vect[20] == 2
        act_sub4_clean.remove_line_status_from_topo(obs)       
        assert act_sub4_clean._set_topo_vect[20] == 2
    
    def test_something_when_nocooldown_butcheck_cooldown(self):
        """test that something is done when no cooldown but the check_cooldown is set"""
        dn = self.env.action_space()
        act = self.env.action_space({"set_bus": {"substations_id": [(1, [1, 2, 2, -1, 1, 1])]}})
        obs, reward, done, info = self.env.step(act)
        assert obs.time_before_cooldown_line[4] == 3
        obs, reward, done, info = self.env.step(dn)
        obs, reward, done, info = self.env.step(dn)
        obs, reward, done, info = self.env.step(dn)
        assert obs.time_before_cooldown_line[4] == 0
        
        # action should not be modified because there is a cooldown
        act_sub4_clean = self.env.action_space({"set_bus": {"substations_id": [(4, [2, 2, 2, 1, 1])]}})
        assert act_sub4_clean._set_topo_vect[20] == 2
        act_sub4_clean.remove_line_status_from_topo(obs, check_cooldown=False)       
        assert act_sub4_clean._set_topo_vect[20] == 0
            
    def test_limit_withoutobs(self):
        """test that it limit the action correctly when no obs is provided"""
        disco = self.env.action_space({"set_line_status": [(4, -1)]})
        reco = self.env.action_space({"set_line_status": [(4, +1)]})

        # limit reco when  set
        act_sub4_clean = self.env.action_space({"set_bus": {"substations_id": [(4, [2, 2, 2, 1, 1])]}})
        act_sub4_clean += disco
        assert act_sub4_clean._set_topo_vect[20] == 2
        assert act_sub4_clean._set_line_status[4] == -1
        with self.assertRaises(AmbiguousAction):
            act_sub4_clean._check_for_ambiguity()
        act_sub4_clean.remove_line_status_from_topo(check_cooldown=False)       
        assert act_sub4_clean._set_topo_vect[20] == 0
        assert act_sub4_clean._set_line_status[4] == -1
        act_sub4_clean._check_for_ambiguity()  # does not raise
        
        # limit reco when change
        act_sub4_clean = self.env.action_space({"change_bus": {"substations_id": [(4, [True, True, True, False, False])]}})
        act_sub4_clean += disco
        assert act_sub4_clean._change_bus_vect[20]
        assert act_sub4_clean._set_line_status[4] == -1
        with self.assertRaises(AmbiguousAction):
            act_sub4_clean._check_for_ambiguity()
        act_sub4_clean.remove_line_status_from_topo(check_cooldown=False)
        assert not act_sub4_clean._change_bus_vect[20]
        assert act_sub4_clean._set_line_status[4] == -1
        act_sub4_clean._check_for_ambiguity()  # does not raise        
        
        # limit disco when  set
        act_sub4_clean = self.env.action_space({"set_bus": {"substations_id": [(4, [2, -1, 2, 1, 1])]}})
        act_sub4_clean += reco
        assert act_sub4_clean._set_topo_vect[20] == -1
        assert act_sub4_clean._set_line_status[4] == 1
        with self.assertRaises(AmbiguousAction):
            act_sub4_clean._check_for_ambiguity()
        act_sub4_clean.remove_line_status_from_topo(check_cooldown=False)       
        assert act_sub4_clean._set_topo_vect[20] == 0
        assert act_sub4_clean._set_line_status[4] == 1
        act_sub4_clean._check_for_ambiguity()  # does not raise
        
    
    
if __name__ == "__main__":
    unittest.main()
