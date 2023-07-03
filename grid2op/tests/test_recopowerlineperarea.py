# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import os
import numpy as np
import grid2op
from grid2op.Agent import RecoPowerlinePerArea
import unittest

import pdb

"""snippet for the "debug" stuff

if hasattr(self, "_debug") and self._debug:
    import pdb
    pdb.set_trace()
"""


class TestRecoPowerlinePerArea(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_idf_2023", test=True)
        param = self.env.parameters
        param.NO_OVERFLOW_DISCONNECTION = True
        self.env.change_parameters(param)
        self.env.seed(0)
        self.env.set_id(0)
        
        self.agent = RecoPowerlinePerArea(self.env.action_space,
                                          self.env._game_rules.legal_action.substations_id_by_area)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
        
    def test_can_act_dn(self):
        obs = self.env.reset()
        act = self.agent.act(obs, None, None)
        assert not act.can_affect_something()
        
    def test_can_act_reco1(self):
        """test it can reconnect one line if one is connected"""
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space({"set_line_status": [(0, -1)]}))
        act = self.agent.act(obs, None, None)
        assert not act.can_affect_something()  # cooldown
        obs, *_ = self.env.step(self.env.action_space())
        act = self.agent.act(obs, None, None)
        assert not act.can_affect_something()  # cooldown
        obs, *_ = self.env.step(self.env.action_space())
        act = self.agent.act(obs, None, None)
        assert not act.can_affect_something()  # cooldown
        obs, *_ = self.env.step(self.env.action_space())
        act = self.agent.act(obs, None, None)
        assert act.can_affect_something()
        assert act.get_topological_impact()[0][0]
        
    def test_can_act_reco2(self):
        """test it can reconnect two lines if two are disconnected, not on the same area"""
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space({"set_line_status": [(0, -1), (3, -1)]}))
        act = self.agent.act(obs, None, None)
        assert not act.can_affect_something()  # cooldown
        obs, *_ = self.env.step(self.env.action_space())
        act = self.agent.act(obs, None, None)
        assert not act.can_affect_something()  # cooldown
        obs, *_ = self.env.step(self.env.action_space())
        act = self.agent.act(obs, None, None)
        assert not act.can_affect_something()  # cooldown
        obs, *_ = self.env.step(self.env.action_space())
        act = self.agent.act(obs, None, None)
        assert act.can_affect_something()
        assert act.get_topological_impact()[0][0]
        assert act.get_topological_impact()[0][3]
        
    def test_can_act_reco_only1(self):
        """test it does not attempt to reconnect two lines on the same area"""
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space({"set_line_status": [(0, -1)]}))
        obs, *_ = self.env.step(self.env.action_space({"set_line_status": [(2, -1)]}))
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        assert np.all(obs.time_before_cooldown_line == 0)
        act = self.agent.act(obs, None, None)
        assert act.get_topological_impact()[0][0]
        assert not act.get_topological_impact()[0][2]
        
    def test_do_not_attempt_reco_cooldown(self):
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space({"set_line_status": [(0, -1)]}))
        obs, *_ = self.env.step(self.env.action_space({"set_line_status": [(3, -1)]}))
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        act = self.agent.act(obs, None, None)  # line 3 still in cooldown
        assert act.get_topological_impact()[0][0]
        assert not act.get_topological_impact()[0][3]
        obs, *_ = self.env.step(act)
        act = self.agent.act(obs, None, None)
        assert act.get_topological_impact()[0][3]
        
if __name__ == "__main__":
    unittest.main()
