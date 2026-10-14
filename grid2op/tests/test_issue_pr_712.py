# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import json
import tempfile
import warnings
import grid2op
from pathlib import Path
import unittest

from grid2op.Agent import RandomAgent



class _CustomRandom(RandomAgent):
    def __init__(self, action_space):
        RandomAgent.__init__(self, action_space)
        self.i = 1

    def my_act(self, transformed_observation, reward, done=False):
        if (self.i % 10) != 0:
            res = 0
        else:
            res = self.action_space.sample()
        self.i += 1
        return res
            
class TestIssue712(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox",
                                    test=True,
                                    _add_to_name=type(self).__name__,
                                    n_busbar=3)
        self.init_obs = self.env.reset(seed=0, options={"time serie id":0})
        self.max_iter = 5
        return super().setUp()
    
    def test_can_make_agent(self):
        myagent = _CustomRandom(self.env.action_space)
        myagent.seed(0)
        
    def test_agent_can_run(self):
        myagent = _CustomRandom(self.env.action_space)
        myagent.seed(0)
        obs = self.env.reset()
        reward = self.env.reward_range[0]
        done = False
        nb_step = 0
        while not done:
            act = myagent.act(obs, reward, done)
            obs, reward, done, info = self.env.step(act)
            nb_step += 1
            if nb_step >= self.max_iter:
                break
    
    def test_agent_correct_act_space(self):
        myagent = _CustomRandom(self.env.action_space)
        assert "_change_bus_vect" not in myagent.action_space._template_act.attr_list_vect
        assert len(myagent.action_space._template_act.attr_list_vect) == 8  # change_bus should be deactivated
        
    
if __name__ == "__main__":
    unittest.main()
