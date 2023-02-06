# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings
import numpy as np
import pdb
import os

from grid2op.tests.helper_path_test import *
from grid2op.Exceptions import NoForecastAvailable
from grid2op.Chronics import MultifolderWithCache
    
import grid2op
import numpy as np


class MultiStepsForcaTester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make(os.path.join(PATH_DATA_TEST, "5bus_example_forecasts"), test=True)
        self.env.seed(0)
        self.env.set_id(0)
    
    def aux_test_for_consistent(self, obs):
        tmp_o_1, *_ = obs.simulate(self.env.action_space(), time_step=1)
        assert (obs.load_p + 1. == tmp_o_1.load_p).all()  # that's how I generated the forecast for this "env"
        tmp_o_2, *_ = obs.simulate(self.env.action_space(), time_step=2)
        assert (obs.load_p + 2. == tmp_o_2.load_p).all()  # that's how I generated the forecast for this "env"
        tmp_o_3, *_ = obs.simulate(self.env.action_space(), time_step=3)
        assert (obs.load_p + 3. == tmp_o_3.load_p).all()
        tmp_o_12, *_ = obs.simulate(self.env.action_space(), time_step=12)
        assert (obs.load_p + 12. == tmp_o_12.load_p).all()
        
    def test_can_do(self):
        obs = self.env.reset()
        self.aux_test_for_consistent(obs)
        
        # should raise because there is no "13 steps ahead forecasts"
        with self.assertRaises(NoForecastAvailable):
            obs.simulate(self.env.action_space(), time_step=13)
        
        # check it's still consistent
        obs, *_ = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        # check it's still consistent
        obs, *_ = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
            
    def test_chunk_size(self):
        self.env.set_chunk_size(1)
        obs = self.env.reset()
        self.aux_test_for_consistent(obs)
        
        # check it's still consistent
        obs, *_ = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
            
        # check it's still consistent
        obs, *_ = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        # check it's still consistent
        obs, *_ = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
    
    def test_max_iter(self):
        max_iter = 4
        self.env.chronics_handler.set_max_iter(max_iter)
        
        obs = self.env.reset()
        self.aux_test_for_consistent(obs)
        
        # check it's still consistent
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        assert done
            

    def test_cache(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            env = grid2op.make(os.path.join(PATH_DATA_TEST, "5bus_example_forecasts"),
                               test=True,
                               chronics_class=MultifolderWithCache)
            
        env.seed(0)
        env.set_id(0)
        env.chronics_handler.reset()
        
        obs = self.env.reset()
        self.aux_test_for_consistent(obs)
        
        # check it's still consistent
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.aux_test_for_consistent(obs)


if __name__ == "__main__":
    unittest.main()