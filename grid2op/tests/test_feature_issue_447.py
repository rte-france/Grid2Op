# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.Runner import Runner
from grid2op.Chronics import MultifolderWithCache  # highly recommended for training
import warnings
import unittest
from grid2op.Exceptions import ChronicsError
from grid2op.Runner import Runner
import pdb


class TestPreventWrongBehaviour(unittest.TestCase):
    def setUp(self) -> None:
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(env_name,
                                    chronics_class=MultifolderWithCache,
                                    test=True,
                                    _add_to_name=type(self).__name__)
    def tearDown(self) -> None:
        self.env.close()
        
    def test_can_make(self):
        pass
        
    def test_cannot_step(self):
        with self.assertRaises(ChronicsError):
            self.env.step(self.env.action_space())
        
    def test_cannot_reset(self):
        with self.assertRaises(ChronicsError):
            obs = self.env.reset()
            
    def test_can_reset(self):
        self.env.chronics_handler.reset()
        obs = self.env.reset()
        self.env.step(self.env.action_space())
        
    def test_can_reset(self):
        self.env.chronics_handler.reset()
        obs = self.env.reset()
        
    def test_when_change_filter(self):
        self.env.chronics_handler.set_filter(lambda x: True)
        with self.assertRaises(ChronicsError):
            obs = self.env.reset()
        self.env.chronics_handler.reset()
        obs = self.env.reset()
        self.env.step(self.env.action_space())

    def test_runner(self):
        with self.assertRaises(ChronicsError):
            runner = Runner(**self.env.get_params_for_runner())
            res = runner.run(nb_episode=1,
                            nb_process=1,
                            max_iter=5
                            )
            
        self.env.chronics_handler.real_data.reset()
        runner = Runner(**self.env.get_params_for_runner())
        res = runner.run(nb_episode=1,
                         nb_process=1,
                         max_iter=5
                        )
    
    def test_copy(self):
        # when the copied env is not init
        env_cpy = self.env.copy()
        with self.assertRaises(ChronicsError):
            env_cpy.step(self.env.action_space())
        with self.assertRaises(ChronicsError):
            env_cpy.reset()
        env_cpy.chronics_handler.reset()
        obs = env_cpy.reset()
        env_cpy.step(env_cpy.action_space())
        env_cpy.close()

        # if the copied env is properly init
        self.env.chronics_handler.reset()
        env_cpy2 = self.env.copy()
        env_cpy2.chronics_handler.reset()
        obs = env_cpy2.reset()
        env_cpy2.step(env_cpy2.action_space())
        
    def test_runner_max_iter(self):
        self.env.chronics_handler.real_data.reset()
        runner = Runner(**self.env.get_params_for_runner())
        res = runner.run(nb_episode=1,
                         nb_process=1,
                         max_iter=5
                        )
        assert res[0][3] == 5, f"Is {res[0][3]} should be 5"
        assert res[0][4] == 5, f"Is {res[0][4]} should be 5"
        
        res = runner.run(nb_episode=1,
                         nb_process=1
                        )
        assert res[0][3] == 575, f"Is {res[0][3]} should be 575"
        assert res[0][4] == 575, f"Is {res[0][4]} should be 575"
        


if __name__ == "__main__":
    unittest.main()
