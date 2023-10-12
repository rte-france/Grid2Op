# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import numpy as np
import warnings
import grid2op
from grid2op.Opponent import (
    GeometricOpponentMultiArea, 
    GeometricOpponent
)
from grid2op.Action import TopologyAction
from grid2op.Opponent.baseActionBudget import BaseActionBudget
from grid2op.dtypes import dt_int
from grid2op.Parameters import Parameters
from grid2op.Runner import Runner
import pdb

LINES_ATTACKED = ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]


class TestMultiAreaOpponentBasic(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.opponent = GeometricOpponentMultiArea(self.env.action_space)
        self.opponent.init(self.env,
                           lines_attacked=[LINES_ATTACKED[:3],LINES_ATTACKED[3:]],
                           attack_every_xxx_hour=24,
                           average_attack_duration_hour=4,
                           minimum_attack_duration_hour=2,
                           pmax_pmin_ratio=4)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_seed(self):
        self.opponent.seed(0)
        obs = self.env.reset()
        initial_budget = 250
        self.opponent.reset(initial_budget)
        assert np.all(self.opponent.list_opponents[0]._attack_times == [160])
        assert np.all(self.opponent.list_opponents[1]._attack_times == [182, 467])
              

class TestMultiAreaOpponent(unittest.TestCase):
    def setUp(self):
        # make an environment and check it works
        params = Parameters()
        params.NO_OVERFLOW_DISCONNECTION = True
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox",
                            test=True,
                            _add_to_name=type(self).__name__+"multiarea",
                            opponent_budget_per_ts=0.17*2,  # 0.17 per area
                            opponent_init_budget=1000,  # I don't really care much right now
                            opponent_attack_cooldown=0,  # otherwise it will not work
                            opponent_attack_duration=96,
                            opponent_budget_class=BaseActionBudget,
                            opponent_class=GeometricOpponentMultiArea,
                            opponent_action_class=TopologyAction,
                            param=params,
                            kwargs_opponent=dict(lines_attacked=[LINES_ATTACKED[:3],LINES_ATTACKED[3:]],
                                                 attack_every_xxx_hour=24,
                                                 average_attack_duration_hour=4,
                                                 minimum_attack_duration_hour=2,
                                                 pmax_pmin_ratio=4)
                            )
        self.env.seed(0)
        self.env.reset()
            
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_when_env_copied(self):
        # check it's properly propagated when copied
        env_cpy = self.env.copy()
        assert isinstance(env_cpy._opponent, GeometricOpponentMultiArea)
        
        # check it's properly propagated in the kwargs
        env_params = self.env.get_kwargs()
        assert env_params["opponent_class"] == GeometricOpponentMultiArea
        
        # check it's properly propagated in the runner
        runner_params = self.env.get_params_for_runner()
        assert runner_params["opponent_class"] == GeometricOpponentMultiArea
        runner = Runner(**runner_params)
        assert runner.opponent_class == GeometricOpponentMultiArea
        # check the runner can make an env with the right opponent space type
        env_runner = runner.init_env()
        assert isinstance(env_runner._opponent, GeometricOpponentMultiArea)
          
    def test_creation_ok(self):
        assert isinstance(self.env._opponent, GeometricOpponentMultiArea)
        assert isinstance(self.env._opponent.list_opponents[0], GeometricOpponent)
        assert isinstance(self.env._opponent.list_opponents[1], GeometricOpponent)
        assert np.all(self.env._opponent.list_opponents[0]._attack_times == [9, 370, 472])
        assert np.all(self.env._opponent.list_opponents[0]._attack_durations == [28, 53, 25])
        assert np.all(self.env._opponent.list_opponents[1]._attack_times == [345])
        assert np.all(self.env._opponent.list_opponents[1]._attack_durations == [55])
        
    def test_does_one_attack(self):
        """test a single opponent can attack at a given step (most basic)"""
        self.env._opponent.list_opponents[0]._attack_durations[0] = 3
        
        for ts in range(9):
            obs, reward, done, info = self.env.step(self.env.action_space())
            assert np.all(obs.line_status), f"error for {ts}"
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"][4]
        assert not obs.line_status[4]
        #attack continues
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"][4]
        assert not obs.line_status[4]
        #attack continues
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"][4]
        assert not obs.line_status[4]
        #attack continues
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"][4]
        assert not obs.line_status[4]
        # attack stops
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is None
        
    def test_does_two_simulatneous_attacks(self):
        """test both opponent can attack at a given step"""
        self.env._opponent.list_opponents[0]._attack_durations[0] = 3
        self.env._opponent.list_opponents[1]._attack_times[0] = 9
        self.env._opponent.list_opponents[1]._attack_waiting_times[0] = 9
        self.env._opponent.list_opponents[1]._attack_durations[0] = 2
        for ts in range(9):
            obs, reward, done, info = self.env.step(self.env.action_space())
            assert np.all(obs.line_status), f"error for {ts}"
        # attack starts
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"].sum() == 2
        assert info["opponent_attack_line"][4]
        assert info["opponent_attack_line"][14]
        assert not obs.line_status[4]
        assert not obs.line_status[14]
        # both attacks continue
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"].sum() == 2
        assert info["opponent_attack_line"][4]
        assert info["opponent_attack_line"][14]
        assert not obs.line_status[4]
        assert not obs.line_status[14]
        # both attacks continue
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"].sum() == 2
        assert info["opponent_attack_line"][4]
        assert info["opponent_attack_line"][14]
        assert not obs.line_status[4]
        assert not obs.line_status[14]
        # second attack stops
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"][4]
        assert info["opponent_attack_line"].sum() == 1
        assert not obs.line_status[4]
        # all attack have stoped
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is None
    
    def test_one_after_the_other(self):
        """test one opponent can attack after the other"""
        self.env._opponent.list_opponents[0]._attack_durations[0] = 3
        self.env._opponent.list_opponents[1]._attack_times[0] = 10
        self.env._opponent.list_opponents[1]._attack_waiting_times[0] = 10
        self.env._opponent.list_opponents[1]._attack_durations[0] = 3
        
        for ts in range(9):
            obs, reward, done, info = self.env.step(self.env.action_space())
            assert np.all(obs.line_status), f"error for {ts}"
            
        # first attack starts
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"].sum() == 1
        assert info["opponent_attack_line"][4]
        assert not obs.line_status[4]
        # second attack starts
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"].sum() == 2
        assert info["opponent_attack_line"][4]
        assert info["opponent_attack_line"][14]
        assert not obs.line_status[4]
        assert not obs.line_status[14]
        # both attacks continue
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"].sum() == 2
        assert info["opponent_attack_line"][4]
        assert info["opponent_attack_line"][14]
        assert not obs.line_status[4]
        assert not obs.line_status[14]
        # both attacks continue
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"].sum() == 2
        assert info["opponent_attack_line"][4]
        assert info["opponent_attack_line"][14]
        assert not obs.line_status[4]
        assert not obs.line_status[14]
        # first attack stops
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"][14]
        assert info["opponent_attack_line"].sum() == 1
        # all attack have stoped
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert info["opponent_attack_line"] is None
        
        
if __name__ == "__main__":
    unittest.main()
