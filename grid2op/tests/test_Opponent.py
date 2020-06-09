# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import warnings
from grid2op.tests.helper_path_test import *
from grid2op.Opponent import BaseOpponent
from grid2op.Action import TopologyAction
from grid2op.MakeEnv import make
from grid2op.Opponent.BaseActionBudget import BaseActionBudget


class TestSuiteBudget_001(BaseActionBudget):
    """just for testing"""
    pass


class TestSuiteOpponent_001(BaseOpponent):
    """test class that disconnects randomly the powerlines"""
    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self.line_id = [0, 1, 2, 3]
        self.possible_attack = [self.action_space.disconnect_powerline(line_id=el) for el in self.line_id]
        self.do_nothing = self.action_space()

    def attack(self, observation, agent_action, env_action, budget, previous_fails):
        if observation is None:  # On first step
            return self.do_nothing
        attack = self.space_prng.choice(self.possible_attack)
        return attack


class TestLoadingOpp(unittest.TestCase):
    def test_creation_BaseOpponent(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env:
                my_opp = BaseOpponent(action_space=env.action_space)

    def test_env_modif_oppo(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True, opponent_class=TestSuiteOpponent_001) as env:
                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                assert isinstance(env.opponent, TestSuiteOpponent_001)

    def test_env_modif_oppobudg(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True, opponent_budget_class=TestSuiteBudget_001) as env:
                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                assert isinstance(env.compute_opp_budget, TestSuiteBudget_001)

    def test_env_modif_opponent_init_budget(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budg = 100.
            with make("rte_case5_example", test=True, opponent_init_budget=init_budg) as env:
                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                assert env.opponent_init_budget == init_budg

    def test_env_modif_opponent_init_budget_ts(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budg = 100.
            with make("rte_case5_example", test=True, opponent_budget_per_ts=init_budg) as env:
                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                assert env.opponent_budget_per_ts == init_budg

    def test_env_modif_opponent_action_class(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True, opponent_action_class=TopologyAction) as env:
                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                assert issubclass(env.opponent_action_class, TopologyAction)

    def test_env_opp_attack(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budg = 100.
            with make("rte_case5_example",
                      test=True,
                      opponent_init_budget=init_budg,
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=TestSuiteBudget_001,
                      opponent_class=TestSuiteOpponent_001) as env:
                obs = env.reset()
                assert env.opponent_init_budget == init_budg
                obs, reward, done, info = env.step(env.action_space())
                assert env.oppSpace.budget == init_budg - 1.0
                obs = env.reset()
                assert env.opponent_init_budget == init_budg
                assert env.oppSpace.budget == init_budg

    def test_env_opp_attack_budget_ts(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budg_ts = 0.5
            with make("rte_case5_example",
                      test=True,
                      opponent_budget_per_ts=init_budg_ts,
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=TestSuiteBudget_001,
                      opponent_class=TestSuiteOpponent_001) as env:
                obs = env.reset()
                assert env.opponent_init_budget == 0.
                obs, reward, done, info = env.step(env.action_space())
                # no attack possible
                assert env.oppSpace.budget == init_budg_ts
                obs, reward, done, info = env.step(env.action_space())
                # i can attack at the second time steps, and budget of an attack is 1, so I have 0 now
                assert env.oppSpace.budget == 0.

                obs = env.reset()
                assert env.opponent_init_budget == 0.
                assert env.opponent_budget_per_ts == 0.5
                assert env.oppSpace.budget == 0.

if __name__ == "__main__":
    unittest.main()
