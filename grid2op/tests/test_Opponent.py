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
from grid2op.Opponent import BaseOpponent, RandomLineOpponent
from grid2op.Action import TopologyAction
from grid2op.MakeEnv import make
from grid2op.Opponent.BaseActionBudget import BaseActionBudget
from grid2op.dtypes import dt_int


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


class TestRandomLineOpponent(RandomLineOpponent):
    """test class to store the action picked"""
    picked_attack = None

    def reset(self, *args):
        super().reset(*args)
        TestRandomLineOpponent.picked_attack = None

    def attack(self, *args):
        attack = super().attack(*args)
        TestRandomLineOpponent.picked_attack = attack

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

    def test_RandomLineOpponent_not_enough_budget(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 3
            with make("rte_case14_realistic",
                      test=True,
                      opponent_init_budget=init_budget,
                      opponent_budget_per_ts=0.,
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=BaseActionBudget,
                      opponent_class=TestRandomLineOpponent) as env:
                obs = env.reset()
                assert env.oppSpace.budget == init_budget
                # The opponent can attack
                for i in range(init_budget):
                    obs, reward, done, info = env.step(env.action_space())
                    assert env.oppSpace.budget == init_budget - i - 1
                    assert env.opponent_class.picked_attack != env.action_space({})
                # There is no more budget
                assert env.oppSpace.budget == 0
                obs, reward, done, info = env.step(env.action_space())
                assert env.opponent_class.picked_attack == env.action_space({})

    def test_RandomLineOpponent_attackable_lines(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            tries = 30
            attackable_lines_case14 = ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]
            with make("rte_case14_realistic",
                      test=True,
                      opponent_init_budget=init_budget,
                      opponent_budget_per_ts=0.,
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=BaseActionBudget,
                      opponent_class=TestRandomLineOpponent) as env:
                # Collect some attacks and check that they belong to the correct lines
                for _ in range(tries):
                    obs = env.reset()
                    assert env.oppSpace.budget == init_budget
                    obs, reward, done, info = env.step(env.action_space())
                    assert env.oppSpace.budget == init_budget - 1

                    attack = env.opponent_class.picked_attack
                    attacked_line = attack.as_dict()['set_line_status']['disconnected_id'][0]
                    line_name = env.action_space.name_line[attacked_line]
                    assert line_name in attackable_lines_case14

    def test_RandomLineOpponent_one_line_at_a_time(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            tries = 30
            with make("rte_case14_realistic",
                      test=True,
                      opponent_init_budget=init_budget,
                      opponent_budget_per_ts=0.,
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=BaseActionBudget,
                      opponent_class=TestRandomLineOpponent) as env:
                # Collect some attacks and check that they belong to the correct lines
                for _ in range(tries):
                    obs = env.reset()
                    assert env.oppSpace.budget == init_budget
                    obs, reward, done, info = env.step(env.action_space())
                    assert env.oppSpace.budget == init_budget - 1

                    attack = env.opponent_class.picked_attack
                    n_disconnected = attack.as_dict()['set_line_status']['nb_disconnected']
                    assert n_disconnected == 1

    def test_RandomLineOpponent_env_updated(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            tries = 10
            with make("rte_case14_realistic",
                      test=True,
                      opponent_init_budget=init_budget,
                      opponent_budget_per_ts=0.,
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=BaseActionBudget,
                      opponent_class=TestRandomLineOpponent) as env:
                # Collect some attacks and check that they belong to the correct lines
                for j in range(tries):
                    obs = env.reset()
                    assert env.oppSpace.budget == init_budget
                    assert np.all(env.times_before_line_status_actionable == 0)
                    for i in range(env.oppSpace.attack_duration):
                        obs, reward, done, info = env.step(env.action_space())
                        assert env.oppSpace.budget == max(init_budget - i - 1, 0)

                        attack = env.opponent_class.picked_attack
                        attacked_line = attack.as_dict()['set_line_status']['disconnected_id'][0]
                        status_actionable = np.zeros_like(env.times_before_line_status_actionable).astype(dt_int)
                        status_actionable[attacked_line] = env.oppSpace.attack_duration - i - 1
                        assert np.all(env.times_before_line_status_actionable == status_actionable)

    def test_RandomLineOpponent_only_attack_connected(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            length = 100
            with make("rte_case14_realistic",
                      test=True,
                      opponent_init_budget=init_budget,
                      opponent_budget_per_ts=0.,
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=BaseActionBudget,
                      opponent_class=TestRandomLineOpponent) as env:
                # Collect some attacks and check that they belong to the correct lines
                env.oppSpace.opponent.downtime = 0 # only for this test
                pre_obs = env.reset()
                done = False
                assert env.oppSpace.budget == init_budget
                for i in range(length):
                    if done:
                        pre_obs = env.reset()
                    obs, reward, done, info = env.step(env.action_space())

                    attack = env.opponent_class.picked_attack
                    if attack == env.action_space({}): # all attackable lines are already disconnected
                        assert np.invert(pre_obs.line_status).sum() == 6 # the number of attackable lines
                        continue

                    attacked_line = attack.as_dict()['set_line_status']['disconnected_id'][0]
                    if env.oppSpace.current_attack_duration < env.oppSpace.attack_duration:
                        # The attack is ungoing. The line must have been disconnected already
                        assert not pre_obs.line_status[attacked_line]
                    else:
                        # A new attack was launched. The line must have been connected
                        assert pre_obs.line_status[attacked_line]
                    pre_obs = obs

if __name__ == "__main__":
    unittest.main()
