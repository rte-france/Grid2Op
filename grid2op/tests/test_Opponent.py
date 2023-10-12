# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import tempfile
import warnings
import unittest

import grid2op
from grid2op.Opponent.opponentSpace import OpponentSpace
from grid2op.tests.helper_path_test import *
from grid2op.Chronics import ChangeNothing
from grid2op.Opponent import (
    BaseOpponent,
    RandomLineOpponent,
    WeightedRandomOpponent,
    GeometricOpponent
)
from grid2op.Action import TopologyAction
from grid2op.Opponent.baseActionBudget import BaseActionBudget
from grid2op.dtypes import dt_int
from grid2op.Parameters import Parameters
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData
from grid2op.Environment import SingleEnvMultiProcess
from grid2op.Exceptions import OpponentError

import pdb

ATTACK_DURATION = 48
ATTACK_COOLDOWN = 100
LINES_ATTACKED = ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]
RHO_NORMALIZATION = [1, 1, 1, 1, 1, 1]


class TestSuiteBudget_001(BaseActionBudget):
    """just for testing"""

    pass


class TestSuiteOpponent_001(BaseOpponent):
    """test class that disconnects randomly the powerlines"""

    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self.line_id = [0, 1, 2, 3]
        self.possible_attack = [
            self.action_space.disconnect_powerline(line_id=el) for el in self.line_id
        ]

    def attack(self, observation, agent_action, env_action, budget, previous_fails):
        if observation is None:  # On first step
            return None
        attack = self.space_prng.choice(self.possible_attack)
        return attack, None


class TestWeightedRandomOpponent(WeightedRandomOpponent):
    def init(self, lines_attacked=[], rho_normalization=[], **kwargs):
        WeightedRandomOpponent.init(
            self,
            lines_attacked=lines_attacked,
            rho_normalization=rho_normalization,
            **kwargs,
        )
        self._attack_counter = 0
        self._attack_continues_counter = 0

    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        self._attack_continues_counter += 1
        WeightedRandomOpponent.tell_attack_continues(
            self, observation, agent_action, env_action, budget
        )

    def attack(self, observation, agent_action, env_action, budget, previous_fails):
        self._attack_counter += 1
        return WeightedRandomOpponent.attack(
            self, observation, agent_action, env_action, budget, previous_fails
        )


class TestLoadingOpp(unittest.TestCase):
    def test_creation_BaseOpponent(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__) as env:
                my_opp = BaseOpponent(action_space=env.action_space)

    def test_env_modif_oppo(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "rte_case5_example", test=True, opponent_class=TestSuiteOpponent_001,
                _add_to_name=type(self).__name__
            ) as env:
                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                assert isinstance(env._opponent, TestSuiteOpponent_001)

    def test_env_modif_oppobudg(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "rte_case5_example",
                test=True,
                opponent_budget_class=TestSuiteBudget_001,
                _add_to_name=type(self).__name__,
            ) as env:
                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                assert isinstance(env._compute_opp_budget, TestSuiteBudget_001)

    def test_env_modif_opponent_init_budget(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budg = 100.0
            with grid2op.make(
                "rte_case5_example", test=True, opponent_init_budget=init_budg,
                _add_to_name=type(self).__name__
            ) as env:
                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                assert env._opponent_init_budget == init_budg

    def test_env_modif_opponent_init_budget_ts(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budg = 100.0
            with grid2op.make(
                "rte_case5_example", test=True, opponent_budget_per_ts=init_budg,
                _add_to_name=type(self).__name__
            ) as env:
                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                assert env._opponent_budget_per_ts == init_budg

    def test_env_modif_opponent_action_class(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "rte_case5_example", test=True, opponent_action_class=TopologyAction,
                _add_to_name=type(self).__name__
            ) as env:
                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                assert issubclass(env._opponent_action_class, TopologyAction)

    def test_env_opp_attack(self):
        # and test reset, which apparently is NOT done correctly
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budg = 100.0
            with grid2op.make(
                "rte_case5_example",
                test=True,
                opponent_init_budget=init_budg,
                opponent_action_class=TopologyAction,
                opponent_budget_class=TestSuiteBudget_001,
                opponent_attack_duration=ATTACK_DURATION,
                opponent_attack_cooldown=ATTACK_COOLDOWN,
                opponent_class=TestSuiteOpponent_001,
                _add_to_name=type(self).__name__,
            ) as env:
                obs = env.reset()
                # opponent should not attack at the first time step
                assert np.all(obs.line_status)
                assert env._opponent_init_budget == init_budg
                obs, reward, done, info = env.step(env.action_space())
                assert env._oppSpace.budget == init_budg - 1.0

                obs = env.reset()
                # opponent should not attack at the first time step
                assert np.all(obs.line_status)
                assert env._opponent_init_budget == init_budg
                assert env._oppSpace.budget == init_budg

    def test_env_opp_attack_budget_ts(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budg_ts = 0.5
            with grid2op.make(
                "rte_case5_example",
                test=True,
                opponent_budget_per_ts=init_budg_ts,
                opponent_attack_duration=1,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=TestSuiteBudget_001,
                opponent_attack_cooldown=ATTACK_COOLDOWN,
                opponent_class=TestSuiteOpponent_001,
                _add_to_name=type(self).__name__,
            ) as env:
                obs = env.reset()
                assert env._opponent_init_budget == 0.0
                obs, reward, done, info = env.step(env.action_space())
                # no attack possible
                assert env._oppSpace.budget == init_budg_ts
                obs, reward, done, info = env.step(env.action_space())
                # i can attack at the second time steps, and budget of an attack is 1, so I have 0 now
                assert env._oppSpace.budget == 0.0

                obs = env.reset()
                assert env._opponent_init_budget == 0.0
                assert env._opponent_budget_per_ts == 0.5
                assert env._oppSpace.budget == 0.0

    def test_RandomLineOpponent_not_enough_budget(self):
        """Tests that the attack is ignored when the budget is too low"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 50
            param = Parameters()
            param.NO_OVERFLOW_DISCONNECTION = True  # otherwise there's a game over
            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                param=param,
                opponent_attack_cooldown=0,  # only for testing
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=ATTACK_DURATION,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                obs = env.reset()
                assert env._oppSpace.budget == init_budget
                # The opponent can attack
                for i in range(env._oppSpace.attack_max_duration):
                    obs, reward, done, info = env.step(env.action_space())
                    attack = env._oppSpace.last_attack
                    assert env._oppSpace.budget == init_budget - i - 1
                    assert any(attack._set_line_status != 0)

                # There is not enough budget for a second attack
                assert (
                    abs(env._oppSpace.budget - (init_budget - ATTACK_DURATION)) <= 1e-5
                )
                obs, reward, done, info = env.step(env.action_space())
                attack = env._oppSpace.last_attack
                assert attack is None

    def test_RandomLineOpponent_attackable_lines(self):
        """Tests that the RandomLineOpponent only attacks the authorized lines"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            tries = 30
            attackable_lines_case14 = LINES_ATTACKED
            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=ATTACK_DURATION,
                opponent_attack_cooldown=ATTACK_COOLDOWN,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                # Collect some attacks and check that they belong to the correct lines
                for _ in range(tries):
                    obs = env.reset()
                    assert env._oppSpace.budget == init_budget
                    obs, reward, done, info = env.step(env.action_space())
                    assert env._oppSpace.budget == init_budget - 1

                    attack = env._oppSpace.last_attack
                    attacked_line = np.where(attack._set_line_status == -1)[0][0]
                    line_name = env.action_space.name_line[attacked_line]
                    assert line_name in attackable_lines_case14

    def test_RandomLineOpponent_disconnects_only_one_line(self):
        """Tests that the RandomLineOpponent does not disconnect several lines at a time"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            tries = 30
            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=ATTACK_DURATION,
                opponent_attack_cooldown=ATTACK_COOLDOWN,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                # Collect some attacks and check that they belong to the correct lines
                for _ in range(tries):
                    obs = env.reset()
                    assert env._oppSpace.budget == init_budget
                    obs, reward, done, info = env.step(env.action_space())
                    assert env._oppSpace.budget == init_budget - 1

                    attack = env._oppSpace.last_attack
                    n_disconnected = np.sum(attack._set_line_status == -1)
                    assert n_disconnected == 1

    def test_RandomLineOpponent_with_agent(self):
        """Tests that the line status cooldown is correctly updated when the opponent attacks a line with an agent"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # to prevent attack on first time step
            init_budget = 8
            opponent_budget_per_ts = 1
            length = 300
            agent_line_cooldown = 30
            attack_duration = 10
            attack_cooldown = 20000  # i do one attack
            param = Parameters()
            param.NO_OVERFLOW_DISCONNECTION = True
            param.NB_TIMESTEP_COOLDOWN_LINE = agent_line_cooldown
            line_opponent_attack = 4
            line_opponent_attack = 15
            lines_attacked = ["3_6_15"]
            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                param=param,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=opponent_budget_per_ts,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=attack_duration,
                opponent_attack_cooldown=attack_cooldown,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": lines_attacked},
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                obs = env.reset()
                reward = 0
                assert env._oppSpace.budget == init_budget
                assert np.all(obs.time_before_cooldown_line == 0)
                # the "agent" does an action (on the same powerline as the opponent attacks)
                obs, reward, done, info = env.step(
                    env.action_space({"set_line_status": [(line_opponent_attack, 1)]})
                )
                assert np.all(obs.line_status)
                assert (
                    obs.time_before_cooldown_line[line_opponent_attack]
                    == agent_line_cooldown
                )

                # check that the opponent cooldown is not taken into account (lower than the cooldown on line)
                for i in range(10):
                    obs, reward, done, info = env.step(env.action_space())
                    assert "opponent_attack_line" in info
                    assert (
                        np.sum(info["opponent_attack_line"]) == 1
                    ), "error at iteration {} for attack".format(i)
                    assert info["opponent_attack_line"][line_opponent_attack]
                    assert obs.time_before_cooldown_line[
                        line_opponent_attack
                    ] == agent_line_cooldown - (i + 1), "error at iteration {}".format(
                        i
                    )

                obs, reward, done, info = env.step(env.action_space())
                assert "opponent_attack_line" in info
                assert info["opponent_attack_line"] is None  # no more attack
                assert (
                    obs.time_before_cooldown_line[line_opponent_attack]
                    == agent_line_cooldown - 11
                )

    def test_RandomLineOpponent_with_maintenance_1(self):
        """Tests that the line status cooldown is correctly updated when the opponent attacks a line with an agent"""

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # to prevent attack on first time step
            init_budget = 8
            opponent_budget_per_ts = 1
            length = 300
            agent_line_cooldown = 30
            attack_duration = 5
            attack_cooldown = 20000  # i do one attack
            param = Parameters()
            param.NO_OVERFLOW_DISCONNECTION = True
            param.NB_TIMESTEP_COOLDOWN_LINE = agent_line_cooldown
            line_opponent_attack = 4
            line_opponent_attack = 11

            # 1. attack is at the same time than the maintenance
            lines_attacked = ["8_13_11"]
            with grid2op.make(
                os.path.join(PATH_CHRONICS, "env_14_test_maintenance"),
                test=True,
                param=param,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=opponent_budget_per_ts,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=attack_duration,
                opponent_attack_cooldown=attack_cooldown,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": lines_attacked},
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                # the opponent has attacked
                assert "opponent_attack_line" in info
                assert (
                    np.sum(info["opponent_attack_line"]) == 1
                ), "error at iteration {} for attack".format(0)
                assert info["opponent_attack_line"][line_opponent_attack]
                # but the maintenance cooldown has priority (longer)
                assert np.all(
                    obs.time_before_cooldown_line
                    == np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0],
                        dtype=dt_int,
                    )
                )

            # 2. attack is before than the maintenance
            init_budget = 8
            opponent_budget_per_ts = 1
            attack_duration = 5
            lines_attacked = ["9_10_12"]
            line_id = 12
            with grid2op.make(
                os.path.join(PATH_CHRONICS, "env_14_test_maintenance"),
                test=True,
                param=param,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=opponent_budget_per_ts,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=attack_duration,
                opponent_attack_cooldown=attack_cooldown,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": lines_attacked},
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                obs = env.reset()
                env.fast_forward_chronics(274)
                obs, reward, done, info = env.step(env.action_space())
                # i have a maintenance in 1 time step
                assert obs.time_next_maintenance[line_id] == 1
                # the opponent has attacked at this time step
                assert "opponent_attack_line" in info
                assert info["opponent_attack_line"] is not None
                assert info["opponent_attack_line"][line_id]
                assert info["opponent_attack_duration"] == 4
                # cooldown should be updated correctly
                assert np.all(
                    obs.time_before_cooldown_line
                    == np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
                        dtype=dt_int,
                    )
                )

                for i in range(3):
                    obs, reward, done, info = env.step(
                        env.action_space()
                    )  # the maintenance is happening
                    # i have a maintenance in 1 time step
                    assert obs.time_next_maintenance[line_id] == 0
                    # the attack continued
                    assert "opponent_attack_line" in info
                    assert info["opponent_attack_line"] is not None
                    assert info["opponent_attack_line"][line_id]
                    assert info["opponent_attack_duration"] == 3 - i
                    assert np.all(
                        obs.time_before_cooldown_line
                        == np.array(
                            [
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                12 - i,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                            ],
                            dtype=dt_int,
                        )
                    )

                # attack should be over
                obs, reward, done, info = env.step(
                    env.action_space()
                )  # the maintenance is happening
                # i have a maintenance in 1 time step
                assert obs.time_next_maintenance[line_id] == 0
                # the attack continued
                assert "opponent_attack_line" in info
                assert info["opponent_attack_line"] is None
                assert info["opponent_attack_duration"] == 0
                assert np.all(
                    obs.time_before_cooldown_line
                    == np.array(
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            12 - 3,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        dtype=dt_int,
                    )
                )

    def test_RandomLineOpponent_only_attack_connected(self):
        """
        Tests that the RandomLineOpponent does not attack lines that are already disconnected
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 10000
            length = 300
            env = grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_attack_cooldown=0,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=ATTACK_DURATION,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__,
            )
            env.seed(0)
            # Collect some attacks
            # and check that they belong to the correct lines
            pre_obs = env.reset()
            done = False
            assert env._oppSpace.budget == init_budget
            for i in range(length):
                obs, reward, done, info = env.step(env.action_space())

                attack = env._oppSpace.last_attack
                attacked_line = np.where(attack._set_line_status == -1)[0][0]
                if (
                    env._oppSpace.current_attack_duration
                    < env._oppSpace.attack_max_duration
                ):
                    # The attack is ungoing. The line must have been disconnected already
                    assert not pre_obs.line_status[attacked_line]
                else:
                    # A new attack was launched. The line must have been connected
                    assert pre_obs.line_status[attacked_line]

                pre_obs = obs
                if done:
                    pre_obs = env.reset()

    def test_RandomLineOpponent_same_attack_order_and_attacks_all_lines(self):
        """Tests that the RandomLineOpponent has the same attack order (when seeded) and attacks all lines"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            length = 30
            # new attack order in version 1.6.5 because of the new reset method
            expected_attack_order = [
                4,
                12,
                14,
                14,
                12,
                13,
                3,
                15,
                15,
                12,
                4,
                15,
                14,
                12,
                15,
                4,
                4,
                3,
                15,
                13,
                12,
                14,
                12,
            ]
            attack_order = []
            has_disconnected_all = False
            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_attack_cooldown=0,  # only for testing
                opponent_attack_duration=1,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__,
            ) as env:
                env.seed(0)
                # Collect some attacks and check that they belong to the correct lines
                obs = env.reset()
                done = False
                assert env._oppSpace.budget == init_budget
                for i in range(length):
                    if done:
                        obs = env.reset()
                    pre_done = done
                    obs, reward, done, info = env.step(env.action_space())

                    attack = env._oppSpace.last_attack
                    if (
                        attack is None and not done
                    ):  # should only happen here if all attackable lines are already disconnected
                        assert np.sum(obs.line_status == False) == 6
                        continue
                    elif done:
                        continue

                    assert any(attack._set_line_status == -1)
                    attacked_line = np.where(attack._set_line_status == -1)[0][0]
                    if pre_done or not (
                        attack_order and attack_order[-1] == attacked_line
                    ):
                        attack_order.append(attacked_line)

                assert len(set(attack_order)) == 6
                assert attack_order == expected_attack_order

    def test_simulate(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            opponent_attack_duration = 15
            opponent_attack_cooldown = 20
            line_id = 4
            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_attack_cooldown=opponent_attack_cooldown,
                opponent_attack_duration=opponent_attack_duration,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__,
            ) as env:
                env.seed(0)
                reco_line = env.action_space({"set_line_status": [(line_id, 1)]})

                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                assert obs.rho[line_id] == 0.0
                assert not obs.line_status[line_id]
                simobs, sim_r, sim_d, sim_info = obs.simulate(env.action_space())
                assert simobs.rho[line_id] == 0.0
                assert not simobs.line_status[line_id]
                simobs, sim_r, sim_d, sim_info = obs.simulate(reco_line)
                assert simobs.rho[line_id] == 0.0
                assert not simobs.line_status[line_id]
                obs, reward, done, info = env.step(reco_line)
                assert obs.rho[line_id] == 0.0
                assert not obs.line_status[line_id]

                # check that the budget of the opponent in the ObsEnv does not decrease
                for i in range(opponent_attack_duration):
                    simobs, sim_r, sim_d, sim_info = obs.simulate(reco_line)
                    assert simobs.rho[line_id] == 0.0
                    assert not simobs.line_status[line_id]

                # check that the opponent continue its attacks
                for i in range(opponent_attack_duration - 2):
                    obs, reward, done, info = env.step(reco_line)
                    assert obs.rho[line_id] == 0.0
                    assert not obs.line_status[line_id]

                # i should be able to simulate a reconnection now
                simobs, sim_r, sim_d, sim_info = obs.simulate(reco_line)
                assert simobs.rho[line_id] > 0.0
                assert simobs.line_status[line_id]
                # this should not affect the environment
                assert obs.rho[line_id] == 0.0
                assert not obs.line_status[line_id]

                # and now that i'm able to reconnect the powerline in step
                obs, reward, done, info = env.step(reco_line)
                assert obs.rho[line_id] > 0.0
                assert obs.line_status[line_id]

    def test_opponent_load(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "rte_case5_example",
                test=True,
                opponent_action_class=TopologyAction,
                opponent_class=RandomLineOpponent,
                _add_to_name=type(self).__name__,
            ) as env_1:
                env_1.seed(0)
                obs, reward, done, info = env_1.step(env_1.action_space())
            with grid2op.make(
                "rte_case118_example",
                test=True,
                opponent_action_class=TopologyAction,
                opponent_class=RandomLineOpponent,
                _add_to_name=type(self).__name__,
            ) as env_2:
                env_2.seed(0)
                obs, reward, done, info = env_2.step(env_2.action_space())

    def test_proper_action_class(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            opponent_attack_duration = 15
            opponent_attack_cooldown = 20
            line_id = 4
            opponent_action_class = TopologyAction

            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_attack_cooldown=opponent_attack_cooldown,
                opponent_attack_duration=opponent_attack_duration,
                opponent_action_class=opponent_action_class,
                opponent_budget_class=BaseActionBudget,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__,
            ) as env:
                env.seed(0)
                assert env._opponent_action_class == opponent_action_class
                assert issubclass(
                    env._oppSpace.action_space.actionClass, opponent_action_class
                )
                assert issubclass(
                    env._opponent_action_space.actionClass, opponent_action_class
                )
                opp_space = env._oppSpace
                attack, duration = opp_space.attack(
                    env.get_obs(), env.action_space(), env.action_space()
                )
                assert isinstance(attack, opponent_action_class)

    def test_get_set_state(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            opponent_attack_duration = 15
            opponent_attack_cooldown = 20
            line_id = 4

            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_attack_cooldown=opponent_attack_cooldown,
                opponent_attack_duration=opponent_attack_duration,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__,
            ) as env:

                env.seed(0)
                agent_action = env.action_space()
                observation = env.get_obs()
                env_action = env.action_space()

                opp_space = env._oppSpace
                # FIRST CHECK: WHEN NO ATTACK ARE PERFORMED
                # test that if i do "a loop of get / set" i get the same stuff
                init_state = opp_space._get_state()
                opp_space._set_state(*init_state)
                second_init_state = opp_space._get_state()
                assert np.all(init_state == second_init_state)

                # now do absolutely anything
                for i in range(70):
                    opp_space.attack(observation, agent_action, env_action)
                # check that indeed the state should have changed
                other_state = opp_space._get_state()
                assert np.any(init_state != other_state)

                # check that if i set the state back, the
                opp_space._set_state(*init_state)
                second_init_state = opp_space._get_state()
                assert np.all(init_state == second_init_state)
                # note due to the "random effect" we don't impose the opponent to act on the same line again...
                # this normal and should be explained in the notebooks.

                # SECOND CHECK WHEN AN ATTACK NEED TO BE CONTINUED
                # now i do an attack that should be continues
                attack1 = opp_space.attack(observation, agent_action, env_action)
                init_state = opp_space._get_state()
                for i in range(70):
                    opp_space.attack(observation, agent_action, env_action)
                opp_space._set_state(*init_state)
                second_init_state = opp_space._get_state()
                assert np.all(init_state == second_init_state)

                # this time the attack continues, so it should be same
                attack2 = opp_space.attack(observation, agent_action, env_action)
                # attack are the same
                assert np.all(attack1[0].to_vect() == attack2[0].to_vect())
                # the second time i attacked twice, the first one only once, i check the budget
                assert np.all(attack1[1] == attack2[1] + 1)

    def test_withrunner(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            opponent_attack_duration = 15
            opponent_attack_cooldown = 30
            opponent_budget_per_ts = 0.0
            opponent_action_class = TopologyAction
            line_id = 3

            p = Parameters()
            p.NO_OVERFLOW_DISCONNECTION = True
            env = grid2op.make(
                "rte_case14_realistic",
                test=True,
                param=p,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=opponent_budget_per_ts,
                opponent_attack_cooldown=opponent_attack_cooldown,
                opponent_attack_duration=opponent_attack_duration,
                opponent_action_class=opponent_action_class,
                opponent_budget_class=BaseActionBudget,
                opponent_class=RandomLineOpponent,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__,
            )
            env.seed(0)
            runner = Runner(**env.get_params_for_runner())
            assert runner.opponent_init_budget == init_budget
            assert runner.opponent_budget_per_ts == opponent_budget_per_ts
            assert runner.opponent_attack_cooldown == opponent_attack_cooldown
            assert runner.opponent_attack_duration == opponent_attack_duration
            assert runner.opponent_action_class == opponent_action_class

            f = tempfile.mkdtemp()
            res = runner.run(
                nb_episode=1,
                env_seeds=[4],
                agent_seeds=[0],
                max_iter=opponent_attack_cooldown - 1,
                path_save=f,
            )
            for i, episode_name, cum_reward, timestep, total_ts in res:
                episode_data = EpisodeData.from_disk(agent_path=f, name=episode_name)
                assert np.any(
                    episode_data.attacks.collection[:, line_id] == -1.0
                ), "no attack on powerline {}".format(line_id)
                assert (
                    np.sum(episode_data.attacks.collection[:, line_id])
                    == -opponent_attack_duration
                ), "too much / not enought attack on powerline {}".format(line_id)
                assert np.all(episode_data.attacks.collection[:, 0] == 0.0)

    def test_env_opponent(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_opponent", test=True, param=param, _add_to_name=type(self).__name__)
        env.seed(0)  # make sure i have reproducible experiments
        obs = env.reset()
        assert env._oppSpace.budget == 0
        assert np.all(obs.line_status)
        obs, reward, done, info = env.step(env.action_space())
        assert env._oppSpace.budget == 0.5
        assert np.all(obs.line_status)
        obs, reward, done, info = env.step(env.action_space())

        env.close()

    def test_multienv_opponent(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_opponent", test=True, param=param, _add_to_name=type(self).__name__)
        env.seed(0)  # make sure i have reproducible experiments
        multi_env = SingleEnvMultiProcess(env=env, nb_env=2)
        obs = multi_env.reset()
        for ob in obs:
            assert np.all(ob.line_status)
        assert np.all(multi_env._opponent[0]._lines_ids == [3, 4, 15, 12, 13, 14])
        assert np.all(multi_env._opponent[1]._lines_ids == [3, 4, 15, 12, 13, 14])
        env.close()
        multi_env.close()

    def test_WeightedRandomOpponent_not_enough_budget(self):
        """Tests that the attack is ignored when the budget is too low"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 50
            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_attack_cooldown=1,  # only for testing
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=ATTACK_DURATION,
                opponent_class=WeightedRandomOpponent,
                kwargs_opponent={
                    "lines_attacked": LINES_ATTACKED,
                    "rho_normalization": RHO_NORMALIZATION,
                    "attack_period": 1,
                },
                _add_to_name=type(self).__name__,
            ) as env:
                env.seed(0)
                obs = env.reset()
                assert env._oppSpace.budget == init_budget
                # The opponent can attack
                for i in range(env._oppSpace.attack_max_duration):
                    obs, reward, done, info = env.step(env.action_space())
                    attack = env._oppSpace.last_attack
                    assert env._oppSpace.budget == init_budget - i - 1
                    assert any(attack._set_line_status != 0)

                # There is not enough budget for a second attack
                assert (
                    abs(env._oppSpace.budget - (init_budget - ATTACK_DURATION)) <= 1e-5
                )
                obs, reward, done, info = env.step(env.action_space())
                attack = env._oppSpace.last_attack
                assert attack is None

    def test_WeightedRandomOpponent_attackable_lines(self):
        """Tests that the WeightedRandomOpponent only attacks the authorized lines"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            tries = 30
            attackable_lines_case14 = LINES_ATTACKED
            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=ATTACK_DURATION,
                opponent_attack_cooldown=ATTACK_COOLDOWN,
                opponent_class=WeightedRandomOpponent,
                kwargs_opponent={
                    "lines_attacked": LINES_ATTACKED,
                    "rho_normalization": RHO_NORMALIZATION,
                    "attack_period": 1,
                },
                _add_to_name=type(self).__name__,
            ) as env:
                env.seed(0)
                # Collect some attacks and check that they belong to the correct lines
                for _ in range(tries):
                    obs = env.reset()
                    assert env._oppSpace.budget == init_budget
                    obs, reward, done, info = env.step(env.action_space())
                    assert env._oppSpace.budget == init_budget - 1

                    attack = env._oppSpace.last_attack
                    attacked_line = np.where(attack._set_line_status == -1)[0][0]
                    line_name = env.action_space.name_line[attacked_line]
                    assert line_name in attackable_lines_case14

    def test_WeightedRandomOpponent_disconnects_only_one_line(self):
        """Tests that the WeightedRandomOpponent does not disconnect several lines at a time"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            tries = 30
            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=ATTACK_DURATION,
                opponent_attack_cooldown=ATTACK_COOLDOWN,
                opponent_class=WeightedRandomOpponent,
                kwargs_opponent={
                    "lines_attacked": LINES_ATTACKED,
                    "rho_normalization": RHO_NORMALIZATION,
                    "attack_period": 1,
                },
                _add_to_name=type(self).__name__,
            ) as env:
                env.seed(0)
                # Collect some attacks and check that they belong to the correct lines
                for _ in range(tries):
                    obs = env.reset()
                    assert env._oppSpace.budget == init_budget
                    obs, reward, done, info = env.step(env.action_space())
                    assert env._oppSpace.budget == init_budget - 1

                    attack = env._oppSpace.last_attack
                    n_disconnected = np.sum(attack._set_line_status == -1)
                    assert n_disconnected == 1

    def test_WeightedRandomOpponent_with_agent(self):
        """Tests that the line status cooldown is correctly updated when the opponent attacks a line with an agent"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # to prevent attack on first time step
            init_budget = 8
            opponent_budget_per_ts = 1
            length = 300
            agent_line_cooldown = 30
            attack_duration = 10
            attack_cooldown = 20000  # i do one attack
            param = Parameters()
            param.NO_OVERFLOW_DISCONNECTION = True
            param.NB_TIMESTEP_COOLDOWN_LINE = agent_line_cooldown
            line_opponent_attack = 4
            line_opponent_attack = 15
            lines_attacked = ["3_6_15"]
            rho_normalization = [1]
            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                param=param,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=opponent_budget_per_ts,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=attack_duration,
                opponent_attack_cooldown=attack_cooldown,
                opponent_class=WeightedRandomOpponent,
                kwargs_opponent={
                    "lines_attacked": lines_attacked,
                    "rho_normalization": rho_normalization,
                    "attack_period": 1,
                },
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                obs = env.reset()
                reward = 0
                assert env._oppSpace.budget == init_budget
                assert np.all(obs.time_before_cooldown_line == 0)
                # the "agent" does an action (on the same powerline as the opponent attacks)
                obs, reward, done, info = env.step(
                    env.action_space({"set_line_status": [(line_opponent_attack, 1)]})
                )
                assert np.all(obs.line_status)
                assert (
                    obs.time_before_cooldown_line[line_opponent_attack]
                    == agent_line_cooldown
                )

                # check that the opponent cooldown is not taken into account (lower than the cooldown on line)
                for i in range(10):
                    obs, reward, done, info = env.step(env.action_space())
                    assert "opponent_attack_line" in info
                    assert (
                        np.sum(info["opponent_attack_line"]) == 1
                    ), "error at iteration {} for attack".format(i)
                    assert info["opponent_attack_line"][line_opponent_attack]
                    assert obs.time_before_cooldown_line[
                        line_opponent_attack
                    ] == agent_line_cooldown - (i + 1), "error at iteration {}".format(
                        i
                    )

                obs, reward, done, info = env.step(env.action_space())
                assert "opponent_attack_line" in info
                assert info["opponent_attack_line"] is None  # no more attack
                assert (
                    obs.time_before_cooldown_line[line_opponent_attack]
                    == agent_line_cooldown - 11
                )

    def test_WeightedRandomOpponent_with_maintenance_1(self):
        """Tests that the line status cooldown is correctly updated when the opponent attacks a line with an agent"""

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # to prevent attack on first time step
            init_budget = 8
            opponent_budget_per_ts = 1
            length = 300
            agent_line_cooldown = 30
            attack_duration = 5
            attack_cooldown = 20000  # i do one attack
            param = Parameters()
            param.NO_OVERFLOW_DISCONNECTION = True
            param.NB_TIMESTEP_COOLDOWN_LINE = agent_line_cooldown
            line_opponent_attack = 4
            line_opponent_attack = 11

            # 1. attack is at the same time than the maintenance
            lines_attacked = ["8_13_11"]
            rho_normalization = [1]
            with grid2op.make(
                os.path.join(PATH_CHRONICS, "env_14_test_maintenance"),
                test=True,
                param=param,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=opponent_budget_per_ts,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=attack_duration,
                opponent_attack_cooldown=attack_cooldown,
                opponent_class=WeightedRandomOpponent,
                kwargs_opponent={
                    "lines_attacked": lines_attacked,
                    "rho_normalization": rho_normalization,
                    "attack_period": 1,
                },
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                # the opponent has attacked
                assert "opponent_attack_line" in info
                assert (
                    np.sum(info["opponent_attack_line"]) == 1
                ), "error at iteration 0 for attack"
                assert info["opponent_attack_line"][line_opponent_attack]
                # but the maintenance cooldown has priority (longer)
                assert np.all(
                    obs.time_before_cooldown_line
                    == np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0],
                        dtype=dt_int,
                    )
                )

            # 2. attack is before than the maintenance
            init_budget = 8
            opponent_budget_per_ts = 1
            attack_duration = 5
            lines_attacked = ["9_10_12"]
            rho_normalization = [1]
            line_id = 12
            with grid2op.make(
                os.path.join(PATH_CHRONICS, "env_14_test_maintenance"),
                test=True,
                param=param,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=opponent_budget_per_ts,
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=attack_duration,
                opponent_attack_cooldown=attack_cooldown,
                opponent_class=WeightedRandomOpponent,
                kwargs_opponent={
                    "lines_attacked": lines_attacked,
                    "rho_normalization": rho_normalization,
                    "attack_period": 1,
                },
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                obs = env.reset()
                env.fast_forward_chronics(274)
                obs, reward, done, info = env.step(env.action_space())
                # i have a maintenance in 1 time step
                assert obs.time_next_maintenance[line_id] == 1
                # the opponent has attacked at this time step
                assert "opponent_attack_line" in info
                assert info["opponent_attack_line"] is not None
                assert info["opponent_attack_line"][line_id]
                assert info["opponent_attack_duration"] == 4
                # cooldown should be updated correctly
                assert np.all(
                    obs.time_before_cooldown_line
                    == np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
                        dtype=dt_int,
                    )
                )
                for i in range(3):
                    obs, reward, done, info = env.step(
                        env.action_space()
                    )  # the maintenance is happening
                    # i have a maintenance in 1 time step
                    assert obs.time_next_maintenance[line_id] == 0
                    # the attack continued
                    assert "opponent_attack_line" in info
                    assert info["opponent_attack_line"] is not None
                    assert info["opponent_attack_line"][line_id]
                    assert info["opponent_attack_duration"] == 3 - i
                    assert np.all(
                        obs.time_before_cooldown_line
                        == np.array(
                            [
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                12 - i,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                            ],
                            dtype=dt_int,
                        )
                    )

                # attack should be over
                obs, reward, done, info = env.step(
                    env.action_space()
                )  # the maintenance is happening
                # i have a maintenance in 1 time step
                assert obs.time_next_maintenance[line_id] == 0
                # the attack continued
                assert "opponent_attack_line" in info
                assert info["opponent_attack_line"] is None
                assert info["opponent_attack_duration"] == 0
                assert np.all(
                    obs.time_before_cooldown_line
                    == np.array(
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            12 - 3,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        dtype=dt_int,
                    )
                )

    def test_WeightedRandomOpponent_only_attack_connected(self):
        """
        Tests that the WeightedRandomOpponent does not attack lines that are already disconnected
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 10000
            length = 300
            env = grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_attack_cooldown=1,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_attack_duration=ATTACK_DURATION,
                opponent_class=WeightedRandomOpponent,
                kwargs_opponent={
                    "lines_attacked": LINES_ATTACKED,
                    "rho_normalization": RHO_NORMALIZATION,
                    "attack_period": 1,
                },
                _add_to_name=type(self).__name__
            )
            env.seed(0)
            # Collect some attacks
            # and check that they belong to the correct lines
            pre_obs = env.reset()
            done = False
            assert env._oppSpace.budget == init_budget
            for i in range(length):
                obs, reward, done, info = env.step(env.action_space())

                attack = env._oppSpace.last_attack
                attacked_line = np.where(attack._set_line_status == -1)[0][0]
                if (
                    env._oppSpace.current_attack_duration
                    < env._oppSpace.attack_max_duration
                ):
                    # The attack is ungoing. The line must have been disconnected already
                    assert not pre_obs.line_status[attacked_line]
                else:
                    # A new attack was launched. The line must have been connected
                    assert pre_obs.line_status[attacked_line]

                pre_obs = obs
                if done:
                    pre_obs = env.reset()

    def test_WeightedRandomOpponent_same_attack_order_and_attacks_all_lines(self):
        """Tests that the WeightedRandomOpponent has the same attack order (when seeded) and attacks all lines"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            length = 30
            expected_attack_order = [
                4,
                3,
                14,
                15,
                12,
                15,
                12,
                3,
                12,
                13,
                3,
                15,
                4,
                14,
                15,
                13,
                14,
                4,
                3,
                3,
                4,
                14,
                15,
                12,
                15,
                13,
                4,
                14,
                12,
                3,
            ]

            attack_order = []
            has_disconnected_all = False
            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_attack_cooldown=1,  # only for testing
                opponent_attack_duration=1,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=WeightedRandomOpponent,
                kwargs_opponent={
                    "lines_attacked": LINES_ATTACKED,
                    "rho_normalization": RHO_NORMALIZATION,
                    "attack_period": 1,
                },
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                # Collect some attacks and check that they belong to the correct lines
                obs = env.reset()
                done = False
                assert env._oppSpace.budget == init_budget
                for i in range(length):
                    if done:
                        obs = env.reset()
                    pre_done = done
                    obs, reward, done, info = env.step(env.action_space())

                    attack = env._oppSpace.last_attack
                    if attack is None and not done:
                        # should only happen here if all attackable lines are already disconnected
                        # OR if there are a game over
                        assert np.sum(obs.line_status == False) == 6 or done
                        continue

                    assert any(attack._set_line_status == -1)
                    attacked_line = np.where(attack._set_line_status == -1)[0][0]
                    if pre_done or not (
                        attack_order and attack_order[-1] == attacked_line
                    ):
                        attack_order.append(attacked_line)
                assert attack_order == expected_attack_order
                assert len(set(attack_order)) == 6

    def test_either_attack_or_tell_attack_continues(self):
        """Tests that at each step, either the attack or the tell_attack_continues method is called once"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            length = 100
            attack_cooldown = 15
            with grid2op.make(
                "rte_case14_realistic",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_attack_cooldown=attack_cooldown,  # only for testing
                opponent_attack_duration=5,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=TestWeightedRandomOpponent,
                kwargs_opponent={
                    "lines_attacked": LINES_ATTACKED,
                    "rho_normalization": RHO_NORMALIZATION,
                    "attack_period": attack_cooldown,
                },
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                # Collect some attacks and check that they belong to the correct lines
                obs = env.reset()
                done = False
                assert env._oppSpace.budget == init_budget
                for i in range(length):
                    if done:
                        obs = env.reset()
                    obs, reward, done, info = env.step(env.action_space())
                assert env._oppSpace.opponent._attack_counter == 70
                assert env._oppSpace.opponent._attack_continues_counter == 30
                assert (
                    env._oppSpace.opponent._attack_counter
                    + env._oppSpace.opponent._attack_continues_counter
                    == length
                )


class TestGeometricOpponent(unittest.TestCase):
    def test_can_create(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__) as env:
                my_opp = GeometricOpponent(action_space=env.action_space)

    def test_can_init(self):
        init_budget = 120.0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "l2rpn_case14_sandbox",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_attack_cooldown=0,  # only for testing
                opponent_attack_duration=1,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=GeometricOpponent,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__,
            ) as env:
                env.seed(0)
                obs = env.reset()

    def test_does_attack_outsideenv(self):
        init_budget = 120.0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "l2rpn_case14_sandbox",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=0.0,
                opponent_attack_cooldown=0,  # only for testing
                opponent_attack_duration=1,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=GeometricOpponent,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                obs = env.reset()
                opponent = env._opponent
                assert np.all(opponent._attack_times == [64, 407, 487, 522])
                assert np.all(opponent._attack_waiting_times == [64, 312, 48, 4])
                assert np.all(opponent._attack_durations == [31, 32, 31, 25])
                assert np.all(opponent._number_of_attacks == 4)

                # now i simulate what happens in the "real" game up to the first attack, it should not attack !
                # 64 is hard coded here because i set the seed ! and it's an error if the seeding does not work
                for i in range(64):
                    attack, duration = opponent.attack(obs, None, None, None, None)
                    assert attack is None
                    assert duration is None

                # it should do an attack
                attack, duration = opponent.attack(obs, None, None, None, None)
                assert attack is not None
                assert duration == 31
                lines_impacted, subs_impacted = attack.get_topological_impact()
                assert np.sum(lines_impacted) == 1
                assert lines_impacted[4]

                # now the attack last 31 steps, so I "tell attack continues" for that long
                for i in range(31):
                    opponent.tell_attack_continues(obs, None, None, None)

                # now i have to wait for another 312 steps
                for i in range(312):
                    attack, duration = opponent.attack(obs, None, None, None, None)
                    assert attack is None, f"error for step {i}"
                    assert duration is None, f"error for step {i}"

                # it should do another attack
                attack, duration = opponent.attack(obs, None, None, None, None)
                assert attack is not None
                assert duration == 32
                lines_impacted, subs_impacted = attack.get_topological_impact()
                assert np.sum(lines_impacted) == 1
                assert lines_impacted[12]

                # now i reset it
                obs = env.reset()  # behaviour changed in 1.6.5
                assert np.all(opponent._attack_times == [189, 250, 351, 446])
                assert np.all(opponent._attack_waiting_times == [189, 18, 67, 66])
                assert np.all(opponent._attack_durations == [43, 34, 29, 49])
                assert np.all(opponent._number_of_attacks == 4)

                for i in range(189):
                    attack, duration = opponent.attack(obs, None, None, None, None)
                    assert attack is None
                    assert duration is None

                # it should do an attack
                attack, duration = opponent.attack(obs, None, None, None, None)
                assert attack is not None
                assert duration == 43
                lines_impacted, subs_impacted = attack.get_topological_impact()
                assert np.sum(lines_impacted) == 1
                assert lines_impacted[13]

    def test_does_attack(self):
        init_budget = 500
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "l2rpn_case14_sandbox",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=200.0,
                opponent_attack_cooldown=0,  # only for testing
                opponent_attack_duration=300,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=GeometricOpponent,
                param=param,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                obs = env.reset()
                opponent = env._opponent
                assert np.all(opponent._attack_times == [64, 407, 487, 522])
                assert np.all(opponent._attack_waiting_times == [64, 312, 48, 4])
                assert np.all(opponent._attack_durations == [31, 32, 31, 25])
                assert np.all(opponent._number_of_attacks == 4)
                # it should not attack before due time
                for i in range(64):
                    obs, reward, done, info = env.step(env.action_space())
                    assert (
                        info["opponent_attack_duration"] == 0
                    ), f"attack detected at iteration {i}"
                    assert (
                        info["opponent_attack_line"] is None
                    ), f"attack detected at iteration {i}"
                # now it should attack
                obs, reward, done, info = env.step(env.action_space())
                assert info["opponent_attack_duration"] == 31
                assert info["opponent_attack_line"][4]
                # here the attack continues
                for i in range(30):
                    obs, reward, done, info = env.step(env.action_space())
                    assert (
                        info["opponent_attack_duration"] == 30 - i
                    ), f"wrong attack duration at iteration {i}"
                    assert info["opponent_attack_line"][
                        4
                    ], f"wrong line attacked at iteration {i}"

                # I will NOT simulate the 312 steps where the opponent does not attack... I only do a few for speed
                for i in range(10):
                    obs, reward, done, info = env.step(env.action_space())
                    assert (
                        info["opponent_attack_duration"] == 0
                    ), f"attack detected at iteration {i}"
                    assert (
                        info["opponent_attack_line"] is None
                    ), f"attack detected at iteration {i}"

                # reset
                obs = env.reset()
                # NOTE this is not the same times as above... Indeed the sequence of prn generated is not the same
                # (because as opposed to test_does_attack_outsideenv, this time i don't simulate everything)
                assert np.all(
                    opponent._attack_times == [189, 250, 351, 446]
                )  # reset changed in grid2op 1.6.5
                assert np.all(opponent._attack_waiting_times == [189, 18, 67, 66])
                assert np.all(opponent._attack_durations == [43, 34, 29, 49])
                assert np.all(opponent._number_of_attacks == 4)
                # it should not attack before due time, but i don't simulate everything...
                for i in range(10):
                    obs, reward, done, info = env.step(env.action_space())
                    assert (
                        info["opponent_attack_duration"] == 0
                    ), f"attack detected at iteration {i}"
                    assert (
                        info["opponent_attack_line"] is None
                    ), f"attack detected at iteration {i}"

    def test_minimum_attack_duration(self):
        init_budget = 500
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "l2rpn_case14_sandbox",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=200.0,
                opponent_attack_cooldown=0,  # only for testing
                opponent_attack_duration=300,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=GeometricOpponent,
                param=param,
                kwargs_opponent={
                    "lines_attacked": LINES_ATTACKED,
                    "attack_every_xxx_hour": 24,
                    "average_attack_duration_hour": 5,
                    "minimum_attack_duration_hour": 4,
                },
                _add_to_name=type(self).__name__,
            ) as env:
                env.seed(0)
                obs = env.reset()
                opponent = env._opponent
                assert np.all(opponent._attack_durations >= 48)
                obs = env.reset()
                assert np.all(opponent._attack_durations >= 48)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "l2rpn_case14_sandbox",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=200.0,
                opponent_attack_cooldown=0,  # only for testing
                opponent_attack_duration=300,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=GeometricOpponent,
                param=param,
                kwargs_opponent={
                    "lines_attacked": LINES_ATTACKED,
                    "attack_every_xxx_hour": 24,
                    "average_attack_duration_hour": 5,
                    "minimum_attack_duration_hour": 1,
                },
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                obs = env.reset()
                opponent = env._opponent
                assert np.all(opponent._attack_durations >= 12)
                obs = env.reset()
                assert np.all(opponent._attack_durations >= 12)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "l2rpn_case14_sandbox",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=200.0,
                opponent_attack_cooldown=0,  # only for testing
                opponent_attack_duration=300,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=GeometricOpponent,
                param=param,
                kwargs_opponent={
                    "lines_attacked": LINES_ATTACKED,
                    "attack_every_xxx_hour": 50,
                    "average_attack_duration_hour": 31,
                    "minimum_attack_duration_hour": 30,
                },
                _add_to_name=type(self).__name__
            ) as env:
                env.seed(0)
                obs = env.reset()
                opponent = env._opponent
                assert np.all(opponent._attack_durations >= 30 * 12)
                obs = env.reset()
                assert np.all(opponent._attack_durations >= 30 * 12)

    def test_average_attack_duration(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "rte_case5_example", test=True, chronics_class=ChangeNothing,
                _add_to_name=type(self).__name__
            ) as env:
                my_opp = GeometricOpponent(action_space=env.action_space)
                with self.assertRaises(OpponentError):
                    # this is not supported, as there are an infinite number of steps in this environment
                    my_opp.init(
                        partial_env=env,
                        lines_attacked=env.name_line,
                        attack_every_xxx_hour=24,
                        average_attack_duration_hour=2,
                        minimum_attack_duration_hour=1,
                    )
                env.set_max_iter(3000000)
                threshold = 1.0  # balance between test speed and precision i ask to match the theoretical average
                for mean_duration_hour in [2, 4, 8, 12, 16, 20]:
                    my_opp.seed(0)
                    my_opp.init(
                        partial_env=env,
                        lines_attacked=env.name_line,
                        attack_every_xxx_hour=24,
                        average_attack_duration_hour=mean_duration_hour,
                        minimum_attack_duration_hour=1,
                    )
                    assert (
                        abs(np.mean(my_opp._attack_durations) - mean_duration_hour * 12)
                        < threshold
                    ), f"error for {mean_duration_hour}: {np.mean(my_opp._attack_durations):.2f}"

    def test_attack_every_xxx_hour(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "rte_case5_example", test=True, chronics_class=ChangeNothing,
                _add_to_name=type(self).__name__
            ) as env:
                my_opp = GeometricOpponent(action_space=env.action_space)
                n_ = 3_000_000
                env.set_max_iter(n_)
                threshold = 0.03  # balance between test speed and precision i ask to match the theoretical average
                average_attack_duration_hour = 2
                for mean_attack_every_xxx_hour in [12, 16, 20, 24, 48]:
                    my_opp.seed(1)
                    my_opp.init(
                        partial_env=env,
                        lines_attacked=env.name_line,
                        attack_every_xxx_hour=mean_attack_every_xxx_hour,
                        average_attack_duration_hour=average_attack_duration_hour,
                        minimum_attack_duration_hour=1,
                    )
                    std = np.sqrt(
                        (1 - my_opp._attack_hazard_rate)
                        / (my_opp._attack_hazard_rate**2)
                    )
                    duration_avg = np.mean(
                        my_opp._attack_waiting_times + my_opp._attack_durations
                    )
                    assert (
                        abs(duration_avg - mean_attack_every_xxx_hour * 12)
                        < threshold * std
                    ), f"error for {mean_attack_every_xxx_hour}: {duration_avg:.2f}"

    def test_cannot_init_with_wrong_param(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__) as env:
                my_opp = GeometricOpponent(action_space=env.action_space)

            with self.assertRaises(OpponentError):
                # i cannot do an attack every 19 hours on average when an attack last 21h on average
                my_opp.init(
                    partial_env=env,
                    lines_attacked=LINES_ATTACKED,
                    attack_every_xxx_hour=19,
                    average_attack_duration_hour=21,
                    minimum_attack_duration_hour=20,
                )

            with self.assertRaises(OpponentError):
                # i cannot do an attack that last 19 hours on average and a minimum of 20 hours
                my_opp.init(
                    partial_env=env,
                    lines_attacked=LINES_ATTACKED,
                    attack_every_xxx_hour=50,
                    average_attack_duration_hour=19,
                    minimum_attack_duration_hour=20,
                )

    def test_simulate(self):
        """test the opponent is working with the simulate function"""
        init_budget = 500
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        line_id = 4
        opponent_attack_duration = 31
        first_attack_ts = 64
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "l2rpn_case14_sandbox",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=200.0,
                opponent_attack_cooldown=0,  # only for testing
                opponent_attack_duration=300,  # only for testing
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=GeometricOpponent,
                param=param,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__,
            ) as env:
                reco_line = env.action_space({"set_line_status": [(line_id, 1)]})
                env.seed(0)
                obs = env.reset()
                opponent = env._opponent
                assert np.all(opponent._attack_times == [64, 407, 487, 522])
                assert np.all(opponent._attack_waiting_times == [64, 312, 48, 4])
                assert np.all(opponent._attack_durations == [31, 32, 31, 25])
                assert np.all(opponent._number_of_attacks == 4)

                # do steps just before the first attack
                for i in range(first_attack_ts):
                    obs, reward, done, info = env.step(env.action_space())

                # i can simulate anything and it should be working
                # opponent won't disconnect anything in simulate
                simobs, sim_r, sim_d, sim_info = obs.simulate(env.action_space())
                assert simobs.rho[line_id] > 0.0
                simobs, sim_r, sim_d, sim_info = obs.simulate(reco_line)
                assert simobs.rho[line_id] > 0.0

                # i do a step, powerline should be disconnected even if i reconnect it
                # => basically i check the attack has been performed
                obs, reward, done, info = env.step(reco_line)
                assert obs.rho[line_id] == 0.0
                assert not obs.line_status[line_id]

                # check that the line disconnected cannot be reconnected
                for i in range(opponent_attack_duration + 1):
                    simobs, sim_r, sim_d, sim_info = obs.simulate(reco_line)
                    assert simobs.rho[line_id] == 0.0
                    assert not simobs.line_status[line_id]

                # check that the opponent continue its attacks
                for i in range(opponent_attack_duration - 1):
                    obs, reward, done, info = env.step(reco_line)
                    assert obs.rho[line_id] == 0.0
                    assert not obs.line_status[line_id]

                # i should be able to simulate a reconnection now (attack is over)
                simobs, sim_r, sim_d, sim_info = obs.simulate(reco_line)
                assert simobs.rho[line_id] > 0.0
                assert simobs.line_status[line_id]
                # this should not affect the environment
                assert obs.rho[line_id] == 0.0
                assert not obs.line_status[line_id]

                # and now that i'm able to reconnect the powerline in the real environment
                obs, reward, done, info = env.step(reco_line)
                assert obs.rho[line_id] > 0.0
                assert obs.line_status[line_id]

    def test_last_attack(self):
        init_budget = 500
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(
                "l2rpn_case14_sandbox",
                test=True,
                opponent_init_budget=init_budget,
                opponent_budget_per_ts=200.0,
                opponent_attack_cooldown=0,  # only for testing
                opponent_attack_duration=30,  # max
                opponent_action_class=TopologyAction,
                opponent_budget_class=BaseActionBudget,
                opponent_class=GeometricOpponent,
                param=param,
                kwargs_opponent={"lines_attacked": LINES_ATTACKED},
                _add_to_name=type(self).__name__,
            ) as env:
                env.seed(0)
                _ = env.reset()
                # opponent = env._opponent
                # opponent._attack_durations : should be [31, 32, 31, 25]
                # opponent._attack_times : should be [64, 407, 487, 522]
                dn = env.action_space()
                for ts in range(522):
                    # here the opponent cannot attack due to the `opponent_attack_duration` that is too low
                    # chosen duration is below max_duration, so attack is not done.
                    obs, reward, done, info = env.step(dn)
                    assert info["opponent_attack_line"] is None

                # opponent should attack at this exact step
                obs, reward, done, info = env.step(dn)
                assert info["opponent_attack_line"] is not None


class TestChangeOppSpace(unittest.TestCase):
    """test i can change the opponent_space_type when creating an environment"""    
    def test_change_opp_space_type(self):
        class OpponentSpaceCust(OpponentSpace):
            pass
                
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_icaps_2021", test=True, _add_to_name=type(self).__name__)
            assert isinstance(env._oppSpace, OpponentSpace) 
            
            # check i can change it from "make"
            env = grid2op.make("l2rpn_icaps_2021", opponent_space_type=OpponentSpaceCust, test=True, _add_to_name=type(self).__name__)
            assert isinstance(env._oppSpace, OpponentSpaceCust) 
            # check it's properly propagated when copied
            env_cpy = env.copy()
            assert isinstance(env_cpy._oppSpace, OpponentSpaceCust)
            # check it's properly propagated in the kwargs
            env_params = env.get_kwargs()
            assert env_params["opponent_space_type"] == OpponentSpaceCust
            # check it's properly propagated in the runner
            runner_params = env.get_params_for_runner()
            assert runner_params["opponent_space_type"] == OpponentSpaceCust
            runner = Runner(**runner_params)
            assert runner._opponent_space_type == OpponentSpaceCust
            # check the runner can make an env with the right opponent space type
            env_runner = runner.init_env()
            assert isinstance(env_runner._oppSpace, OpponentSpaceCust)
            
            
if __name__ == "__main__":
    unittest.main()
