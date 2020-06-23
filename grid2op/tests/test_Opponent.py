# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import tempfile
import warnings
from grid2op.tests.helper_path_test import *
from grid2op.Opponent import BaseOpponent, RandomLineOpponent
from grid2op.Action import TopologyAction
from grid2op.MakeEnv import make
from grid2op.Opponent.BaseActionBudget import BaseActionBudget
from grid2op.dtypes import dt_int
from grid2op.Parameters import Parameters
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData
from grid2op.Environment import SingleEnvMultiProcess
from grid2op.Agent import BaseAgent
import pdb

ATTACK_DURATION = 48
ATTACK_COOLDOWN = 100


class TestSuiteBudget_001(BaseActionBudget):
    """just for testing"""
    pass


class TestSuiteOpponent_001(BaseOpponent):
    """test class that disconnects randomly the powerlines"""
    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self.line_id = [0, 1, 2, 3]
        self.possible_attack = [self.action_space.disconnect_powerline(line_id=el) for el in self.line_id]

    def attack(self, observation, agent_action, env_action, budget, previous_fails):
        if observation is None:  # On first step
            return None
        attack = self.space_prng.choice(self.possible_attack)
        return attack


class ReconnectAgent(BaseAgent):
    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space)
        self.rotate_counter = 0

    def act(self, observation, reward, done=False):
        if np.all(observation.line_status):
            res = self.action_space({})
        else:
            dc_ids = np.argwhere(observation.line_status == False).ravel()
            line_id = dc_ids[self.rotate_counter % len(dc_ids)]
            res = self.action_space({'set_line_status': [(line_id, 1)]})
            self.rotate_counter += 1
        return res


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
        # and test reset, which apparently is NOT done correctly
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budg = 100.
            with make("rte_case5_example",
                      test=True,
                      opponent_init_budget=init_budg,
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=TestSuiteBudget_001,
                      opponent_attack_duration=ATTACK_DURATION,
                      opponent_attack_cooldown=ATTACK_COOLDOWN,
                      opponent_class=TestSuiteOpponent_001) as env:
                obs = env.reset()
                # opponent should not attack at the first time step
                assert np.all(obs.line_status)
                assert env.opponent_init_budget == init_budg
                obs, reward, done, info = env.step(env.action_space())
                assert env.oppSpace.budget == init_budg - 1.0

                obs = env.reset()
                # opponent should not attack at the first time step
                assert np.all(obs.line_status)
                assert env.opponent_init_budget == init_budg
                assert env.oppSpace.budget == init_budg

    def test_env_opp_attack_budget_ts(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budg_ts = 0.5
            with make("rte_case5_example",
                      test=True,
                      opponent_budget_per_ts=init_budg_ts,
                      opponent_attack_duration=1,  # only for testing
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=TestSuiteBudget_001,
                      opponent_attack_cooldown=ATTACK_COOLDOWN,
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
        """Tests that the attack is ignored when the budget is too low"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 50
            with make("rte_case14_realistic",
                      test=True,
                      opponent_attack_cooldown=0,  # only for testing
                      opponent_init_budget=init_budget,
                      opponent_budget_per_ts=0.,
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=BaseActionBudget,
                      opponent_attack_duration=ATTACK_DURATION,
                      opponent_class=RandomLineOpponent) as env:
                env.seed(0)
                obs = env.reset()
                assert env.oppSpace.budget == init_budget
                # The opponent can attack
                for i in range(env.oppSpace.attack_duration):
                    obs, reward, done, info = env.step(env.action_space())
                    attack = env.oppSpace.last_attack
                    assert env.oppSpace.budget == init_budget - i - 1
                    assert any(attack._set_line_status != 0)

                # There is not enough budget for a second attack
                assert abs(env.oppSpace.budget - (init_budget - ATTACK_DURATION)) <= 1e-5
                obs, reward, done, info = env.step(env.action_space())
                attack = env.oppSpace.last_attack
                assert attack is None

    def test_RandomLineOpponent_attackable_lines(self):
        """Tests that the RandomLineOpponent only attacks the authorized lines"""
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
                      opponent_attack_duration=ATTACK_DURATION,
                      opponent_attack_cooldown=ATTACK_COOLDOWN,
                      opponent_class=RandomLineOpponent) as env:
                env.seed(0)
                # Collect some attacks and check that they belong to the correct lines
                for _ in range(tries):
                    obs = env.reset()
                    assert env.oppSpace.budget == init_budget
                    obs, reward, done, info = env.step(env.action_space())
                    assert env.oppSpace.budget == init_budget - 1

                    attack = env.oppSpace.last_attack
                    attacked_line = np.where(attack._set_line_status == -1)[0][0]
                    line_name = env.action_space.name_line[attacked_line]
                    assert line_name in attackable_lines_case14

    def test_RandomLineOpponent_disconnects_only_one_line(self):
        """Tests that the RandomLineOpponent does not disconnect several lines at a time"""
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
                      opponent_attack_duration=ATTACK_DURATION,
                      opponent_attack_cooldown=ATTACK_COOLDOWN,
                      opponent_class=RandomLineOpponent) as env:
                env.seed(0)
                # Collect some attacks and check that they belong to the correct lines
                for _ in range(tries):
                    obs = env.reset()
                    assert env.oppSpace.budget == init_budget
                    obs, reward, done, info = env.step(env.action_space())
                    assert env.oppSpace.budget == init_budget - 1

                    attack = env.oppSpace.last_attack
                    n_disconnected = np.sum(attack._set_line_status == -1)
                    assert n_disconnected == 1

    def test_RandomLineOpponent_no_overflow(self):
        """Tests that the line status cooldown is correctly updated when the opponent attacks a line"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            length = 300
            agent_line_cooldown = 15
            param = Parameters()
            param.NO_OVERFLOW_DISCONNECTION = True
            param.NB_TIMESTEP_COOLDOWN_LINE = agent_line_cooldown
            with make("rte_case14_realistic",
                        test=True,
                        param=param,
                        opponent_attack_cooldown=0, # only for testing
                        opponent_attack_duration=10, # only for testing
                        opponent_init_budget=init_budget,
                        opponent_action_class=TopologyAction,
                        opponent_budget_class=BaseActionBudget,
                        opponent_class=RandomLineOpponent) as env:
                agent = ReconnectAgent(env.action_space)
                env.seed(0)
                obs = env.reset()
                reward = 0
                assert env.oppSpace.budget == init_budget
                assert np.all(obs.time_before_cooldown_line == 0)
                # Collect some attacks and check that they belong to the correct lines
                for i in range(length):
                    pre_cooldown = obs.time_before_cooldown_line.copy()
                    agent_action = agent.act(obs, reward)
                    obs, reward, done, info = env.step(agent_action)
                    assert env.oppSpace.budget == init_budget - i - 1
    
                    status_actionable = np.maximum(0, pre_cooldown - 1)

                    # Add attack cooldown
                    attack = env.oppSpace.last_attack
                    attacked_line = np.where(attack._set_line_status == -1)[0][0]
                    status_actionable[attacked_line] = env.oppSpace.current_attack_duration - 1

                    # Add agent cooldown
                    if any(agent_action._set_line_status == 1):
                        reco_line = np.where(agent_action._set_line_status == 1)[0][0]
                        if pre_cooldown[reco_line] == 0:
                            status_actionable[reco_line] = agent_line_cooldown
    
                    ## Add maintenance? Where to find?
    
                    assert np.all(obs.time_before_cooldown_line == status_actionable)

    def test_RandomLineOpponent_only_attack_connected(self):
        """
        Tests that the RandomLineOpponent does not attack lines that are already disconnected
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 10000
            length = 300
            env = make("rte_case14_realistic",
                       test=True,
                       opponent_init_budget=init_budget,
                       opponent_budget_per_ts=0.,
                       opponent_attack_cooldown=0, # only for testing
                       opponent_action_class=TopologyAction,
                       opponent_budget_class=BaseActionBudget,
                       opponent_attack_duration=ATTACK_DURATION,
                       opponent_class=RandomLineOpponent)
            env.seed(0)
            # Collect some attacks
            # and check that they belong to the correct lines
            pre_obs = env.reset()
            done = False
            assert env.oppSpace.budget == init_budget
            for i in range(length):
                obs, reward, done, info = env.step(env.action_space())
                    
                attack = env.oppSpace.last_attack
                attacked_line = np.where(attack._set_line_status == -1)[0][0]
                if env.oppSpace.current_attack_duration < env.oppSpace.attack_duration:
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
            expected_attack_order = [
                4, 12, 14, 3,
                3, 15, 14, 14,
                12, 15, 4, 15,
                13, 12, 14, 12,
                3, 12, 15, 14,
                15, 4, 3, 14,
                12, 13, 4, 15,
                3, 13
            ]

            attack_order = []
            has_disconnected_all = False
            with make("rte_case14_realistic",
                      test=True,
                      opponent_init_budget=init_budget,
                      opponent_budget_per_ts=0.,
                      opponent_attack_cooldown=0,  # only for testing
                      opponent_attack_duration=1,  # only for testing
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=BaseActionBudget,
                      opponent_class=RandomLineOpponent) as env:
                env.seed(0)
                # Collect some attacks and check that they belong to the correct lines
                obs = env.reset()
                done = False
                assert env.oppSpace.budget == init_budget
                for i in range(length):
                    if done:
                        obs = env.reset()
                    pre_done = done
                    obs, reward, done, info = env.step(env.action_space())

                    attack = env.oppSpace.last_attack
                    if attack is None: # should only happen here if all attackable lines are already disconnected
                        assert np.sum(obs.line_status == False) == 6
                        continue

                    assert any(attack._set_line_status == -1)
                    attacked_line = np.where(attack._set_line_status == -1)[0][0]
                    if pre_done or not (attack_order and attack_order[-1] == attacked_line):
                        attack_order.append(attacked_line)

                assert attack_order == expected_attack_order
                assert len(set(attack_order)) == 6

    def test_simulate(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            opponent_attack_duration = 15
            opponent_attack_cooldown = 20
            line_id = 4
            with make("rte_case14_realistic",
                      test=True,
                      opponent_init_budget=init_budget,
                      opponent_budget_per_ts=0.,
                      opponent_attack_cooldown=opponent_attack_cooldown,
                      opponent_attack_duration=opponent_attack_duration,
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=BaseActionBudget,
                      opponent_class=RandomLineOpponent) as env:
                env.seed(0)
                reco_line = env.action_space({"set_line_status": [(line_id, 1)]})

                obs = env.reset()
                obs, reward, done, info = env.step(env.action_space())
                assert obs.rho[line_id] == 0.
                assert not obs.line_status[line_id]
                simobs, sim_r, sim_d, sim_info = obs.simulate(env.action_space())
                assert simobs.rho[line_id] == 0.
                assert not simobs.line_status[line_id]
                simobs, sim_r, sim_d, sim_info = obs.simulate(reco_line)
                assert simobs.rho[line_id] == 0.
                assert not simobs.line_status[line_id]
                obs, reward, done, info = env.step(reco_line)
                assert obs.rho[line_id] == 0.
                assert not obs.line_status[line_id]

                # check that the budget of the opponent in the ObsEnv does not decrease
                for i in range(opponent_attack_duration):
                    simobs, sim_r, sim_d, sim_info = obs.simulate(reco_line)
                    assert simobs.rho[line_id] == 0.
                    assert not simobs.line_status[line_id]

                # check that the opponent continue its attacks
                for i in range(opponent_attack_duration - 2):
                    obs, reward, done, info = env.step(reco_line)
                    assert obs.rho[line_id] == 0.
                    assert not obs.line_status[line_id]

                # i should be able to simulate a reconnection now
                simobs, sim_r, sim_d, sim_info = obs.simulate(reco_line)
                assert simobs.rho[line_id] > 0.
                assert simobs.line_status[line_id]
                # this should not affect the environment
                assert obs.rho[line_id] == 0.
                assert not obs.line_status[line_id]

                # and now that i'm able to reconnect the powerline in step
                obs, reward, done, info = env.step(reco_line)
                assert obs.rho[line_id] > 0.
                assert obs.line_status[line_id]

    def test_opponent_load(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example",
                      test=True,
                      opponent_action_class=TopologyAction,
                      opponent_class=RandomLineOpponent) as env_1:
                env_1.seed(0)
                obs, reward, done, info = env_1.step(env_1.action_space())
            with make("rte_case118_example",
                      test=True,
                      opponent_action_class=TopologyAction,
                      opponent_class=RandomLineOpponent) as env_2:
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

            with make("rte_case14_realistic",
                      test=True,
                      opponent_init_budget=init_budget,
                      opponent_budget_per_ts=0.,
                      opponent_attack_cooldown=opponent_attack_cooldown,
                      opponent_attack_duration=opponent_attack_duration,
                      opponent_action_class=opponent_action_class,
                      opponent_budget_class=BaseActionBudget,
                      opponent_class=RandomLineOpponent) as env:
                env.seed(0)
                assert env.opponent_action_class == opponent_action_class
                assert issubclass(env.oppSpace.action_space.actionClass, opponent_action_class)
                assert issubclass(env.opponent_action_space.actionClass, opponent_action_class)
                opp_space = env.oppSpace
                attack, duration = opp_space.attack(env.get_obs(), env.action_space(), env.action_space())
                assert isinstance(attack, opponent_action_class)


    def test_get_set_state(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            opponent_attack_duration = 15
            opponent_attack_cooldown = 20
            line_id = 4

            with make("rte_case14_realistic",
                      test=True,
                      opponent_init_budget=init_budget,
                      opponent_budget_per_ts=0.,
                      opponent_attack_cooldown=opponent_attack_cooldown,
                      opponent_attack_duration=opponent_attack_duration,
                      opponent_action_class=TopologyAction,
                      opponent_budget_class=BaseActionBudget,
                      opponent_class=RandomLineOpponent) as env:
                env.seed(0)
                agent_action = env.action_space()
                observation = env.get_obs()
                env_action = env.action_space()

                opp_space = env.oppSpace
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
                assert np.all(attack1[1] == attack2[1]+1)

    def test_withrunner(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_budget = 1000
            opponent_attack_duration = 15
            opponent_attack_cooldown = 30
            opponent_budget_per_ts = 0.
            opponent_action_class = TopologyAction
            line_id = 3

            p = Parameters()
            p.NO_OVERFLOW_DISCONNECTION = True
            with make("rte_case14_realistic",
                      test=True, param=p,
                      opponent_init_budget=init_budget,
                      opponent_budget_per_ts=opponent_budget_per_ts,
                      opponent_attack_cooldown=opponent_attack_cooldown,
                      opponent_attack_duration=opponent_attack_duration,
                      opponent_action_class=opponent_action_class,
                      opponent_budget_class=BaseActionBudget,
                      opponent_class=RandomLineOpponent) as env:
                env.seed(0)
                runner = Runner(**env.get_params_for_runner())
                assert runner.opponent_init_budget == init_budget
                assert runner.opponent_budget_per_ts == opponent_budget_per_ts
                assert runner.opponent_attack_cooldown == opponent_attack_cooldown
                assert runner.opponent_attack_duration == opponent_attack_duration
                assert runner.opponent_action_class == opponent_action_class

                res = runner.run(nb_episode=1,
                                 max_iter=opponent_attack_cooldown,
                                 env_seeds=[0], agent_seeds=[0])
                f = tempfile.mkdtemp()
                res = runner.run(nb_episode=1, max_iter=opponent_attack_cooldown, path_save=f)
                for i, episode_name, cum_reward, timestep, total_ts in res:
                    episode_data = EpisodeData.from_disk(agent_path=f, name=episode_name)
                    assert np.any(episode_data.attack[:, line_id] == -1.), "no attack on powerline {}".format(line_id)
                    assert np.sum(episode_data.attack[:, line_id]) == -opponent_attack_duration, "too much / not enought attack on powerline {}".format(line_id)
                    assert np.all(episode_data.attack[:, 0] == 0.)

    def test_env_opponent(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("rte_case14_opponent", test=True, param=param)
        env.seed(0)  # make sure i have reproducible experiments
        obs = env.reset()
        assert env.oppSpace.budget == 0
        assert np.all(obs.line_status)
        obs, reward, done, info = env.step(env.action_space())
        assert env.oppSpace.budget == 0.5
        assert np.all(obs.line_status)
        obs, reward, done, info = env.step(env.action_space())

        env.close()

    def test_multienv_opponent(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("rte_case14_opponent", test=True, param=param)
        env.seed(0)  # make sure i have reproducible experiments
        multi_env = SingleEnvMultiProcess(env=env, nb_env=2)
        obs = multi_env.reset()
        for ob in obs:
            assert np.all(ob.line_status)
        env.close()
        multi_env.close()


if __name__ == "__main__":
    unittest.main()
