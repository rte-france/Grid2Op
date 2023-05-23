# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import pdb
import warnings
import numbers
from abc import ABC, abstractmethod

import grid2op
from grid2op.tests.helper_path_test import *
from grid2op.Reward import *
from grid2op.MakeEnv import make
from grid2op.Parameters import Parameters
from grid2op.Runner import Runner
from grid2op.Agent import BaseAgent

import warnings


class TestLoadingReward(ABC):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make(
                "rte_case5_example", test=True, reward_class=self._reward_type()
            )

        self.action = self.env.action_space()
        self.has_error = False
        self.is_done = False
        self.is_illegal = False
        self.is_ambiguous = False

    def tearDown(self):
        self.env.close()

    @abstractmethod
    def _reward_type(self):
        pass

    def test_reward(self):
        _, r_, _, _ = self.env.step(self.action)
        assert isinstance(r_, numbers.Number)
        assert issubclass(self._reward_type(), BaseReward)


class TestLoadingConstantReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return ConstantReward


class TestLoadingEconomicReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return EconomicReward


class TestLoadingFlatReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return FlatReward


class TestLoadingL2RPNReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return L2RPNReward


class TestLoadingRedispReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return RedispReward


class TestLoadingBridgeReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return BridgeReward


class TestLoadingL2RPNSandBoxScore(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return L2RPNSandBoxScore


class TestLoadingLinesCapacityReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return LinesCapacityReward


class TestDistanceReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return DistanceReward

    def test_do_nothing(self):
        self.env.reset()

        dn_action = self.env.action_space({})

        obs, r, d, info = self.env.step(dn_action)
        max_reward = self.env._reward_helper.range()[1]
        assert r == max_reward

    def test_disconnect(self):
        self.env.reset()

        set_status = self.env.action_space.get_set_line_status_vect()
        set_status[1] = -1
        disconnect_action = self.env.action_space({"set_line_status": set_status})

        obs, r, d, info = self.env.step(disconnect_action)
        assert r < 1.0

    def test_setBus2(self):
        self.env.reset()

        set_action = self.env.action_space({"set_bus": {"lines_or_id": [(0, 2)]}})

        obs, r, d, info = self.env.step(set_action)
        assert r != 1.0


class TestLoadingGameplayReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return GameplayReward


class TestCombinedReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return CombinedReward

    def test_add_reward(self):
        cr = self.env._reward_helper.template_reward
        assert cr is not None
        cr.addReward("Gameplay", GameplayReward(), 1.0)
        cr.addReward("Flat", FlatReward(), 1.0)
        cr.initialize(self.env)

    def test_remove_reward(self):
        cr = self.env._reward_helper.template_reward
        assert cr is not None
        added = cr.addReward("Gameplay", GameplayReward(), 1.0)
        assert added == True
        removed = cr.removeReward("Gameplay")
        assert removed == True
        removed = cr.removeReward("Unknow")
        assert removed == False

    def test_update_reward_weight(self):
        cr = self.env._reward_helper.template_reward
        assert cr is not None
        added = cr.addReward("Gameplay", GameplayReward(), 1.0)
        assert added == True
        updated = cr.updateRewardWeight("Gameplay", 0.5)
        assert updated == True
        updated = cr.updateRewardWeight("Unknow", 0.5)
        assert updated == False

    def test_combine_distance_gameplay(self):
        cr = self.env._reward_helper.template_reward
        assert cr is not None
        added = cr.addReward("Gameplay", GameplayReward(), 0.5)
        assert added == True
        distance_reward = DistanceReward()
        added = cr.addReward("Distance", distance_reward, 0.5)
        assert added == True
        self.env.reset()
        cr.initialize(self.env)

        set_action = self.env.action_space({"set_bus": {"lines_or_id": [(1, 2)]}})
        obs, r, d, info = self.env.step(set_action)
        assert r < 1.0

    def test_combine_simulate(self):
        cr = self.env._reward_helper.template_reward
        assert cr is not None
        gr = GameplayReward()
        gr.set_range(-21.0, 21.0)
        added = cr.addReward("Gameplay", gr, 2.0)
        assert added is True

        self.env.change_reward(cr)
        obs = self.env.reset()
        assert self.env.reward_range == (-42, 42)
        _, reward, done, info = obs.simulate(self.env.action_space({}))
        assert done is False
        assert reward == 42.0


class TestIncreaseFlatReward(unittest.TestCase):
    def test_ok(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make(
                "l2rpn_case14_sandbox", reward_class=IncreasingFlatReward, test=True
            )

        assert env.nb_time_step == 0
        obs, reward, done, info = env.step(env.action_space())
        assert env.nb_time_step == 1
        assert reward == 1
        obs, reward, done, info = env.step(env.action_space())
        assert env.nb_time_step == 2
        assert reward == 2
        obs = env.reset()
        assert env.nb_time_step == 0
        obs, reward, done, info = env.step(env.action_space())
        assert env.nb_time_step == 1
        assert reward == 1


class TestEpisodeDurationReward(unittest.TestCase):
    def test_ok(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make(
                "l2rpn_case14_sandbox", reward_class=EpisodeDurationReward, test=True
            )

        assert env.nb_time_step == 0
        obs, reward, done, info = env.step(env.action_space())
        assert env.nb_time_step == 1
        assert reward == 0

        obs, reward, done, info = env.step(env.action_space())
        assert env.nb_time_step == 2
        assert reward == 0

        obs, reward, done, info = env.step(
            env.action_space({"set_bus": {"generators_id": [(0, -1)]}})
        )
        assert done
        assert env.nb_time_step == 3
        assert reward == 3.0 / 575.0

        obs = env.reset()
        assert env.nb_time_step == 0
        obs, reward, done, info = env.step(env.action_space())
        assert env.nb_time_step == 1
        assert reward == 0

        env.fast_forward_chronics(573)
        obs, reward, done, info = env.step(env.action_space())
        assert done
        assert env.nb_time_step == 575
        assert reward == 1.0


class TestN1Reward(unittest.TestCase):
    def test_ok(self):
        L_ID = 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make(
                "l2rpn_case14_sandbox", reward_class=N1Reward(l_id=L_ID), test=True
            )

        obs = env.reset()
        obs, reward, *_ = env.step(env.action_space())
        # obs._obs_env._reward_helper.template_reward._DEBUG = True
        obs_n1, *_ = obs.simulate(
            env.action_space({"set_line_status": [(L_ID, -1)]}), time_step=0
        )
        assert obs_n1.rho[L_ID] == 0  # line should have been disconnected
        assert (
            abs(reward - obs_n1.rho.max()) <= 1e-5
        ), "the correct reward has not been computed"
        env.close()

        L_IDS = [0, 1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "l2rpn_case14_sandbox",
                other_rewards={f"line_{l_id}": N1Reward(l_id=l_id) for l_id in L_IDS},
                test=True,
            )
        obs, reward, done, info = env.step(env.action_space())
        for l_id in L_IDS:
            obs_n1, *_ = obs.simulate(
                env.action_space({"set_line_status": [(l_id, -1)]}), time_step=0
            )
            assert (
                abs(info["rewards"][f"line_{l_id}"] - obs_n1.rho.max()) <= 1e-5
            ), f"the correct reward has not been computed for line {l_id}"
        env.close()


class TMPRewardForTest(BaseReward):
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            assert not has_error
        return super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)


class ErrorAgent(BaseAgent):
    def act(self, observation, reward, done=False):
        if observation.current_step == 9:
            return self.action_space({"set_bus": {"loads_id": [(0, -1)]}})  # force a game over
        return super().act(observation, reward, done)
    
    
class TestEndOfEpisode(unittest.TestCase):
    """test the appropriate flags at the end of an episode"""
    def setUp(self) -> None:
        param = Parameters()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, reward_class=TMPRewardForTest)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_ok_end_of_episode(self):
        # done = False
        # i = 0
        # while not done:
        #     obs, reward, done, info = self.env.step(self.env.action_space())
        #     i += 1
        # assert i == 575, f"{i = } vs 575"
        # above passed and took more than 30s
        
        self.env.set_max_iter(10)
        # episode goes until the end, no error is raised
        self.env.reset()
        done = False
        i = 0
        while not done:
            obs, reward, done, info = self.env.step(self.env.action_space())
            i += 1
        assert i == 10, f"{i = } vs 10"
        
        # agent does a game over, the reward should raise an error
        self.env.reset()
        done = False
        i = 0
        while i <= 1:
            obs, reward, done, info = self.env.step(self.env.action_space())
            i += 1
        with self.assertRaises(AssertionError):
            obs, reward, done, info = self.env.step(self.env.action_space({"set_bus": {"loads_id": [(0, -1)]}}))
        
        # agent does a game over at last step, the reward should raise an error
        self.env.reset()
        done = False
        i = 0
        while i <= 8:
            obs, reward, done, info = self.env.step(self.env.action_space())
            i += 1
        with self.assertRaises(AssertionError):
            obs, reward, done, info = self.env.step(self.env.action_space({"set_bus": {"loads_id": [(0, -1)]}}))
    
    def test_runner(self):
        runner = Runner(**self.env.get_params_for_runner())
        res = runner.run(nb_episode=1, max_iter=10)
        assert res[0][3] == 10
        
        runner = Runner(**self.env.get_params_for_runner(),
                        agentClass=ErrorAgent)
        # error before last observation
        with self.assertRaises(AssertionError):
            res = runner.run(nb_episode=1, max_iter=11)
        # error just at last observation
        with self.assertRaises(AssertionError):
            res = runner.run(nb_episode=1, max_iter=10)
        # no error
        res = runner.run(nb_episode=1, max_iter=9)

if __name__ == "__main__":
    unittest.main()
