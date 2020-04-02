import os
import sys
import unittest
import numpy as np
import pdb
import warnings
import numbers
from abc import ABC, abstractmethod
from grid2op.tests.helper_path_test import *
from grid2op.Reward import BaseReward, ConstantReward, EconomicReward, FlatReward, L2RPNReward, RedispReward
from grid2op import make


class TestLoadingReward(ABC):
    def setUp(self):
        self.env = make("case5_example")
        self.action = self.env.action_space()
        self.has_error = False
        self.is_done = False
        self.is_illegal = False
        self.is_ambiguous = False

    @abstractmethod
    def _create_reward(self):
        pass

    def test_reward(self):
        reward = self._create_reward()
        reward.initialize(self.env)
        r_ = reward(self.action, self.env, self.is_done, self.is_ambiguous, self.is_illegal, self.is_ambiguous)
        assert isinstance(r_, numbers.Number)
        assert isinstance(reward, BaseReward)


class TestLoadingConstantReward(TestLoadingReward, unittest.TestCase):
    def _create_reward(self):
        return ConstantReward()


class TestLoadingEconomicReward(TestLoadingReward, unittest.TestCase):
    def _create_reward(self):
        return EconomicReward()


class TestLoadingFlatReward(TestLoadingReward, unittest.TestCase):
    def _create_reward(self):
        return FlatReward()


class TestLoadingL2RPNReward(TestLoadingReward, unittest.TestCase):
    def _create_reward(self):
        return L2RPNReward()


class TestLoadingRedispReward(TestLoadingReward, unittest.TestCase):
    def _create_reward(self):
        return RedispReward()


if __name__ == "__main__":
    unittest.main()