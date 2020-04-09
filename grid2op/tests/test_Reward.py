# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import sys
import unittest
import numpy as np
import pdb
import warnings
import numbers
from abc import ABC, abstractmethod
from grid2op.tests.helper_path_test import *
from grid2op.Reward import *
from grid2op import make


class TestLoadingReward(ABC):
    def setUp(self):
        self.env = make("case5_example", reward_class=self._reward_type())
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

class TestDistanceReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return DistanceReward

    def test_do_nothing(self):
        self.env.reset()
              
        dn_action = self.env.action_space({})

        obs, r, d, info = self.env.step(dn_action)
        max_reward = self.env.reward_helper.range()[1]
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

        set_action = self.env.action_space({"set_bus": {"lines_or_id": [(0,2)]}})

        obs, r, d, info = self.env.step(set_action)
        assert r != 1.0

class TestLoadingGameplayReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return GameplayReward

class TestCombinedReward(TestLoadingReward, unittest.TestCase):
    def _reward_type(self):
        return CombinedReward

    def test_add_reward(self):
        cr = self.env.reward_helper.template_reward
        assert cr is not None
        cr.addReward("Gameplay", GameplayReward(), 1.0)
        cr.addReward("Flat", FlatReward(), 1.0)
        cr.initialize(self.env)

    def test_remove_reward(self):
        cr = self.env.reward_helper.template_reward
        assert cr is not None
        added = cr.addReward("Gameplay", GameplayReward(), 1.0)
        assert added == True
        removed = cr.removeReward("Gameplay")
        assert removed == True
        removed = cr.removeReward("Unknow")
        assert removed == False

    def test_update_reward_weight(self):
        cr = self.env.reward_helper.template_reward
        assert cr is not None
        added = cr.addReward("Gameplay", GameplayReward(), 1.0)
        assert added == True
        updated = cr.updateRewardWeight("Gameplay", 0.5)
        assert updated == True
        updated = cr.updateRewardWeight("Unknow", 0.5)
        assert updated == False

    def test_combine_distance_gameplay(self):
        cr = self.env.reward_helper.template_reward
        assert cr is not None
        added = cr.addReward("Gameplay", GameplayReward(), 0.5)
        assert added == True
        distance_reward = DistanceReward()
        added = cr.addReward("Distance", distance_reward, 0.5)
        assert added == True
        self.env.reset()
        cr.initialize(self.env)

        set_action = self.env.action_space({"set_bus": {"lines_or_id": [(1,2)]}})
        obs, r, d, info = self.env.step(set_action)
        assert r < 1.0 
        
if __name__ == "__main__":
    unittest.main()
