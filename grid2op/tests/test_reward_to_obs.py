# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import warnings
import unittest
from grid2op.Observation import CompleteObservation
from grid2op.Reward import BaseReward
from grid2op.Runner import Runner


class CustomTestReward(BaseReward):
    nb_call = 0
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        env._reward_to_obs[env.nb_time_step] = type(self).nb_call
        type(self).nb_call += 1
        return super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)


class CustomTestObservation(CompleteObservation):
    def update_after_reward(self, env):
        self.stuff = env._reward_to_obs[env.nb_time_step]
        return super().update_after_reward(env)
    
    
class BaseTestPlot(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True,
                                    observation_class=CustomTestObservation,
                                    reward_class=CustomTestReward)
        CustomTestReward.nb_call = 0

    def tearDown(self):
        self.env.close()

    def test_info(self):
        obs = self.env.reset()
        assert not hasattr(obs, "stuff")
        assert CustomTestReward.nb_call == 1  # ideally it should be 0 but..
        
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert hasattr(obs, "stuff")
        assert obs.stuff == 1
        assert CustomTestReward.nb_call == 2
        
    def test_copy(self):
        obs = self.env.reset()
        assert not hasattr(obs, "stuff")
        assert CustomTestReward.nb_call == 1  # ideally it should be 0 but..
        
        env_cpy = self.env.copy()
        obs, reward, done, info = env_cpy.step(self.env.action_space())
        assert hasattr(obs, "stuff")
        assert obs.stuff == 1
        assert CustomTestReward.nb_call == 2, f"{CustomTestReward.nb_call} vs 2"
        
        # attr is not copied as it is a class attribute !
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert hasattr(obs, "stuff")
        assert obs.stuff == 2
        assert CustomTestReward.nb_call == 3, f"{CustomTestReward.nb_call} vs 3"
        
    def test_runner(self):
        runner = Runner(**self.env.get_params_for_runner())
        CustomTestReward.nb_call = 0
        runner.run(nb_episode=1, max_iter=10)
        assert CustomTestReward.nb_call == 12, f"{CustomTestReward.nb_call} vs 12"
        
        
if __name__ == "__main__":
    unittest.main()
