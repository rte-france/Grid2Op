# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings
import copy
from grid2op.Parameters import Parameters
from grid2op.Exceptions import (
    SimulateUsedTooMuchThisStep,
    SimulateUsedTooMuchThisEpisode,
)


class TestSimulateCount(unittest.TestCase):
    """
    This class tests the possibility in grid2op to limit the number of call to "obs.simulate"
    """

    def _aux_make_env(self, param=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if param is not None:
                env = grid2op.make("l2rpn_case14_sandbox", test=True, param=param, _add_to_name=type(self).__name__)
            else:
                env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        return env

    def test_simple_cases(self):
        env = self._aux_make_env()
        obs = env.reset()
        # basic test
        obs.simulate(env.action_space())
        assert env.observation_space.nb_simulate_called_this_step == 1

        obs.simulate(env.action_space())
        obs.simulate(env.action_space())
        assert env.observation_space.nb_simulate_called_this_step == 3

        obs = env.reset()
        assert env.observation_space.nb_simulate_called_this_step == 0

    def test_with_copies(self):
        env = self._aux_make_env()
        # test with copies
        env_cpy = env.copy()
        obs_cpy = env_cpy.reset()
        assert env_cpy.observation_space.nb_simulate_called_this_step == 0

        obs = env.reset()
        obs.simulate(env.action_space())
        assert env.observation_space.nb_simulate_called_this_step == 1
        assert env_cpy.observation_space.nb_simulate_called_this_step == 0

        obs_cpy.simulate(env.action_space())
        assert env.observation_space.nb_simulate_called_this_step == 1
        assert env_cpy.observation_space.nb_simulate_called_this_step == 1

        obs_cpy.simulate(env.action_space())
        assert env.observation_space.nb_simulate_called_this_step == 1
        assert env_cpy.observation_space.nb_simulate_called_this_step == 2

        obs_cpy = env_cpy.reset()
        assert env.observation_space.nb_simulate_called_this_step == 1
        assert env_cpy.observation_space.nb_simulate_called_this_step == 0

    def test_max_step(self):
        MAX_SIMULATE_PER_STEP = 10
        param = Parameters()
        param.MAX_SIMULATE_PER_STEP = MAX_SIMULATE_PER_STEP
        env = self._aux_make_env(param)
        obs = env.reset()
        for i in range(MAX_SIMULATE_PER_STEP):
            obs.simulate(env.action_space())
        with self.assertRaises(SimulateUsedTooMuchThisStep):
            obs.simulate(env.action_space())  # raises a SimulateUsedTooMuchThisStep

        # should be OK now
        obs, *_ = env.step(env.action_space())
        obs.simulate(env.action_space())

    def test_max_episode(self):
        MAX_SIMULATE_PER_EPISODE = 10
        param = Parameters()
        param.MAX_SIMULATE_PER_EPISODE = MAX_SIMULATE_PER_EPISODE
        env = self._aux_make_env(param)
        obs = env.reset()
        for i in range(MAX_SIMULATE_PER_EPISODE):
            obs.simulate(env.action_space())
            obs, *_ = env.step(env.action_space())

        with self.assertRaises(SimulateUsedTooMuchThisEpisode):
            obs.simulate(env.action_space())  # raises a SimulateUsedTooMuchThisEpisode

        obs = env.reset()
        for i in range(MAX_SIMULATE_PER_EPISODE):
            obs.simulate(env.action_space())  # should work now (reset called)
        obs, *_ = env.step(env.action_space())

        with self.assertRaises(SimulateUsedTooMuchThisEpisode):
            obs.simulate(env.action_space())  # raises a SimulateUsedTooMuchThisEpisode

        obs = env.reset()
        obs.simulate(env.action_space())

    def test_max_step_with_copy(self):
        MAX_SIMULATE_PER_STEP = 10
        MAX_SIMULATE_PER_STEP_CPY = 5
        param = Parameters()
        param.MAX_SIMULATE_PER_STEP = MAX_SIMULATE_PER_STEP
        env = self._aux_make_env(param)

        param = copy.deepcopy(param)
        param.MAX_SIMULATE_PER_STEP = MAX_SIMULATE_PER_STEP_CPY
        env_cpy = env.copy()
        env_cpy.change_parameters(param)
        obs = env.reset()
        obs_cpy = env_cpy.reset()
        for i in range(MAX_SIMULATE_PER_STEP):
            obs.simulate(env.action_space())
        with self.assertRaises(SimulateUsedTooMuchThisStep):
            obs.simulate(env.action_space())  # raises a SimulateUsedTooMuchThisStep

        for i in range(MAX_SIMULATE_PER_STEP_CPY):
            obs_cpy.simulate(env.action_space())  # should work

        with self.assertRaises(SimulateUsedTooMuchThisStep):
            obs_cpy.simulate(env.action_space())  # raises a SimulateUsedTooMuchThisStep

        # should be OK now
        obs, *_ = env.step(env.action_space())
        obs.simulate(env.action_space())  # I can simulate on the original env correctly
        with self.assertRaises(SimulateUsedTooMuchThisStep):
            obs_cpy.simulate(env.action_space())  # raises a SimulateUsedTooMuchThisStep

    def test_max_episode_with_copy(self):
        MAX_SIMULATE_PER_EPISODE = 10
        MAX_SIMULATE_PER_EPISODE_CPY = 10
        param = Parameters()
        param.MAX_SIMULATE_PER_EPISODE = MAX_SIMULATE_PER_EPISODE
        env = self._aux_make_env(param)
        param = copy.deepcopy(param)
        param.MAX_SIMULATE_PER_EPISODE = MAX_SIMULATE_PER_EPISODE_CPY
        env_cpy = env.copy()
        env_cpy.change_parameters(param)
        obs = env.reset()
        obs_cpy = env_cpy.reset()

        for i in range(MAX_SIMULATE_PER_EPISODE):
            obs.simulate(env.action_space())
            obs, *_ = env.step(env.action_space())
        with self.assertRaises(SimulateUsedTooMuchThisEpisode):
            obs.simulate(env.action_space())  # raises a SimulateUsedTooMuchThisEpisode

        for i in range(MAX_SIMULATE_PER_EPISODE_CPY):
            obs_cpy.simulate(env.action_space())  # should not raise
            obs_cpy, *_ = env_cpy.step(env.action_space())
        with self.assertRaises(SimulateUsedTooMuchThisEpisode):
            obs_cpy.simulate(
                env.action_space()
            )  # raises a SimulateUsedTooMuchThisEpisode

        obs = env.reset()
        for i in range(MAX_SIMULATE_PER_EPISODE):
            obs.simulate(env.action_space())  # should work now (reset called)
        with self.assertRaises(SimulateUsedTooMuchThisEpisode):
            obs_cpy.simulate(
                env.action_space()
            )  # raises a SimulateUsedTooMuchThisEpisode (copy not reset)

    def test_no_limit(self):
        MAX_SIMULATE_PER_EPISODE = 7
        env = self._aux_make_env()
        obs = env.reset()
        for _ in range(MAX_SIMULATE_PER_EPISODE + 1):
            obs.simulate(env.action_space())

        # change parameters and see if the limit works
        param = Parameters()
        param.MAX_SIMULATE_PER_EPISODE = MAX_SIMULATE_PER_EPISODE
        env.change_parameters(param)
        obs = env.reset()
        for _ in range(MAX_SIMULATE_PER_EPISODE):
            obs.simulate(env.action_space())

        with self.assertRaises(SimulateUsedTooMuchThisEpisode):
            obs.simulate(
                env.action_space()
            )  # raises a SimulateUsedTooMuchThisEpisode (copy not reset)


if __name__ == "__main__":
    unittest.main()
