# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import gym
from grid2op.gym_compat.gym_obs_space import GymObservationSpace
from grid2op.gym_compat.gym_act_space import GymActionSpace


class GymEnv(gym.Env):
    """
    fully implements the openAI gym API by using the :class:`GymActionSpace` and :class:`GymObservationSpace`
    for compliance with openAI gym.

    They can handle action_space_converter or observation_space converter to change the representation of data
    that will be fed to the agent.  #TODO

    Notes
    ------
    The environment passed as input is copied. It is not modified by this "gym environment"

    """
    def __init__(self, env_init):
        self._init_env = env_init.copy()
        self.action_space = GymActionSpace(self._init_env)
        self.observation_space = GymObservationSpace(self._init_env)
        self.reward_range = self._init_env.reward_range
        self.metadata = self._init_env.metadata

    def step(self, gym_action):
        g2op_act = self.action_space.from_gym(gym_action)
        g2op_obs, reward, done, info = self._init_env.step(g2op_act)
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, reward, done, info

    def reset(self):
        g2op_obs = self._init_env.reset()
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs

    def render(self, mode='human'):
        self._init_env.render(mode=mode)

    def close(self):
        self._init_env.close()

    def seed(self, seed=None):
        self._init_env.seed()
        # TODO seed also env space and observation space
