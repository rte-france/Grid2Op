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
from grid2op.gym_compat.utils import check_gym_version


class GymEnv(gym.Env):
    """
    fully implements the openAI gym API by using the :class:`GymActionSpace` and :class:`GymObservationSpace`
    for compliance with openAI gym.

    They can handle action_space_converter or observation_space converter to change the representation of data
    that will be fed to the agent.  #TODO

    Notes
    ------
    The environment passed as input is copied. It is not modified by this "gym environment"

    Examples
    --------
    This can be used like:

    .. code-block:: python

        import grid2op
        from grid2op.gym_compat import GymEnv

        env_name = ...
        env = grid2op.make(env_name)
        gym_env = GymEnv(env)  # is a gym environment properly inheriting from gym.Env !


    """
    def __init__(self, env_init):
        check_gym_version()
        self.init_env = env_init.copy()
        self.action_space = GymActionSpace(self.init_env)
        self.observation_space = GymObservationSpace(self.init_env)
        self.reward_range = self.init_env.reward_range
        self.metadata = self.init_env.metadata

    def step(self, gym_action):
        g2op_act = self.action_space.from_gym(gym_action)
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, float(reward), done, info

    def reset(self):
        g2op_obs = self.init_env.reset()
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs

    def render(self, mode='human'):
        """for compatibility with open ai gym render function"""
        super(GymEnv, self).render(mode=mode)
        self.init_env.render(mode=mode)

    def close(self):
        if hasattr(self, "init_env") and self.init_env is None:
            self.init_env.close()
            del self.init_env
        self.init_env = None
        if hasattr(self, "action_space") and self.action_space is not None:
            self.action_space.close()
        self.action_space = None
        if hasattr(self, "observation_space") and self.observation_space is not None:
            self.observation_space.close()
        self.observation_space = None

    def seed(self, seed=None):
        self.init_env.seed(seed)
        # TODO seed also env space and observation space

    def __del__(self):
        # delete possible dangling reference
        self.close()
