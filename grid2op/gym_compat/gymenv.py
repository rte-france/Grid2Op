# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import gym

from grid2op.dtypes import dt_int
from grid2op.Chronics import Multifolder
from grid2op.gym_compat.gym_obs_space import GymObservationSpace
from grid2op.gym_compat.gym_act_space import GymActionSpace
from grid2op.gym_compat.utils import (check_gym_version, sample_seed,
                                      _MAX_GYM_VERSION_RANDINT, GYM_VERSION)


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

    def __init__(self, env_init, shuffle_chronics=True):
        check_gym_version()
        self.init_env = env_init.copy()
        self.action_space = GymActionSpace(self.init_env)
        self.observation_space = GymObservationSpace(self.init_env)
        self.reward_range = self.init_env.reward_range
        self.metadata = self.init_env.metadata
        self._shuffle_chronics = shuffle_chronics
        
        if GYM_VERSION <= _MAX_GYM_VERSION_RANDINT:
            self.seed = self._aux_seed
            self.reset = self._aux_reset
            self.step = self._aux_step
        else:
            self.reset = self._aux_reset_new
            self.step = self._aux_step_new
            
    def _aux_step(self, gym_action):
        # used for gym < 0.26
        g2op_act = self.action_space.from_gym(gym_action)
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, float(reward), done, info
    
    def _aux_step_new(self, gym_action):
        # used for gym >= 0.26
        # TODO refacto with _aux_step
        g2op_act = self.action_space.from_gym(gym_action)
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        gym_obs = self.observation_space.to_gym(g2op_obs)
        truncated = g2op_obs.current_step == g2op_obs.max_step
        return gym_obs, float(reward), done, truncated, info

    def _aux_reset(self, seed=None, return_info=None, options=None):
        # used for gym < 0.26
        if self._shuffle_chronics and isinstance(
            self.init_env.chronics_handler.real_data, Multifolder
        ):
            self.init_env.chronics_handler.sample_next_chronics()
         
        if seed is not None:
            self._aux_seed(seed)
            
        g2op_obs = self.init_env.reset()
        gym_obs = self.observation_space.to_gym(g2op_obs)
            
        if return_info:
            chron_id = self.init_env.chronics_handler.get_id()
            return gym_obs, {"time serie id": chron_id}
        else:
            return gym_obs

    def _aux_reset_new(self, seed=None, options=None):
        # used for gym > 0.26
        return self._aux_reset(seed, True, options)
        
    def render(self, mode="human"):
        """for compatibility with open ai gym render function"""
        super(GymEnv, self).render(mode=mode)
        self.init_env.render(mode=mode)

    def close(self):
        if hasattr(self, "init_env") and self.init_env is not None:
            self.init_env.close()
            del self.init_env
        self.init_env = None
        if hasattr(self, "action_space") and self.action_space is not None:
            self.action_space.close()
        self.action_space = None
        if hasattr(self, "observation_space") and self.observation_space is not None:
            self.observation_space.close()
        self.observation_space = None

    def _aux_seed(self, seed=None):
        # deprecated in gym >=0.26
        if seed is not None:
            # seed the gym env
            super().reset(seed=seed)
            # then seed the underlying grid2op env
            max_ = np.iinfo(dt_int).max 
            next_seed = sample_seed(max_, self._np_random)
            self.init_env.seed(next_seed)

    def __del__(self):
        # delete possible dangling reference
        self.close()
