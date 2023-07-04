# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.dtypes import dt_int
from grid2op.Chronics import Multifolder
from grid2op.gym_compat.utils import GYM_AVAILABLE, GYMNASIUM_AVAILABLE
from grid2op.gym_compat.utils import (check_gym_version, sample_seed)
    
    
def conditional_decorator(condition):
    def decorator(func):
        if condition:
            # Return the function unchanged, not decorated.
            return func
        return NotImplementedError()  # anything that is not a callbe anyway
    return decorator


class __AuxGymEnv:
    """
    fully implements the openAI gym API by using the :class:`GymActionSpace` and :class:`GymObservationSpace`
    for compliance with openAI gym.

    They can handle action_space_converter or observation_space converter to change the representation of data
    that will be fed to the agent.  #TODO

    .. warning::
        The `gym` package has some breaking API change since its version 0.26. Depending on the version installed,
        we attempted, in grid2op, to maintain compatibility both with former version and later one. This makes this
        class behave differently depending on the version of gym you have installed !
        
        The main changes involve the functions `env.step` and `env.reset`
        
    If you want to use the same version of the GymEnv regardless of the gym / gymnasium version installed you can use:
    
    - :class:`GymnasiumEnv` if `gymnasium` is available
    - :class:`GymEnv_Legacy` for gym < 0.26
    - :class:`GymEnv_Modern` for gym >= 0.26

    .. warning::
        Depending on the presence absence of gymnasium and gym packages this class might behave differently.
        
        In grid2op we tried to maintain compatibility both with gymnasium (newest) and gym (legacy, 
        no more maintained) RL packages. The behaviour is the following:
        
        - :class:`GymEnv` will inherit from gymnasium if it's installed 
          (in this case it will be :class:`GymnasiumEnv`), otherwise it will
          inherit from gym (and will be exactly :class:`GymEnv_Legacy` - gym < 0.26- 
          or :class:`GymEnv_Modern` - for gym >= 0.26)
        - :class:`GymnasiumEnv` will inherit from gymnasium if it's available and never from
          from gym
        - :class:`GymEnv_Legacy` and :class:`GymEnv_Modern` will inherit from gym if it's 
          available and never from from gymnasium
        
        See :ref:`gymnasium_gym` for more information
        
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

    There are a few difference between "raw" grid2op environment and gymnasium environments.
    
    One of the major difference is that, to our knowledge, gymnasium does not support the 
    `simulate` feature (which allows an agent to test the impact of a given action 
    on the grid without having to perform a `step` see :ref:`model_based_rl` for more information)
    [NB if you know or better are developping some "model based RL library" let us know !]
    
    Another difference is in the way to do some actions. In grid2op, actions are a dedicated class
    and can be made with an `action_space` and a dictionary, or  using the properties of the action
    class.
    
    In gym, there are no specific representations of the action class. More precisely, for each action
    type (:class:`MultiDiscreteActSpace`, :class:`DiscreteActSpace`, :class:`BoxGymActSpace` or 
    :class:`GymActionSpace`) there is a way to encode it. For example, by default (:class:`GymActionSpace`)
    an action is represented through an OrderedDict (`from collection import OrderedDict`)
    """

    def __init__(self, env_init, shuffle_chronics=True, render_mode="rgb_array"):
        check_gym_version(type(self)._gymnasium)
        self.init_env = env_init.copy()
        self.action_space = type(self)._ActionSpaceType(self.init_env)
        self.observation_space = type(self)._ObservationSpaceType(self.init_env)
        self.reward_range = self.init_env.reward_range
        self.metadata = self.init_env.metadata
        self.init_env.render_mode = render_mode
        self._shuffle_chronics = shuffle_chronics
            
        super().__init__()  # super should reference either gym.Env or gymnasium.Env
        if not hasattr(self, "_np_random"):
            # for older version of gym it does not exist
            self._np_random = np.random.RandomState()
        
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
        g2op_obs, reward, terminated, info = self.init_env.step(g2op_act)
        gym_obs = self.observation_space.to_gym(g2op_obs)
        truncated = False # see https://github.com/openai/gym/pull/2752
        return gym_obs, float(reward), terminated, truncated, info

    def _aux_reset(self, seed=None, return_info=None, options=None):
        # used for gym < 0.26
        if self._shuffle_chronics and isinstance(
            self.init_env.chronics_handler.real_data, Multifolder
        ):
            self.init_env.chronics_handler.sample_next_chronics()
         
        if seed is not None:
            seed_, next_seed, underlying_env_seeds = self._aux_seed(seed)
            
        g2op_obs = self.init_env.reset()
        gym_obs = self.observation_space.to_gym(g2op_obs)
            
        if return_info:
            chron_id = self.init_env.chronics_handler.get_id()
            info = {"time serie id": chron_id}
            if seed is not None:
                info["seed"] = seed
                info["grid2op_env_seed"] = next_seed
                info["underlying_env_seeds"] = underlying_env_seeds
            return gym_obs, info
        else:
            return gym_obs

    def _aux_reset_new(self, seed=None, options=None):
        # used for gym > 0.26
        if self._shuffle_chronics and isinstance(
            self.init_env.chronics_handler.real_data, Multifolder
        ):
            self.init_env.chronics_handler.sample_next_chronics()
        
        super().reset(seed=seed)    
        if seed is not None:
            self._aux_seed_spaces()
            seed, next_seed, underlying_env_seeds = self._aux_seed_g2op(seed)
            
        g2op_obs = self.init_env.reset()
        gym_obs = self.observation_space.to_gym(g2op_obs)
            
        chron_id = self.init_env.chronics_handler.get_id()
        info = {"time serie id": chron_id}
        if seed is not None:
            info["seed"] = seed
            info["grid2op_env_seed"] = next_seed
            info["underlying_env_seeds"] = underlying_env_seeds
        return gym_obs, info
        
    def render(self):
        """for compatibility with open ai gym render function"""
        return self.init_env.render()

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

    def _aux_seed_spaces(self):
        max_ = np.iinfo(dt_int).max 
        next_seed = sample_seed(max_, self._np_random)
        self.action_space.seed(next_seed)
        next_seed = sample_seed(max_, self._np_random)
        self.observation_space.seed(next_seed)
            
    def _aux_seed_g2op(self, seed):
            # then seed the underlying grid2op env
            max_ = np.iinfo(dt_int).max 
            next_seed = sample_seed(max_, self._np_random)
            underlying_env_seeds = self.init_env.seed(next_seed)
            return seed, next_seed, underlying_env_seeds
        
    def _aux_seed(self, seed=None):
        # deprecated in gym >=0.26
        if seed is not None:
            # seed the gym env
            super().seed(seed)
            self._np_random.seed(seed)
            self._aux_seed_spaces()
            return self._aux_seed_g2op(seed)
        return None, None, None

    def __del__(self):
        # delete possible dangling reference
        self.close()

if GYM_AVAILABLE:
    from gym import Env as LegacyGymEnv
    from grid2op.gym_compat.gym_obs_space import LegacyGymObservationSpace
    from grid2op.gym_compat.gym_act_space import LegacyGymActionSpace
    _AuxGymEnv = type("_AuxGymEnv",
                      (__AuxGymEnv, LegacyGymEnv),
                      {"_gymnasium": False,
                       "_ActionSpaceType": LegacyGymActionSpace,
                       "_ObservationSpaceType": LegacyGymObservationSpace,
                       "__module__": __name__})
    _AuxGymEnv.__doc__ = __AuxGymEnv.__doc__
    class GymEnv_Legacy(_AuxGymEnv):
        # for old version of gym        
        def reset(self, *args, **kwargs):
            return self._aux_reset(*args, **kwargs)

        def step(self, action):
            return self._aux_step(action)

        def seed(self, seed):
            # defined only on some cases
            return self._aux_seed(seed)


    class GymEnv_Modern(_AuxGymEnv):
        # for new version of gym
        def reset(self,
                *,
                seed=None,
                options=None,):
            return self._aux_reset_new(seed, options)

        def step(self, action):
            return self._aux_step_new(action)
    GymEnv_Legacy.__doc__ = __AuxGymEnv.__doc__
    GymEnv_Modern.__doc__ = __AuxGymEnv.__doc__


if GYMNASIUM_AVAILABLE:
    from gymnasium import Env
    from grid2op.gym_compat.gym_act_space import GymnasiumActionSpace
    from grid2op.gym_compat.gym_obs_space import GymnasiumObservationSpace
    _AuxGymnasiumEnv = type("_AuxGymnasiumEnv",
                            (__AuxGymEnv, Env),
                            {"_gymnasium": True,
                             "_ActionSpaceType": GymnasiumActionSpace,
                             "_ObservationSpaceType": GymnasiumObservationSpace,
                             "__module__": __name__})
    _AuxGymnasiumEnv.__doc__ = __AuxGymEnv.__doc__
    
    class GymnasiumEnv(_AuxGymnasiumEnv):
        # for new version of gym
        def reset(self,
                *,
                seed=None,
                options=None,):
            return self._aux_reset_new(seed, options)

        def step(self, action):
            return self._aux_step_new(action)
    GymnasiumEnv.__doc__ = __AuxGymEnv.__doc__