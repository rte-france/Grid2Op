# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import sys
import copy

from grid2op.Observation.serializableObservationSpace import SerializableObservationSpace
from grid2op.Reward import RewardHelper
from grid2op.Observation.completeObservation import CompleteObservation
from grid2op.Observation._ObsEnv import _ObsEnv


class ObservationSpace(SerializableObservationSpace):
    """
    Helper that provides useful functions to manipulate :class:`BaseObservation`.

    BaseObservation should only be built using this Helper. It is absolutely not recommended to make an observation
    directly form its constructor.

    This class represents the same concept as the "BaseObservation Space" in the OpenAI gym framework.

    Attributes
    ----------
    with_forecast: ``bool``
        If ``True`` the :func:`BaseObservation.simulate` will be available. If ``False`` it will deactivate this
        possibility. If `simulate` function is not used, setting it to ``False`` can lead to non neglectible speed-ups.

    observationClass: ``type``
        Class used to build the observations. It defaults to :class:`CompleteObservation`

    _simulate_parameters: :class:`grid2op.Parameters.Parameters`
        Type of Parameters used to compute powerflow for the forecast.

    rewardClass: ``type``
        Class used by the :class:`grid2op.Environment.Environment` to send information about its state to the
        :class:`grid2op.BaseAgent.BaseAgent`. You can change this class to differentiate between the reward of output of
        :func:`BaseObservation.simulate`  and the reward used to train the BaseAgent.

    action_helper_env: :class:`grid2op.Action.ActionSpace`
        BaseAction space used to create action during the :func:`BaseObservation.simulate`

    reward_helper: :class:`grid2op.Reward.HelperReward`
        BaseReward function used by the the :func:`BaseObservation.simulate` function.

    obs_env: :class:`_ObsEnv`
        Instance of the environment used by the BaseObservation Helper to provide forcecast of the grid state.

    _empty_obs: :class:`BaseObservation`
        An instance of the observation with appropriate dimensions. It is updated and will be sent to he BaseAgent.

    """
    def __init__(self,
                 gridobj,
                 env,
                 rewardClass=None,
                 observationClass=CompleteObservation,
                 actionClass=None,
                 with_forecast=True,
                 kwargs_observation=None,):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Env: requires :attr:`grid2op.Environment.parameters` and :attr:`grid2op.Environment.backend` to be valid
        """

        if actionClass is None:
            from grid2op.Action import CompleteAction
            actionClass = CompleteAction

        SerializableObservationSpace.__init__(self, gridobj, observationClass=observationClass)
        self.with_forecast = with_forecast
        self._simulate_parameters = copy.deepcopy(env.parameters)
        self._legal_action = env._game_rules.legal_action
        self._env_param  = copy.deepcopy(env.parameters)

        if rewardClass is None:
            self._reward_func = env._reward_helper.template_reward
        else:
            self._reward_func = rewardClass

        # helpers
        self.action_helper_env = env._helper_action_env
        self.reward_helper = RewardHelper(reward_func=self._reward_func)
        self.reward_helper.initialize(env)

        other_rewards = {k: v.rewardClass for k, v in env.other_rewards.items()}

        # TODO here: have another backend class maybe
        self._backend_obs = env.backend.copy()
        _ObsEnv_class = _ObsEnv.init_grid(type(env.backend), force_module=_ObsEnv.__module__)
        _ObsEnv_class._INIT_GRID_CLS = _ObsEnv  # otherwise it's lost
        setattr(sys.modules[_ObsEnv.__module__], _ObsEnv_class.__name__, _ObsEnv_class)
        self.obs_env = _ObsEnv_class(init_grid_path=None,  # don't leak the path of the real grid to the observation space
                                     backend_instanciated=self._backend_obs,
                                     obsClass=CompleteObservation,  # do not put self.observationClass otherwise it's initialized twice
                                     parameters=self._simulate_parameters,
                                     reward_helper=self.reward_helper,
                                     action_helper=self.action_helper_env,
                                     thermal_limit_a=env.get_thermal_limit(),
                                     legalActClass=copy.deepcopy(env._legalActClass),
                                     other_rewards=other_rewards,
                                     helper_action_class=env._helper_action_class,
                                     helper_action_env=env._helper_action_env,
                                     epsilon_poly=env._epsilon_poly,
                                     tol_poly=env._tol_poly,
                                     has_attention_budget=env._has_attention_budget,
                                     attention_budget_cls=env._attention_budget_cls,
                                     kwargs_attention_budget=env._kwargs_attention_budget,
                                     max_episode_duration=env.max_episode_duration(),
                                     _complete_action_cls=env._complete_action_cls,
                                     _ptr_orig_obs_space=self,
                                     )
        for k, v in self.obs_env.other_rewards.items():
            v.initialize(env)

        self._empty_obs = self._template_obj
        self._update_env_time = 0.
        self.__nb_simulate_called_this_step = 0
        self.__nb_simulate_called_this_episode = 0

        # extra argument to build the observation
        if kwargs_observation is None:
            kwargs_observation = {}
        self._ptr_kwargs_observation = kwargs_observation

    def simulate_called(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Tells this class that the "obs.simulate" function has been called.
        """
        self.__nb_simulate_called_this_step += 1
        self.__nb_simulate_called_this_episode += 1

    @property
    def nb_simulate_called_this_episode(self):
        return self.__nb_simulate_called_this_episode

    @property
    def nb_simulate_called_this_step(self):
        return self.__nb_simulate_called_this_step

    def can_use_simulate(self) -> bool:
        """
        This checks on the rules if the agent has not made too many calls to "obs.simulate" this step
        """
        return self._legal_action.can_use_simulate(self.__nb_simulate_called_this_step, self.__nb_simulate_called_this_episode, self._env_param)

    def _change_parameters(self, new_param):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        change the parameter of the "simulate" environment
        """
        self.obs_env.change_parameters(new_param)
        self._simulate_parameters = new_param

    def change_other_rewards(self, dict_reward):
        """
        this function is used to change the "other rewards" used when you perform simulate.

        This can be used, for example, when you want to do faster call to "simulate". In this case you can remove all
        the "other_rewards" that will be used by the simulate function.

        Parameters
        ----------
        dict_reward: ``dict``
            see description of :attr:`grid2op.Environment.BaseEnv.other_rewards`

        Examples
        ---------
        If you want to deactivate the reward in the simulate function, you can do as following:

        .. code-block:: python

           import grid2op
           from grid2op.Reward import CloseToOverflowReward, L2RPNReward, RedispReward
           env_name = "l2rpn_case14_sandbox"
           other_rewards = {"close_overflow": CloseToOverflowReward,
                            "l2rpn": L2RPNReward,
                            "redisp": RedispReward}
           env = grid2op.make(env_name, other_rewards=other_rewards)

           env.observation_space.change_other_rewards({})

        """
        from grid2op.Reward import BaseReward
        from grid2op.Exceptions import Grid2OpException
        self.obs_env.other_rewards = {}
        for k, v in dict_reward.items():
            if not issubclass(v, BaseReward):
                raise Grid2OpException("All values of \"rewards\" key word argument should be classes that inherit "
                                       "from \"grid2op.BaseReward\"")
            if not isinstance(k, str):
                raise Grid2OpException("All keys of \"rewards\" should be of string type.")
            self.obs_env.other_rewards[k] = RewardHelper(v)

        for k, v in self.obs_env.other_rewards.items():
            v.initialize(self.obs_env)

    def change_reward(self, reward_func):
        self.obs_env._reward_helper.change_reward(reward_func)

    def reset_space(self):
        if self.with_forecast:
            self.obs_env.reset_space()
        self.action_helper_env.actionClass.reset_space()

    def __call__(self, env, _update_state=True):
        if self.with_forecast:
            self.obs_env.update_grid(env)

        res = self.observationClass(obs_env=self.obs_env,
                                    action_helper=self.action_helper_env,
                                    random_prng=self.space_prng,
                                    **self._ptr_kwargs_observation)
        self.__nb_simulate_called_this_step = 0
        if _update_state:
            # TODO how to make sure that whatever the number of time i call "simulate" i still get the same observations
            # TODO use self.obs_prng when updating actions
            res.update(env=env, with_forecast=self.with_forecast)
        return res

    def size_obs(self):
        """
        Size if the observation vector would be flatten
        :return:
        """
        return self.n

    def get_empty_observation(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        return an empty observation, for internal use only.
        """
        return copy.deepcopy(self._empty_obs)

    def reset(self, real_env):
        """reset the observation space with the new values of the environment"""
        self.obs_env._reward_helper.reset(real_env)
        self.__nb_simulate_called_this_step = 0
        self.__nb_simulate_called_this_episode = 0
        for k, v in self.obs_env.other_rewards.items():
            v.reset(real_env)
        self._env_param  = copy.deepcopy(real_env.parameters)

    def _custom_deepcopy_for_copy(self, new_obj):
        """implements a faster "res = copy.deepcopy(self)" to use 
        in "self.copy"
        Do not use it anywhere else...
        """
        # TODO clean that after it is working... (ie make this method per class...)
        # fill the super classes
        super()._custom_deepcopy_for_copy(new_obj)

        # now fill my class
        new_obj.with_forecast = self.with_forecast
        new_obj._simulate_parameters = copy.deepcopy(self._simulate_parameters)
        new_obj._reward_func = copy.deepcopy(self._reward_func)
        new_obj.action_helper_env = self.action_helper_env  # const
        new_obj.reward_helper = copy.deepcopy(self.reward_helper)
        new_obj._backend_obs = self._backend_obs  # ptr to a backend for simulate
        new_obj.obs_env = self.obs_env  # it is None anyway !
        new_obj._update_env_time = self._update_env_time
        new_obj.__nb_simulate_called_this_step = self.__nb_simulate_called_this_step
        new_obj.__nb_simulate_called_this_episode = self.__nb_simulate_called_this_episode
        new_obj._env_param  = copy.deepcopy(self._env_param)

        # as it's a "pointer" it's init from the env when needed here
        # this is why i don't deep copy it here !
        new_obj._ptr_kwargs_observation = self._ptr_kwargs_observation  

    def copy(self, copy_backend=False):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Perform a deep copy of the Observation space.

        """
        backend = self._backend_obs
        self._backend_obs = None
        obs_ = self._empty_obs
        self._empty_obs = None
        obs_env = self.obs_env
        self.obs_env = None

        # performs the copy
        # res = copy.deepcopy(self)  # painfully slow...
        # create an empty "me"
        my_cls = type(self)
        res = my_cls.__new__(my_cls)
        self._custom_deepcopy_for_copy(res)

        if not copy_backend:
            res._backend_obs = backend
            res._empty_obs = obs_.copy()
            res.obs_env = obs_env
        else:
            res.obs_env = obs_env.copy()
            res.obs_env._ptr_orig_obs_space = res
            res._backend_obs = res.obs_env.backend
            res._empty_obs = obs_.copy()
            res._empty_obs._obs_env = res.obs_env

        # assign back the results
        self._backend_obs = backend
        self._empty_obs = obs_
        self.obs_env = obs_env

        return res

    def close(self):
        if self.obs_env is not None:
            self.obs_env.close()
            
        del self.obs_env
        self.obs_env = None
