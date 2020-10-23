# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy

from grid2op.Observation.SerializableObservationSpace import SerializableObservationSpace
from grid2op.Reward import RewardHelper
from grid2op.Observation.CompleteObservation import CompleteObservation
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

    parameters: :class:`grid2op.Parameters.Parameters`
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
                 with_forecast=True):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Env: requires :attr:`grid2op.Environment.parameters` and :attr:`grid2op.Environment.backend` to be valid
        """

        SerializableObservationSpace.__init__(self, gridobj, observationClass=observationClass)

        self.with_forecast = with_forecast
        # print("ObservationSpace init with rewardClass: {}".format(rewardClass))
        self.parameters = copy.deepcopy(env.parameters)
        # for the observation, I switch between the _parameters for the environment and for the simulation
        self.parameters.ENV_DC = self.parameters.FORECAST_DC

        if rewardClass is None:
            self.rewardClass = env.rewardClass
        else:
            self.rewardClass = rewardClass

        # helpers
        self.action_helper_env = env._helper_action_env
        self.reward_helper = RewardHelper(rewardClass=self.rewardClass)
        self.reward_helper.initialize(env)

        other_rewards = {k: v.rewardClass for k, v in env.other_rewards.items()}

        # TODO here: have another backend maybe
        self._backend_obs = env.backend.copy()

        _ObsEnv_class = _ObsEnv.init_grid(self._backend_obs)
        self.obs_env = _ObsEnv_class(backend_instanciated=self._backend_obs,
                                     obsClass=self.observationClass,
                                     parameters=env.parameters,
                                     reward_helper=self.reward_helper,
                                     action_helper=self.action_helper_env,
                                     thermal_limit_a=env.get_thermal_limit(),
                                     legalActClass=env._legalActClass,
                                     donothing_act=env._helper_action_player(),
                                     other_rewards=other_rewards,
                                     completeActionClass=env._helper_action_env.actionClass,
                                     helper_action_class=env._helper_action_class,
                                     helper_action_env=env._helper_action_env)
        for k, v in self.obs_env.other_rewards.items():
            v.initialize(env)

        self._empty_obs = self.observationClass(obs_env=self.obs_env,
                                                action_helper=self.action_helper_env)
        self._update_env_time = 0.

    def reset_space(self):
        if self.with_forecast:
            self.obs_env.reset_space()
        self.action_helper_env.actionClass.reset_space()

    def __call__(self, env):
        if self.with_forecast:
            self.obs_env.update_grid(env)

        res = self.observationClass(obs_env=self.obs_env,
                                    action_helper=self.action_helper_env)

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
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        return an empty observation, for internal use only."""
        return copy.deepcopy(self._empty_obs)

    def copy(self):
        """
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
        res = copy.deepcopy(self)
        res._backend_obs = backend.copy()
        res._empty_obs = obs_.copy()
        res.obs_env = obs_env.copy()

        # assign back the results
        self._backend_obs = backend
        self._empty_obs = obs_
        self.obs_env = obs_env

        return res
