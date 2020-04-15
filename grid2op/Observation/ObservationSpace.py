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
    Helper that provides usefull functions to manipulate :class:`BaseObservation`.

    BaseObservation should only be built using this Helper. It is absolutely not recommended to make an observation
    directly form its constructor.

    This class represents the same concept as the "BaseObservation Space" in the OpenAI gym framework.

    Attributes
    ----------

    observationClass: ``type``
        Class used to build the observations. It defaults to :class:`CompleteObservation`

    _empty_obs: :class:`grid2op.Observation.BaseObservation`
        An empty observation with the proper dimensions.

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
        An instance of the observation that is updated and will be sent to he BaseAgent.

    """
    def __init__(self,
                 gridobj,
                 env,
                 rewardClass=None,
                 observationClass=CompleteObservation):
        """
        Env: requires :attr:`grid2op.Environment.parameters` and :attr:`grid2op.Environment.backend` to be valid
        """

        SerializableObservationSpace.__init__(self, gridobj, observationClass=observationClass)

        # TODO DOCUMENTATION !!!

        # print("ObservationSpace init with rewardClass: {}".format(rewardClass))
        self.parameters = copy.deepcopy(env.parameters)
        # for the observation, I switch between the _parameters for the environment and for the simulation
        self.parameters.ENV_DC = self.parameters.FORECAST_DC

        if rewardClass is None:
            self.rewardClass = env.rewardClass
        else:
            self.rewardClass = rewardClass

        # helpers
        self.action_helper_env = env.helper_action_env
        self.reward_helper = RewardHelper(rewardClass=self.rewardClass)
        self.reward_helper.initialize(env)

        other_rewards = {k: v.rewardClass for k, v in env.other_rewards.items()}

        # TODO here: have another backend maybe
        self.backend_obs = env.backend.copy()

        self.obs_env = _ObsEnv(backend_instanciated=self.backend_obs, obsClass=self.observationClass,
                              parameters=env.parameters,
                              reward_helper=self.reward_helper,
                              action_helper=self.action_helper_env,
                              thermal_limit_a=env._thermal_limit_a,
                              legalActClass=env.legalActClass,
                              donothing_act=env.helper_action_player(),
                              other_rewards=other_rewards)

        for k, v in self.obs_env.other_rewards.items():
            v.initialize(env)

        self._empty_obs = self.observationClass(gridobj=self,
                                                obs_env=self.obs_env,
                                                action_helper=self.action_helper_env)
        self._update_env_time = 0.

    def __call__(self, env):
        self.obs_env.update_grid(env)

        res = self.observationClass(gridobj=self,
                                    obs_env=self.obs_env,
                                    action_helper=self.action_helper_env)

        # TODO how to make sure that whatever the number of time i call "simulate" i still get the same observations
        # TODO use self.obs_prng when updating actions
        res.update(env=env)
        return res

    def size_obs(self):
        """
        Size if the observation vector would be flatten
        :return:
        """
        return self.n
