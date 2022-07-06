# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np

from grid2op.Exceptions import SimulateError
from grid2op.Observation import ObservationSpace, CompleteObservation
from grid2op.multi_agent.subGridObjects import SubGridObjects


class SubGridObservation(SubGridObjects, CompleteObservation):
    # TODO NOT TESTED in general
    def __init__(self,
                 is_complete_obs=False,
                 obs_env=None,
                 action_helper=None,
                 random_prng=None):
        SubGridObjects.__init__(self)
        CompleteObservation.__init__(self, obs_env, action_helper, random_prng)
        self._is_complete_obs = is_complete_obs
    
    def simulate(self, actions, time_step=1):
        raise NotImplementedError("You should think about what it means to do a simulate !")
        if not self._is_complete_obs:
            raise SimulateError("Impossible to forecast the impact of a local action if you "
                                "have the complete information about the grid. "
                                "Your agent can only observe a small part of the grid.")
            
        # 1) convert the actions to a global action
        complete_obs, global_reward, done, info = super().simulate(action, time_step)
        # 2) convert complete_obs in the Observation, dispatch reward and all...
    
    def update(self, env, complete_obs, with_forecast=True):
        # env: centralized env
        # complete_obs: the complete observation at this step
        
        if self._is_complete_obs:
            CompleteObservation.update(self, env, with_forecast)
        else:
            self._reset_matrices()
            self.reset()
            self._update_from_complete_obs(complete_obs)
        
    def _update_from_complete_obs(self, complete_obs):
        # TODO read the correct data from the observation in this case
        # and update the "interco" data
        raise NotImplementedError("This is not implemented at the moment")
        
    
class SubGridObservationSpace(SubGridObjects, ObservationSpace):
    # modeling choice : one observation space per "sub agent"
    
    # TODO NOT TESTED
    def __init__(
        self,
        full_gridobj,  # full grid, used for "simulate"
        ma_env,
        local_gridobj,  # gridobject for the observation space of this agent
        is_complete_obs=True, # whether the observation is complete or not
        rewardClass=None,
        observationClass=SubGridObservation,  # should be a local observation
        actionClass=None,  # Complete action, used internally for simulate
        with_forecast=True,
        kwargs_observation=None,
        logger=None,
    ):
        SubGridObjects.__init__(self)
        ObservationSpace.__init__(self,
                                  full_gridobj,
                                  ma_env._cent_env,  # pass the centralized env for the observation space, maybe ?
                                  rewardClass,
                                  CompleteObservation,
                                  actionClass,
                                  with_forecast,
                                  kwargs_observation,
                                  logger)
        self.local_obs_cls = observationClass.init_grid(local_gridobj)
        self._is_complete_obs = is_complete_obs
    
    def __call__(self, ma_env, cent_obs, _update_state=True):
        # cent obs is the centralized observation
        # ma_env is the multi agent env
        if self.with_forecast:
            # update the backend for the "simulate"
            self.obs_env.update_grid(ma_env._cent_env)
        
        # self.__nb_simulate_called_this_step = 0 
        self._ObservationSpace__nb_simulate_called_this_step = 0

        res = self.local_obs_cls(
            self._is_complete_obs,
            obs_env=self.obs_env if self.obs_env.is_valid() else None,
            action_helper=self.action_helper_env,
            random_prng=self.space_prng,
            **self._ptr_kwargs_observation
        )
        
        if _update_state:
            res.update(ma_env._cent_env,
                       cent_obs,
                       with_forecast=self.with_forecast)
        return res