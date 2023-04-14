# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import time
from math import ceil, floor
from grid2op.Environment.Environment import Environment


class TimedOutEnvironment(Environment):  # TODO heritage ou alors on met un truc de base
    """    This class is the grid2op implementation of a "timed out environment" entity in the RL framework.
    A TimedOutEnvironment instance has an attribute time_out_ms. This attribute represents the duration before a do_nothing action
    is performed, if no action is received. if the action is received after a x * time_out_ms duration, with n < x < n + 1 
    (n is an integer), n do_nothing actions are performed before the received action.

    Attributes
    ----------

    name: ``str``
        The name of the environment

    time_out_ms: ``int``
        maximum duration before performing a do_nothing action and updating to the next time_step.
    
    action_space: :class:`grid2op.Action.ActionSpace`
        Another name for :attr:`Environment.helper_action_player` for gym compatibility.

    observation_space:  :class:`grid2op.Observation.ObservationSpace`
        Another name for :attr:`Environment.helper_observation` for gym compatibility.

    reward_range: ``(float, float)``
        The range of the reward function

    metadata: ``dict``
        For gym compatibility, do not use

    spec: ``None``
        For Gym compatibility, do not use

    _viewer: ``object``
        Used to display the powergrid. Currently properly supported.

    """


    def __init__(self, time_out_ms: int=1e3) -> None:
        super().__init__()
        self.__last_act_send = 0.
        self.__last_act_received = 0.
        self.time_out_ms = time_out_ms  # in ms
    
    @staticmethod
    def __call__(regular_env):  # TimedOutEnvironment(regular_env)
        # TODO benjamin !
        raise NotImplementedError()

    def step(self, action):
        """Overload of the step function defined in BaseEnv.
        For each received action, measure the time elapsed since the last received action.
        If this duration is above the time_out, perform a do_nothing action for each elapsed time_out before executing the received action

        Args:
            action (_type_): the action send by the agent at this given time step

        Returns:
            obs, reward, done, info: internal variables to process the next step and calculate the score
        """        
        self.__last_act_received = time.perf_counter()
        
        # do the "do nothing"
        nb_dn = floor(1000. * (self.__last_act_received  - self.__last_act_send) / (self.time_out_ms))
        do_nothing_action = self.action_space()
        for _ in range(nb_dn):
            obs, reward, done, info = super().step(do_nothing_action)
            if done:
                return obs, reward, done, info
        
        # now do the action
        obs, reward, done, info = super().step(action)
        info["nb_do_nothing"] = nb_dn
        self.__last_act_send = time.perf_counter()
        return obs, reward, done, info

    def reset(self):
        res = super().reset()
        self.__last_act_send = time.perf_counter()
        return res
