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
    def __init__(self,
                 grid2op_env: Environment,
                 time_out_ms: int=1e3) -> None:
        self.time_out_ms = time_out_ms  # in ms
        self.__last_act_send = time.perf_counter()
        self.__last_act_received = self.__last_act_send
        self._nb_dn_last = 0
        self._is_init_dn = False
        super().__init__(**grid2op_env.get_kwargs())
        self._is_init_dn = True

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
        
        # do the "do nothing" actions
        self._nb_dn_last = 0
        if self._is_init_dn:
            nb_dn = floor(1000. * (self.__last_act_received  - self.__last_act_send) / (self.time_out_ms))
        else:
            nb_dn = 0
        do_nothing_action = self.action_space()
        for _ in range(nb_dn):
            obs, reward, done, info = super().step(do_nothing_action)
            if done:
                info["nb_do_nothing"] = nb_dn
                info["nb_do_nothing_made"] = self._nb_dn_last
                return obs, reward, done, info
            self._nb_dn_last += 1
        
        # now do the action
        obs, reward, done, info = super().step(action)
        info["nb_do_nothing"] = nb_dn
        info["nb_do_nothing_made"] = self._nb_dn_last
        self.__last_act_send = time.perf_counter()
        return obs, reward, done, info

    def reset(self):
        self.__last_act_send = time.perf_counter()
        self.__last_act_received = self.__last_act_send
        self._is_init_dn = False
        res = super().reset()
        self.__last_act_send = time.perf_counter()
        self._is_init_dn = True
        return res
    
    def _custom_deepcopy_for_copy(self, new_obj):
        super()._custom_deepcopy_for_copy(new_obj)
        new_obj.__last_act_send = time.perf_counter()
        new_obj.__last_act_received = new_obj.__last_act_send
        new_obj._is_init_dn = self._is_init_dn
        new_obj.time_out_ms = self.time_out_ms
