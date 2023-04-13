# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import time
from math import ceil
from grid2op.Environment.Environment import Environment


class TimedOutEnvironment(Environment):  # TODO heritage ou alors on met un truc de base
    """_summary_

    Args:
        BaseEnv (_type_): _description_
    """
    def __init__(self, time_out_ms=1.0) -> None:
        super().__init__()
        self.__last_act_send = 0.
        self.__last_act_received = 0.
        self.time_out_ms = 1.  # in ms
    
    @staticmethod
    def __call__(regular_env):  # TimedOutEnvironment(regular_env)
        # TODO benjamin !
        raise NotImplementedError()

    def step(self, action):
        self.__last_act_received = time.perf_counter()
        
        # do the "do nothing"
        nb_dn = int(ceil(1000. * (self.__last_act_received  - self.__last_act_send)) // int(self.time_out_ms))
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
