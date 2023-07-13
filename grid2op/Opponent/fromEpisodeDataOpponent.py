# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy

from grid2op.Opponent.baseOpponent import BaseOpponent
from grid2op.Chronics import FromOneEpisodeData
from grid2op.Exceptions import OpponentError


class FromEpisodeDataOpponent(BaseOpponent):
    """
    .. warning::
        This can only be used if your environment uses :class:`grid2op.Chronics.FromOneEpisodeData` 
        or XXX (from a list of episode data or directory)
        class otherwise it will NOT work.

    .. versionadded:: 1.9.2
    
    Parameters
    ----------
    BaseOpponent : _type_
        _description_
    """
    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self._attacks = []
        self._ptr_env = None
        
    def init(self, partial_env, **kwargs):
        if not isinstance(partial_env.chronics_handler.real_data, FromOneEpisodeData):
            raise OpponentError("FromEpisodeDataOpponent can only be used with FromOneEpisodeData time series !")
        self._ptr_env = partial_env
        self._attacks = copy.deepcopy(self._ptr_env.chronics_handler.real_data._episode_data.attacks)
        
    def reset(self, initial_budget):
        self._attacks = copy.deepcopy(self._ptr_env.chronics_handler.real_data._episode_data.attacks)
        
    def attack(self, observation, agent_action, env_action, budget, previous_fails):
        step = observation.current_step
        attack = None
        time = None
        tmp = self._attacks[step]
        if tmp.can_affect_something():
            # there as been an attack at this step
            attack = tmp
            time = 1
        return attack, time
    
    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        # should not be called at all
        pass
    
    def get_state(self):
        return (self._ptr_env, copy.deepcopy(self._attacks))
    
    def set_state(self, state):
        self._ptr_env = state[0]
        self._attacks = state[1]
