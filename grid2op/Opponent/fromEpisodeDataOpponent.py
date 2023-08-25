# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import warnings

from grid2op.Opponent.baseOpponent import BaseOpponent
from grid2op.Chronics import FromOneEpisodeData
from grid2op.Exceptions import OpponentError


class FromEpisodeDataOpponent(BaseOpponent):
    """
    .. warning::
        This can only be used if your environment uses :class:`grid2op.Chronics.FromOneEpisodeData` 
        or XXX (from a list of episode data or directory)
        class otherwise it will NOT work.

    .. versionadded:: 1.9.4
    
    .. seealso::
        :class:`grid2op.Chronics.FromOneEpisodeData`
    
    Examples
    --------
    
    Provided that you stored some data in `path_agent` using a :class:`grid2op.Runner.Runner` for example
    you can use this class with:
    
    .. code-block:: python
    
        import grid2op
        from grid2op.Chronics import FromOneEpisodeData
        from grid2op.Opponent import FromEpisodeDataOpponent
        from grid2op.Episode import EpisodeData
        
        path_agent = ....  # same as above
        env_name = .... # same as above
        
        # path_agent is the path where data coming from a grid2op runner are stored
        # NB it should come from a do nothing agent, or at least
        # an agent that does not modify the injections (no redispatching, curtailment, storage)
        li_episode = EpisodeData.list_episode(path_agent)
        ep_data = li_episode[0]
        
        env = grid2op.make(env_name,
                           chronics_class=FromOneEpisodeData,  # super important
                           data_feeding_kwargs={"ep_data": ep_data},  # super important
                           opponent_class=FromEpisodeDataOpponent,  # important
                           opponent_attack_cooldown=1,  # super important
                      )
        # ep_data can be either a tuple of 2 elements (like above)
        # or a full path to a saved episode
        # or directly an object of type EpisodeData

        obs = env.reset()
    
        # and now you can use "env" as any grid2op environment.
        
    Parameters
    ----------
    BaseOpponent : _type_
        _description_
    """
    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self._attacks = []
        self._ptr_env = None
        
        self._warning_cooldown_issued = False
        
    def init(self, partial_env, **kwargs):
        if not isinstance(partial_env.chronics_handler.real_data, FromOneEpisodeData):
            raise OpponentError("FromEpisodeDataOpponent can only be used with FromOneEpisodeData time series !")
        self._ptr_env = partial_env
        self._attacks = copy.deepcopy(self._ptr_env.chronics_handler.real_data._episode_data.attacks)
        
    def reset(self, initial_budget):
        self._attacks = copy.deepcopy(self._ptr_env.chronics_handler.real_data._episode_data.attacks)
        if self._ptr_env._oppSpace.attack_cooldown > 1:
            if not self._warning_cooldown_issued:
                self._warning_cooldown_issued = True
                warnings.warn('When using FromEpisodeDataOpponent, make sure that your '
                              'environment is made with kwargs `opponent_attack_cooldown=1` for '
                              'this class to work properly.')
        
    def attack(self, observation, agent_action, env_action, budget, previous_fails):
        step = observation.current_step
        attack = None
        time = None
        tmp = self._attacks[step]
        if tmp is not None and tmp.can_affect_something():
            # there as been an attack at this step
            attack = tmp
            time = 1
        return attack, time
    
    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        pass
    
    def get_state(self):
        return (self._ptr_env, copy.deepcopy(self._attacks))
    
    def set_state(self, state):
        self._ptr_env = state[0]
        self._attacks = state[1]
