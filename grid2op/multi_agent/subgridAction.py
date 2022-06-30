# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Action import PlayableAction, ActionSpace, BaseAction
from grid2op.multi_agent.subGridObjects import SubGridObjects


# TODO (later) make that a meta class too
class SubGridActionSpace(SubGridObjects, ActionSpace):
    def __init__(
        self,
        gridobj,
        legal_action,
        agent_name,
        actionClass=BaseAction,  # need to be a base grid2op type (and not a type generated on the fly)
        ):
        SubGridObjects.__init__(self)
        ActionSpace.__init__(self,
                             gridobj=gridobj,
                             legal_action=legal_action,
                             actionClass=actionClass,
                             _extra_name=agent_name)
    
    def _get_possible_action_types(self):
        """Overrides an ActionSpace's method

        Returns
        -------
        list
            All possible action types
        """
        rnd_types = []
        cls = type(self)
        if self.n_line > 0: #TODO interco v0.1
            if "set_line_status" in self.actionClass.authorized_keys:
                rnd_types.append(cls.SET_STATUS_ID)
            if "change_line_status" in self.actionClass.authorized_keys:
                rnd_types.append(cls.CHANGE_STATUS_ID)
            if "set_bus" in self.actionClass.authorized_keys:
                rnd_types.append(cls.SET_BUS_ID)
            if "change_bus" in self.actionClass.authorized_keys:
                rnd_types.append(cls.CHANGE_BUS_ID)
        
        if self.n_gen > 0 and (self.gen_redispatchable).any():
            if "redispatch" in self.actionClass.authorized_keys:
                rnd_types.append(cls.REDISPATCHING_ID)
                
        if self.n_storage > 0 and "storage_power" in self.actionClass.authorized_keys:
            rnd_types.append(cls.STORAGE_POWER_ID)
            
        if self.dim_alarms > 0 and "raise_alarm" in self.actionClass.authorized_keys:
            rnd_types.append(cls.RAISE_ALARM_ID)
        return rnd_types
         
        
# TODO (later) make that a "metaclass" with argument the ActionType (here playable action)
class SubGridAction(SubGridObjects, PlayableAction):
    def __init__(self):
        SubGridObjects.__init__(self)
        PlayableAction.__init__(self)
