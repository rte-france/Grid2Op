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

    def _obj_caract_from_topo_id(self, id_):
        obj_id = None
        objt_type = None
        array_subid = None
        for l_id, id_in_topo in enumerate(self.load_pos_topo_vect):
            if id_in_topo == id_:
                obj_id = l_id
                objt_type = "load"
                array_subid = self.load_to_subid
        if obj_id is None:
            for l_id, id_in_topo in enumerate(self.gen_pos_topo_vect):
                if id_in_topo == id_:
                    obj_id = l_id
                    objt_type = "generator"
                    array_subid = self.gen_to_subid
        if obj_id is None:
            for l_id, id_in_topo in enumerate(self.line_or_pos_topo_vect):
                if id_in_topo == id_:
                    obj_id = l_id
                    objt_type = self._line_or_str
                    array_subid = self.line_or_to_subid
        if obj_id is None:
            for l_id, id_in_topo in enumerate(self.line_ex_pos_topo_vect):
                if id_in_topo == id_:
                    obj_id = l_id
                    objt_type = self._line_ex_str
                    array_subid = self.line_ex_to_subid
        if obj_id is None:
            for l_id, id_in_topo in enumerate(self.storage_pos_topo_vect):
                if id_in_topo == id_:
                    obj_id = l_id
                    objt_type = "storage"
                    array_subid = self.storage_to_subid
        if obj_id is None:
            for l_id, id_in_topo in enumerate(self.interco_pos_topo_vect):
                if id_in_topo == id_:
                    obj_id = l_id
                    objt_type = "interconnection"
                    array_subid = self.interco_to_subid
        substation_id = array_subid[obj_id]
        return obj_id, objt_type, substation_id