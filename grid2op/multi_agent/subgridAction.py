# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np

from grid2op.dtypes import dt_bool, dt_int
from grid2op.Exceptions import IllegalAction
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
            
        # TODO INTERCO !!!
        return rnd_types
         
        
# TODO (later) make that a "metaclass" with argument the ActionType (here playable action)
class SubGridAction(SubGridObjects, PlayableAction):
    
    # TODO make the "PlayableAction" type generic !
    authorized_keys = copy.deepcopy(PlayableAction.authorized_keys)
    authorized_keys.add('change_interco_status')
    authorized_keys.add('set_interco_status')
    
    attr_list_vect = copy.deepcopy(PlayableAction.attr_list_vect)
    attr_list_vect.append("_set_interco_status")
    attr_list_vect.append("_switch_interco_status")
        
    def __init__(self):
        SubGridObjects.__init__(self)
        PlayableAction.__init__(self)
    
        self.authorized_keys_to_digest["change_interco_status"] = self._digest_change_interco_status
        self.authorized_keys_to_digest["set_interco_status"] = self._digest_set_interco_status
        
        # add the things for the interco
        self._set_interco_status = np.full(shape=self.n_interco, fill_value=0, dtype=dt_int)
        self._switch_interco_status = np.full(
            shape=self.n_interco, fill_value=False, dtype=dt_bool
        )
        
        self._modif_interco_set_status = False
        self._modif_interco_change_status = False
    
    def _obj_caract_from_topo_id_others(self, id_):
        obj_id = None
        objt_type = None
        array_subid = None
        for l_id, id_in_topo in enumerate(self.interco_pos_topo_vect):
            if id_in_topo == id_:
                obj_id = l_id
                side_ = "(or)" if self.interco_is_origin[l_id] else "(ex)"
                objt_type = f"interco {side_}"
                array_subid = self.interco_to_subid
        return obj_id, objt_type, array_subid
    
    @property
    def interco_change_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which the origin side of powerlines are **changed**.

        It behaves similarly as :attr:`BaseAction.gen_change_bus`. See the help there for more information.
        """
        res = self.change_bus[self.interco_pos_topo_vect]
        res.flags.writeable = False
        return res

    @interco_change_bus.setter
    def interco_change_bus(self, values):
        if "change_bus" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the line (origin) bus (with "change") with this action type.'
            )
        orig_ = self.interco_change_bus
        try:
            self._aux_affect_object_bool(
                values,
                "interco",
                self.n_interco,
                self.name_interco,
                self.interco_pos_topo_vect,
                self._change_bus_vect,
            )
            self._modif_change_bus = True
        except Exception as exc_:
            self._change_bus_vect[self.interco_pos_topo_vect] = orig_
            raise IllegalAction(
                f"Impossible to modify the interconnection bus with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def interco_set_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which the origin side of each powerline is **set**.

        It behaves similarly as :attr:`BaseAction.gen_set_bus`. See the help there for more information.
        """
        res = self.set_bus[self.interco_pos_topo_vect]
        res.flags.writeable = False
        return res

    @interco_set_bus.setter
    def interco_set_bus(self, values):
        if "set_bus" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the line (origin) bus (with "set") with this action type.'
            )
        orig_ = self.interco_set_bus
        try:
            self._aux_affect_object_int(
                values,
                "interco",
                self.n_interco,
                self.name_interco,
                self.interco_pos_topo_vect,
                self._set_topo_vect,
            )
            self._modif_set_bus = True
        except Exception as exc_:
            self._aux_affect_object_int(
                orig_,
                'interco',
                self.n_interco,
                self.name_interco,
                self.interco_pos_topo_vect,
                self._set_topo_vect,
            )
            raise IllegalAction(
                f"Impossible to modify the interco bus with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )
            
    def _digest_setbus_other_elements(self, ddict_, handled):
        """may be used by the derived classes to set_bus with some other elements"""
        if "intercos_id" in ddict_:
            self.interco_set_bus = ddict_["intercos_id"]
            handled = True
        return handled
    
    def _digest_changebus_other_elements(self, ddict_, handled):
        """may be used by the derived classes to set_bus with some other elements"""
        if "intercos_id" in ddict_:
            self.interco_change_bus = ddict_["intercos_id"]
            handled = True
        return handled

    @property
    def interco_set_status(self) -> np.ndarray:
        if "set_interco_status" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the status of interconnections (with "set") with this action type.'
            )
        res = 1 * self._set_interco_status
        res.flags.writeable = False
        return res
    
    @interco_set_status.setter
    def interco_set_status(self, values):
        if "set_interco_status" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the status of interconnections (with "set") with this action type.'
            )
        orig_ = 1 * self._set_interco_status
        try:
            self._aux_affect_object_int(
                values,
                "interco status",
                self.n_interco,
                self.name_interco,
                np.arange(self.n_interco),
                self._set_interco_status,
                max_val=1,
            )
            self._modif_interco_set_status = True
        except Exception as exc_:
            self._aux_affect_object_int(
                orig_,
                "interco status",
                self.n_interco,
                self.name_interco,
                np.arange(self.n_interco),
                self._set_interco_status,
                max_val=1,
            )
            raise IllegalAction(
                f"Impossible to modify the interco status with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def interco_change_status(self) -> np.ndarray:
        """
        Property to set the status of the powerline.

        It behave similarly than :attr:`BaseAction.gen_change_bus` but with the following convention:

        * ``False`` will not affect the powerline
        * ``True`` will change the status of the powerline. If it was connected, it will attempt to
          disconnect it, if it was disconnected, it will attempt to reconnect it.

        """
        res = copy.deepcopy(self._switch_interco_status)
        res.flags.writeable = False
        return res
                
    @interco_change_status.setter
    def interco_change_status(self, values):
        if "change_interco_status" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the status of interconnections (with "change") with this action type.'
            )
        orig_ = 1 * self._switch_interco_status
        try:
            self._aux_affect_object_bool(
                values,
                "interco status",
                self.n_interco,
                self.name_interco,
                np.arange(self.n_interco),
                self._switch_interco_status,
            )
            self._modif_interco_change_status = True
        except Exception as exc_:
            self._switch_interco_status[:] = orig_
            raise IllegalAction(
                f"Impossible to modify the interco status with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    def _digest_set_interco_status(self, dict_):
        if "set_interco_status" in dict_:
            # this action can both disconnect or reconnect an interconnection
            self.interco_set_status = dict_["set_interco_status"]

    def _digest_change_interco_status(self, dict_):
        if "change_interco_status" in dict_:
            if dict_["change_interco_status"] is not None:
                self.interco_change_status = dict_["change_interco_status"]
