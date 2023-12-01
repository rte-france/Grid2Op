# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import warnings
import numpy as np
from typing import Tuple

from grid2op.dtypes import dt_bool, dt_int
from grid2op.Exceptions import IllegalAction, AmbiguousAction, Grid2OpException
from grid2op.Action import PlayableAction, ActionSpace, BaseAction
from grid2op.multi_agent.subGridObjects import SubGridObjects


# TODO (later) make that a meta class too
class SubGridActionSpace(SubGridObjects, ActionSpace):
    INTERCO_SET_BUS_ID = ActionSpace.RAISE_ALARM_ID + 1
    INTERCO_CHANGE_BUS_ID = ActionSpace.RAISE_ALARM_ID + 2
    
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
        act_cls = self.actionClass
        if self.n_line > 0: #TODO interco v0.1
            if "set_line_status" in act_cls.authorized_keys:
                rnd_types.append(cls.SET_STATUS_ID)
            if "change_line_status" in act_cls.authorized_keys:
                rnd_types.append(cls.CHANGE_STATUS_ID)
        
        if "set_bus" in act_cls.authorized_keys:
            rnd_types.append(cls.SET_BUS_ID)
        if "change_bus" in act_cls.authorized_keys:
            rnd_types.append(cls.CHANGE_BUS_ID)
        
        if self.n_gen > 0 and (self.gen_redispatchable).any():
            if "redispatch" in act_cls.authorized_keys:
                rnd_types.append(cls.REDISPATCHING_ID)
                
        if self.n_storage > 0 and "storage_power" in act_cls.authorized_keys:
            rnd_types.append(cls.STORAGE_POWER_ID)
            
        if self.dim_alarms > 0 and "raise_alarm" in act_cls.authorized_keys:
            rnd_types.append(cls.RAISE_ALARM_ID)
        
        if self.n_interco and "change_interco_status" in act_cls.authorized_keys:
            rnd_types.append(cls.INTERCO_CHANGE_BUS_ID)
            
        if self.n_interco and "set_interco_status" in act_cls.authorized_keys:
            rnd_types.append(cls.INTERCO_SET_BUS_ID)
        return rnd_types

    def _sample_interco_change_bus(self, rnd_update=None):
        if rnd_update is None:
            rnd_update = {}
        rnd_line = self.space_prng.randint(self.n_interco)
        rnd_update["change_interco_status"] = [rnd_line]
        return rnd_update

    def _sample_interco_set_bus(self, rnd_update=None):
        if rnd_update is None:
            rnd_update = {}
        rnd_line = self.space_prng.randint(self.n_interco)
        rnd_status = self.space_prng.choice([1, -1])
        rnd_update["set_interco_status"] = [(rnd_line, rnd_status)]
        return rnd_update
    
    def _aux_sample_other_element(self, rnd_type) -> dict:
        # sample other elements for subclass if needed
        rnd_update = None
        if rnd_type == self.INTERCO_SET_BUS_ID:
            rnd_update = self._sample_interco_set_bus()
        elif rnd_type == self.INTERCO_CHANGE_BUS_ID:
            rnd_update = self._sample_interco_change_bus()
        return rnd_update
    
    @staticmethod
    def _aux_get_powerline_id(action_space, sub_id_):
        cls = type(action_space)
        ids = super(ActionSpace, action_space)._aux_get_powerline_id(action_space, sub_id_)
        interco_id = action_space.interco_to_sub_pos[
            action_space.interco_to_subid == sub_id_
        ]
        ids = np.concatenate((ids, interco_id))
        return ids
        
    def from_global(self, global_action: BaseAction):
        """Convert the global action to a local action handled by this space

        Parameters
        ----------
        global_action : BaseAction
            The global action

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        """
        
        # TODO not tested
        my_cls = type(self)
        local_action = self()
        
        if global_action._modif_set_bus:
            if self.supports_type("set_bus"):
                tmp_ = global_action._set_topo_vect[my_cls.mask_orig_pos_topo_vect]
                if np.any(tmp_ != 0):
                    local_action._modif_set_bus = True
                    local_action._set_topo_vect = tmp_
            else:
                warnings.warn("The set_bus part of this global action has been removed because "
                              "the target action type does not suppor it")

        if global_action._modif_change_bus:        
            if self.supports_type("change_bus"):
                tmp_ = global_action._change_bus_vect[my_cls.mask_orig_pos_topo_vect]
                if np.any(tmp_):
                    local_action._modif_change_bus = True
                    local_action._change_bus_vect = tmp_
            else:
                warnings.warn("The change_bus part of this global action has been removed because "
                              "the target action type does not suppor it")
        
        if global_action._modif_set_status:
            if self.supports_type("set_line_status"):
                tmp_ = global_action._set_line_status[my_cls.mask_line]
                if np.any(tmp_ != 0):
                    # regular lines (in the local grid) have been modified
                    local_action._modif_set_status = True
                    local_action._set_line_status = tmp_
                
                tmp_ = global_action._set_line_status[my_cls.mask_interco]
                if np.any(tmp_ != 0):
                    # interco (in the local grid) have been modified
                    local_action._modif_interco_set_status = True
                    local_action._set_interco_status = tmp_
            else:
                warnings.warn("The set_line_status part of this global action has been removed because "
                              "the target action type does not suppor it")
       
        if global_action._modif_change_status:     
            if self.supports_type("change_line_status"):
                tmp_ = global_action._switch_line_status[my_cls.mask_line]
                if np.any(tmp_):
                    # regular lines (in the local grid) have been modified
                    local_action._modif_change_status = True
                    local_action._switch_line_status = tmp_
                
                tmp_ = global_action._switch_line_status[my_cls.mask_interco]
                if np.any(tmp_):
                    # interco (in the local grid) have been modified
                    local_action._modif_interco_change_status = True
                    local_action._switch_line_status = tmp_
            else:
                warnings.warn("The change_line_status part of this global action has been removed because "
                              "the target action type does not suppor it")
        
        if global_action._modif_redispatch:
            if self.supports_type("redispatch"):
                tmp_ = global_action._redispatch[my_cls.gen_orig_ids]
                if np.any(tmp_ != 0.):
                    local_action._modif_redispatch = True
                    local_action._redispatch = tmp_
            else:
                warnings.warn("The redispatch part of this global action has been removed because "
                              "the target action type does not suppor it")
        
        if global_action._modif_storage:
            if self.supports_type("storage_power"):
                tmp_ = global_action._storage_power[my_cls.storage_orig_ids]
                if np.any(tmp_ != 0.):
                    local_action._modif_storage = True
                    local_action._storage_power = tmp_
            else:
                warnings.warn("The storage_power part of this global action has been removed because "
                              "the target action type does not suppor it")
        
        if global_action._modif_curtailment:
            if self.supports_type("curtail"):
                tmp_ = global_action._curtail[my_cls.gen_orig_ids]
                if np.any(tmp_ != -1.):
                    local_action._modif_curtailment = True
                    local_action._curtail = global_action._curtail[my_cls.gen_orig_ids]
            else:
                warnings.warn("The curtail part of this global action has been removed because "
                              "the target action type does not suppor it")
                
        if global_action._modif_inj:
            raise NotImplementedError("What to do if global_action modified an injection ?")
        if global_action._modif_alarm:
            raise NotImplementedError("What to do if global_action modified an alarm ?")
        
        return local_action    
        
        
# TODO (later) make that a "metaclass" with argument the ActionType (here playable action)
# TODO run the extensive grid2op test for the action for this class
class SubGridAction(SubGridObjects, PlayableAction):
    
    # TODO make the "PlayableAction" type generic !
    authorized_keys = copy.deepcopy(PlayableAction.authorized_keys)
    authorized_keys.add('change_interco_status')
    authorized_keys.add('set_interco_status')
    
    attr_list_vect = copy.deepcopy(PlayableAction.attr_list_vect)
    attr_list_vect.append("_set_interco_status")
    attr_list_vect.append("_switch_interco_status")
    
    attr_list_set = set(attr_list_vect)
        
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
    
    def get_topological_impact(self):
        # TODO
        return super().get_topological_impact()
    
    def as_serializable_dict(self) -> dict:
        # TODO
        return super().as_serializable_dict()
    
    def __iadd__(self, other):
        raise NotImplementedError("You are not supposed to add local action together. But maybe add local action to global action !")
        super().__iadd__(other)
        # TODO
        return self
    
    def as_dict(self) -> dict:
        # TODO
        return super().as_dict()    
    
    def get_types(self) -> Tuple[bool, bool, bool, bool, bool, bool, bool]:
        # TODO
        res = super().get_types()
        # TODO (And i got a real problem here !)
        return res
    
    def _check_for_ambiguity(self):
        # TODO required for version 0.1
        super()._check_for_ambiguity()
        raise NotImplementedError("There are some ways for this type of action to be ambiguous, we did not think about it yet !")
    
    def can_affect_something(self) -> bool:
        # TODO not tested
        res = super().can_affect_something()
        res = res or self._modif_interco_set_status or self._modif_interco_change_status 
        return res
    
    def _reset_modified_flags(self):
        # TODO not tested
        super()._reset_modified_flags()
        self._modif_interco_set_status = False
        self._modif_interco_change_status = False
        
    def _aux_copy(self, other):
        # TODO not tested
        super()._aux_copy(other)
        attr_simple = [
            "_modif_interco_set_status",
            "_modif_interco_change_status"
        ]
        attr_vect = [
            "_set_interco_status",
            "_switch_interco_status"
        ]
        self._aux_aux_copy(other, attr_simple, attr_vect)
    
    def to_global(self, global_action_space: ActionSpace):
        """Convert the local action to a global action using the provided action space

        Parameters
        ----------
        global_action_space : ActionSpace
            The action space

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        """
        my_cls = type(self)
        global_action = global_action_space()
        # We check for every action type if there's a change
        # if it is the case, we take local changes and copy 
        # them in the corresponding global positions for that
        # action type via mask_orig_pos_topo_vect.
        if self._modif_set_bus:
            if global_action_space.supports_type("set_bus"):
                global_action._modif_set_bus = True
                global_action._set_topo_vect[my_cls.mask_orig_pos_topo_vect] = self._set_topo_vect
            else:
                warnings.warn("The set_bus part of this local action has been removed because "
                              "the target action type does not suppor it")
        
        if self._modif_change_bus:
            if global_action_space.supports_type("change_bus"):
                global_action._modif_change_bus = True
                global_action._change_bus_vect[my_cls.mask_orig_pos_topo_vect] = self._change_bus_vect
            else:
                warnings.warn("The change_bus part of this local action has been removed because "
                              "the target action type does not suppor it")
            
        if self._modif_set_status:
            if global_action_space.supports_type("set_line_status"):
                global_action._modif_set_status = True
                global_action._set_line_status[my_cls.line_orig_ids] = self._set_line_status
            else:
                warnings.warn("The set_line_status part of this local action has been removed because "
                              "the target action type does not suppor it")
            
        if self._modif_change_status:
            if global_action_space.supports_type("change_line_status"):
                global_action._modif_change_status = True
                global_action._switch_line_status[my_cls.line_orig_ids] = self._switch_line_status
            else:
                warnings.warn("The change_line_status part of this local action has been removed because "
                              "the target action type does not suppor it")
        
        if self._modif_redispatch:
            if global_action_space.supports_type("redispatch"):
                global_action._modif_redispatch = True
                global_action._redispatch[my_cls.gen_orig_ids] = self._redispatch
            else:
                warnings.warn("The redispatch part of this local action has been removed because "
                              "the target action type does not suppor it")
        
        if self._modif_storage:
            if global_action_space.supports_type("storage_power"):
                global_action._modif_storage = True
                global_action._storage_power[my_cls.storage_orig_ids] = self._storage_power
            else:
                warnings.warn("The storage_power part of this local action has been removed because "
                              "the target action type does not suppor it")
        
        if self._modif_curtailment:
            if global_action_space.supports_type("curtail"):
                global_action._modif_curtailment = True
                global_action._curtail[my_cls.gen_orig_ids] = self._curtail
            else:
                warnings.warn("The curtail part of this local action has been removed because "
                              "the target action type does not suppor it")
        
        if self._modif_interco_set_status:
            raise NotImplementedError("What to do if I modified an interco status (set) ?")
        if self._modif_interco_change_status:
            raise NotImplementedError("What to do if I modified an interco status (change) ?")
        if self._modif_inj:
            raise NotImplementedError("What to do if I modified an injection ?")
        if self._modif_alarm:
            raise NotImplementedError("What to do if I modified an alarm ?")
        return global_action
        
        
    def impact_on_objects(self) -> dict:
        # TODO not tested
        res = super().impact_on_objects()
        
        force_interco_status = {
            "changed": False,
            "reconnections": {"count": 0, "intercos": []},
            "disconnections": {"count": 0, "intercos": []},
        }
        if np.any(self._set_interco_status == 1):
            force_interco_status["changed"] = True
            res["has_impact"] = True
            force_interco_status["reconnections"]["count"] = np.sum(
                self._set_interco_status == 1
            )
            force_interco_status["reconnections"]["intercos"] = np.where(
                self._set_interco_status == 1
            )[0]

        if np.any(self._set_interco_status == -1):
            force_interco_status["changed"] = True
            res["has_impact"] = True
            force_interco_status["disconnections"]["count"] = np.sum(
                self._set_interco_status == -1
            )
            force_interco_status["disconnections"]["intercos"] = np.where(
                self._set_interco_status == -1
            )[0]

        # handles action on swtich line status
        switch_interco_status = {"changed": False, "count": 0, "intercos": []}
        if np.sum(self._switch_interco_status):
            res["has_impact"] = True
            switch_interco_status["changed"] = True
            switch_interco_status["count"] = np.sum(self._switch_interco_status)
            switch_interco_status["intercos"] = np.where(self._switch_interco_status)[0]
        
        res["force_interco"] = force_interco_status
        res["switch_interco"] = switch_interco_status
        return res
        
    def _str_for_other_elements(self, current_li_str, impact):
        # TODO not tested        
        if not self._modif_interco_change_status:
            current_li_str.append("\t - NOT change the status of any interconnections")
        else:
            swith_interco_impact = impact["switch_interco"]
            current_li_str.append(
                "\t - Switch status of {} intercos ({})".format(
                    swith_interco_impact["count"], swith_interco_impact["intercos"]
                )
            )
        
        if not self._modif_interco_set_status:
            current_li_str.append("\t - NOT force the status of any interconnections")
        else:
            force_interco = impact["force_interco"]
            reconnections = force_interco["reconnections"]
            if reconnections["count"] > 0:
                current_li_str.append(
                    "\t - Force reconnection of {} intercos ({})".format(
                        reconnections["count"], reconnections["intercos"]
                    )
                )
            disconnections = force_interco["disconnections"]
            if disconnections["count"] > 0:
                current_li_str.append(
                    "\t - Force disconnection of {} intercos ({})".format(
                        disconnections["count"], disconnections["intercos"]
                    )
                )
 
    def effect_on(
        self,
        _sentinel=None,
        load_id=None,
        gen_id=None,
        line_id=None,
        substation_id=None,
        storage_id=None,
        interco_id=None
    ) -> dict:
        # TODO not tested
        
        EXCEPT_TOO_MUCH_ELEMENTS = (
            "You can only the inspect the effect of an action on one single element"
        )
        if interco_id is None:
            if (load_id is not None
                or gen_id is not None
                or line_id is not None
                or storage_id is not None
                or substation_id is not None
                ):
                raise Grid2OpException(EXCEPT_TOO_MUCH_ELEMENTS)
            return super().effect_on(_sentinel, load_id, gen_id, line_id, substation_id, storage_id)  
        else:
            return self._aux_effect_on_interco(interco_id)
    
    def _aux_effect_on_interco(self, interco_id):
        # TODO not tested
        if interco_id >= self.n_interco:
            raise Grid2OpException(
                f"There are only {self.n_interco} interco on the grid. Cannot check impact on "
                f"`interco_id={interco_id}`"
            )
        if interco_id < 0:
            raise Grid2OpException(f"`interco_id` should be positive.")
        
        res = {}
        # origin topology
        my_id = self.line_or_pos_topo_vect[interco_id]
        res["change_bus"] = self._change_bus_vect[my_id]
        res["set_bus"] = self._set_topo_vect[my_id]
        res["set_line_status"] = self._set_interco_status[my_id]
        res["change_line_status"] = self._switch_interco_status[my_id]
        return res
        
    def _post_process_from_vect(self):
        # TODO not tested
        super()._post_process_from_vect()
        self._modif_interco_set_status = np.any(self._set_interco_status != 0)
        self._modif_interco_change_status = np.any(self._switch_interco_status)
    
    def _dont_affect_topology(self) -> bool:
        # TODO not tested
        res = super()._dont_affect_topology()
        res = (res and 
               (not self._modif_interco_set_status) and 
               (not self._modif_interco_change_status)
              )
        return res
    
    def __eq__(self, other) -> bool:
        # TODO not tested
        res = super().__eq__(other)
        if not res:
            return False
        
        if (self._modif_interco_set_status != other._modif_interco_set_status) or not np.all(
            self._set_interco_status == other._set_interco_status
        ):
            return False        
        
        if (self._modif_interco_change_status != other._modif_interco_change_status) or not np.all(
            self._switch_interco_status == other._switch_interco_status
        ):
            return False        
        
        return True
    
    def reset(self):
        # TODO not tested
        super().reset()
        self._set_interco_status[:] = 0
        self._switch_interco_status[:] = False
    
    def _check_for_correct_modif_flags(self):
        # TODO not tested
        super()._check_for_correct_modif_flags()        
        if np.any(self._set_interco_status != 0):
            if not self._modif_interco_set_status:
                raise AmbiguousAction(
                    "A action of type interco_set_status is performed while the appropriate flag is not "
                    "set. Please use the official grid2op action API to modify the status of "
                    "interco using "
                    "'set'."
                )
            if "set_interco_status" not in self.authorized_keys:
                raise IllegalAction(
                    "You illegally act on the interco status (using set)"
                )

        if np.any(self._switch_interco_status):
            if not self._modif_interco_change_status:
                raise AmbiguousAction(
                    "A action of type interco_change_status is performed while the appropriate flag "
                    "is not "
                    "set. Please use the official grid2op action API to modify the status of "
                    "interco using 'change'."
                )
            if "change_interco_status" not in self.authorized_keys:
                raise IllegalAction(
                    "You illegally act on the interco status (using change)"
                )
        
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
