# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import numpy as np

from grid2op.Converter.Converters import Converter
from grid2op.dtypes import dt_int


def IdToActSingleElement(substation_id, element_pos):
    """
    Returns the corresponding converter for the actions on the element defined by its id and type.

    :id: int, the id of the element
    :element_type: str, the kind of element to look for
                        can be 'line_or', 'line_ex', 'gen' or 'load'
    """

    class IdToActSingleElementConverter(Converter):
        """
        This type of converter allows to represent disconnection/connection actions of a single element with unique id. Instead of manipulating complex objects, it allows
        to manipulate only positive integer.
    
        **NB** The actions that are initialized by default uses the "set" way and not the "change" way (see the description
        of :class:`grid2op.BaseAction.BaseAction` for more information).
    
        In this converter:
    
        - `encoded_act` is a positive integer, representing the index of the action.
        - `transformed_obs` is a regular observations.
        """
        def __init__(self, action_space):
            Converter.__init__(self, action_space)
            self.__class__ = IdToActSingleElementConverter.init_grid(action_space)

            self.substation_id = substation_id
            self.element_pos = element_pos
            self.actions = []
            # add the do nothing topology
            self.actions.append(super().__call__())
            self.n = 1
    
        def init_converter(self):
            """
            This function is used ato initialize the converter. When the converter is created, this method should be called
            otherwise the converter might be in an unstable state.
            """
            self.actions = []
            num_el = self.sub_info[self.substation_id]
            indx = np.full(shape=num_el, fill_value=0, dtype=dt_int)
            for bus in [-1, 1, 2]: # disconnected, bus 1, bus 2
                indx[self.element_pos] = bus
                action = self({"set_bus": {"substations_id": [(self.substation_id, indx)]}})
                self.actions.append(action)

            self.n = len(self.actions)
    
        def convert_act(self, encoded_act):
            """
            In this converter, we suppose that "encoded_act" is an id of an action stored in the
            actions list.
    
            Converting an id of an action (here called "encoded_act") into a valid action is then easy:
            we just need to take the "encoded_act"-th element of all_actions.
    
            Parameters
            ----------
            encoded_act: ``int``
                The id of the action
    
            Returns
            -------
            action: :class:`grid2op.Action.Action`
                The action corresponding to id "encoded_act"
            """
    
            return self.actions[encoded_act]

    cls_name = f'IdToActSingleElement_{substation_id}_{element_pos}'
    IdToActSingleElementConverter.__name__ = cls_name
    IdToActSingleElementConverter.__qualname__ = cls_name

    return IdToActSingleElementConverter
