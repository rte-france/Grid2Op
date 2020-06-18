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


class LineDisconnection(Converter):
    """
    This type of converter allows to represent disconnection of lines with unique id. Instead of manipulating complex objects, it allows
    to manipulate only positive integer.

    **NB** The actions that are initialized by default uses the "set" way and not the "change" way (see the description
    of :class:`grid2op.BaseAction.BaseAction` for more information).

    In this converter:

    - `encoded_act` is a positive integer, representing the index of the action.
    - `transformed_obs` is a regular observations.
    """
    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.__class__ = LineDisconnection.init_grid(action_space)

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
        # powerline switch: disconnection
        self.actions.append(super().__call__())
        for i in range(self.n_line):
            self.actions.append(self.disconnect_powerline(line_id=i))
    
        self.n = len(self.actions)

    def filter_lines(self, requested_lines):
        """
        This function allows you to "easily" filter generated actions.

        **NB** the action space will change after a call to this function, especially its size. It is NOT recommended
        to apply it once training has started.

        Parameters
        ----------
        filtering_fun: ``function``
            This takes an action as input and should retrieve ``True`` meaning "this action will be kept" or
            ``False`` meaning "this action will be dropped.

        """
        ids = np.argwhere(np.in1d(self.name_line, requested_lines))
        self.actions = np.array([el for el in self.actions
                                    if not el.as_dict()
                                    or el.as_dict()['set_line_status']['disconnected_id'][0] in ids])
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
