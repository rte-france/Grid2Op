# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import warnings
import itertools

import pdb

from grid2op.Exceptions import *
from grid2op.Space import SerializableSpace
from grid2op.Action.BaseAction import BaseAction


class SerializableActionSpace(SerializableSpace):
    """
    This class allows serializing/ deserializing the action space.

    It should not be used inside an :attr:`grid2op.Environment.Environment` , as some functions of the action might not
    be compatible with the serialization, especially the checking of whether or not an action is legal or not.

    Attributes
    ----------

    actionClass: ``type``
        Type used to build the :attr:`SerializableActionSpace.template_act`

    _template_act: :class:`BaseAction`
        An instance of the "*actionClass*" provided used to provide higher level utilities, such as the size of the
        action (see :func:`Action.size`) or to sample a new Action (see :func:`grid2op.Action.Action.sample`)

    """
    def __init__(self, gridobj, actionClass=BaseAction):
        """

        Parameters
        ----------
        gridobj: :class:`grid2op.Space.GridObjects`
            Representation of the underlying powergrid.

        actionClass: ``type``
            Type of action used to build :attr:`Space.SerializableSpace._template_obj`. It should derived from
            :class:`BaseAction`.

        """
        SerializableSpace.__init__(self, gridobj=gridobj, subtype=actionClass)

        self.actionClass = self.subtype
        self._template_act = self._template_obj

    @staticmethod
    def from_dict(dict_):
        """
        Allows the de-serialization of an object stored as a dictionary (for example in the case of JSON saving).

        Parameters
        ----------
        dict_: ``dict``
            Representation of an BaseAction Space (aka SerializableActionSpace) as a dictionary.

        Returns
        -------
        res: :class:``SerializableActionSpace``
            An instance of an action space matching the dictionary.


        """
        tmp = SerializableSpace.from_dict(dict_)
        res = SerializableActionSpace(gridobj=tmp, actionClass=tmp.subtype)
        return res

    def sample(self):
        """
        A utility used to sample :class:`Action`.

        This method is under development, use with care (actions are not sampled on the full action space, and are
        not uniform in general).

        Returns
        -------
        res: :class:`BaseAction`
            A random action sampled from the :attr:`ActionSpace.actionClass`

        """
        res = self.actionClass(gridobj=self)  # only the GridObjects part of "self" is actually used
        res.sample()
        return res

    def disconnect_powerline(self, line_id, previous_action=None):
        """
        Utilities to disconnect a powerline more easily.

        Parameters
        ----------
        line_id: ``int``
            The powerline to be disconnected.

        previous_action

        Returns
        -------

        """
        if previous_action is None:
            res = self.actionClass(gridobj=self)
        else:
            if not isinstance(previous_action, self.actionClass):
                raise AmbiguousAction("The action to update using `ActionSpace` is of type \"{}\" "
                                      "which is not the type of action handled by this helper "
                                      "(\"{}\")".format(type(previous_action), self.actionClass))
            res = previous_action
        if line_id > self.n_line:
            raise AmbiguousAction("You asked to disconnect powerline of id {} but this id does not exist. The "
                                  "grid counts only {} powerline".format(line_id, self.n_line))
        res.update({"set_line_status": [(line_id, -1)]})
        return res

    def reconnect_powerline(self, line_id, bus_or, bus_ex, previous_action=None):
        """
        Utilities to reconnect a powerline more easily.

        Note that in case "bus_or" or "bus_ex" are not the current bus to which the powerline is connected, they
        will be affected by this action.

        Parameters
        ----------
        line_id: ``int``
            The powerline to be disconnected.

        bus_or: ``int``
            On which bus to reconnect the powerline at its origin end

        bus_ex: ``int``
            On which bus to reconnect the powerline at its extremity end
        previous_action

        Returns
        -------

        """
        if previous_action is None:
            res = self.actionClass(gridobj=self)
        else:
            if not isinstance(previous_action, self.actionClass):
                raise AmbiguousAction("The action to update using `ActionSpace` is of type \"{}\" "
                                      "which is not the type of action handled by this helper "
                                      "(\"{}\")".format(type(previous_action), self.actionClass))
            res = previous_action
        if line_id > self.n_line:
            raise AmbiguousAction("You asked to disconnect powerline of id {} but this id does not exist. The "
                                  "grid counts only {} powerline".format(line_id, self.n_line))
        res.update({"set_line_status": [(line_id, 1)],
                    "set_bus": {"lines_or_id": [(line_id, bus_or)],
                                "lines_ex_id": [(line_id, bus_ex)]}})
        return res

    def change_bus(self, name_element, extremity=None, substation=None, type_element=None, previous_action=None):
        """
        Utilities to change the bus of a single element if you give its name. **NB** Changing a bus has the effect to
        assign the object to bus 1 if it was before that connected to bus 2, and to assign it to bus 2 if it was
        connected to bus 1. It should not be mixed up with :func:`ActionSpace.set_bus`.

        If the parameter "*previous_action*" is not ``None``, then the action given to it is updated (in place) and
        returned.

        Parameters
        ----------
        name_element: ``str``
            The name of the element you want to change the bus
        extremity: ``str``
            "or" or "ex" for origin or extremity, ignored if an element is not a powerline.
        substation: ``int``, optional
            Its substation ID, if you know it will increase the performance. Otherwise, the method will search for it.
        type_element: ``int``, optional
            Type of the element to look for. It is here to speed up the computation. One of "line", "gen" or "load"
        previous_action: :class:`Action`, optional
            The (optional) action to update. It should be of the same type as :attr:`ActionSpace.actionClass`

        Returns
        -------
        res: :class:`BaseAction`
            The action with the modification implemented

        Raises
        ------
        :class:`grid2op.Exception.AmbiguousAction`
            If *previous_action* has not the same type as :attr:`ActionSpace.actionClass`.

        """
        if previous_action is None:
            res = self.actionClass(gridobj=self)
        else:
            if not isinstance(previous_action, self.actionClass):
                raise AmbiguousAction("The action to update using `ActionSpace` is of type \"{}\" "
                                      "which is not the type of action handled by this helper "
                                      "(\"{}\")".format(type(previous_action), self.actionClass))
            res = previous_action

        dict_, to_sub_pos, my_id, my_sub_id = self._extract_dict_action(name_element, extremity, substation,
                                                                        type_element, res)
        dict_["change_bus"][to_sub_pos[my_id]] = True
        res.update({"change_bus": {"substations_id": [(my_sub_id, dict_["change_bus"])]}})
        # res.update(dict_)
        return res

    def _extract_database_powerline(self, extremity):
        if extremity[:2] == "or":
            to_subid = self.line_or_to_subid
            to_sub_pos = self.line_or_to_sub_pos
            to_name = self.name_line
        elif extremity[:2] == "ex":
            to_subid = self.line_ex_to_subid
            to_sub_pos = self.line_ex_to_sub_pos
            to_name = self.name_line
        elif extremity is None:
            raise Grid2OpException("It is mandatory to know on which ends you want to change the bus of the powerline")
        else:
            raise Grid2OpException("unknown extremity specifier \"{}\". Extremity should be \"or\" or \"ex\""
                                   "".format(extremity))
        return to_subid, to_sub_pos, to_name

    def _extract_dict_action(self, name_element, extremity=None, substation=None, type_element=None, action=None):
        to_subid = None
        to_sub_pos = None
        to_name = None

        if type_element == "line":
            to_subid, to_sub_pos, to_name = self._extract_database_powerline(extremity)
        elif type_element[:3] == "gen" or type_element[:4] == "prod":
            to_subid = self.gen_to_subid
            to_sub_pos = self.gen_to_sub_pos
            to_name = self.name_gen
        elif type_element == "load":
            to_subid = self.load_to_subid
            to_sub_pos = self.load_to_sub_pos
            to_name = self.name_load
        elif type_element is None:
            # i have to look through all the objects to find it
            if name_element in self.name_load:
                to_subid = self.load_to_subid
                to_sub_pos = self.load_to_sub_pos
                to_name = self.name_load
            elif name_element in self.name_gen:
                to_subid = self.gen_to_subid
                to_sub_pos = self.gen_to_sub_pos
                to_name = self.name_gen
            elif name_element in self.name_line:
                to_subid, to_sub_pos, to_name = self._extract_database_powerline(extremity)
            else:
                AmbiguousAction(
                    "Element \"{}\" not found in the powergrid".format(
                        name_element))
        else:
            raise AmbiguousAction("unknown type_element specifier \"{}\". type_element should be \"line\" or \"load\" "
                                  "or \"gen\"".format(extremity))

        my_id = None
        for i, nm in enumerate(to_name):
            if nm == name_element:
                my_id = i
                break
        if my_id is None:
            raise AmbiguousAction("Element \"{}\" not found in the powergrid".format(name_element))
        my_sub_id = to_subid[my_id]

        dict_ = action.effect_on(substation_id=my_sub_id)
        return dict_, to_sub_pos, my_id, my_sub_id

    def set_bus(self, name_element, new_bus, extremity=None, substation=None, type_element=None, previous_action=None):
        """
        Utilities to set the bus of a single element if you give its name. **NB** Setting a bus has the effect to
        assign the object to this bus. If it was before that connected to bus 1, and you assign it to bus 1 (*new_bus*
        = 1) it will stay on bus 1. If it was on bus 2 (and you still assign it to bus 1) it will be moved to bus 2.
        1. It should not be mixed up with :func:`ActionSpace.change_bus`.

        If the parameter "*previous_action*" is not ``None``, then the action given to it is updated (in place) and
        returned.

        Parameters
        ----------
        name_element: ``str``
            The name of the element you want to change the bus

        new_bus: ``int``
            Id of the new bus to connect the object to.

        extremity: ``str``
            "or" or "ext" for origin or extremity, ignored if the element is not a powerline.

        substation: ``int``, optional
            Its substation ID, if you know it will increase the performance. Otherwise, the method will search for it.

        type_element: ``str``, optional
            Type of the element to look for. It is here to speed up the computation. One of "line", "gen" or "load"

        previous_action: :class:`Action`, optional
            The (optional) action to update. It should be of the same type as :attr:`ActionSpace.actionClass`

        Returns
        -------
        res: :class:`BaseAction`
            The action with the modification implemented

        Raises
        ------
        AmbiguousAction
            If *previous_action* has not the same type as :attr:`ActionSpace.actionClass`.

        """
        if previous_action is None:
            res = self.actionClass(gridobj=self)
        else:
            res = previous_action

        dict_, to_sub_pos, my_id, my_sub_id = self._extract_dict_action(name_element, extremity, substation,
                                                                        type_element, res)
        dict_["set_bus"][to_sub_pos[my_id]] = new_bus
        res.update({"set_bus": {"substations_id": [(my_sub_id, dict_["set_bus"])]}})
        return res

    def get_set_line_status_vect(self):
        """
        Computes and returns a vector that can be used in the "set_status" keyword if building an :class:`BaseAction`

        Returns
        -------
        res: :class:`numpy.array`, dtype:np.int
            A vector that doesn't affect the grid, but can be used in "set_line_status"

        """
        return self._template_act.get_set_line_status_vect()

    def get_change_line_status_vect(self):
        """
        Computes and return a vector that can be used in the "change_line_status" keyword if building an :class:`BaseAction`

        Returns
        -------
        res: :class:`numpy.array`, dtype:np.bool
            A vector that doesn't affect the grid, but can be used in "change_line_status"

        """
        return self._template_act.get_change_line_status_vect()

    @staticmethod
    def get_all_unitary_line_set(action_space):
        res = []

        # powerline switch: disconnection
        for i in range(action_space.n_line):
            res.append(action_space.disconnect_powerline(line_id=i))

        # powerline switch: reconnection
        for bus_or in [1, 2]:
            for bus_ex in [1, 2]:
                for i in range(action_space.n_line):
                    act = action_space.reconnect_powerline(line_id=i, bus_ex=bus_ex, bus_or=bus_or)
                    res.append(act)

        return res

    @staticmethod
    def get_all_unitary_line_change(action_space):
        res = []

        for i in range(action_space.n_line):
            status = action_space.get_change_line_status_vect()
            status[i] = True
            res.append(action_space({"change_line_status": status}))

        return res

    @staticmethod
    def get_all_unitary_topologies_change(action_space):
        """
        This methods allows to compute and return all the unitary topological changes that can be performed on a
        powergrid.

        The changes will be performed using the "change_bus" method. The "do nothing" action will be counted only
        once.

        Parameters
        ----------
        action_space: :class:`grid2op.BaseAction.ActionHelper`
            The action space used.

        Returns
        -------
        res: ``list``
            The list of all the topological actions that can be performed.

        """
        res = []
        S = [0, 1]
        for sub_id, num_el in enumerate(action_space.sub_info):
            if num_el < 4:
                pass

            for tup in itertools.product(S, repeat=num_el - 1):
                indx = np.full(shape=num_el, fill_value=False, dtype=np.bool)
                tup = np.array((0, *tup)).astype(np.bool)  # add a zero to first element -> break symmetry
                indx[tup] = True
                if np.sum(indx) >= 2 and np.sum(~indx) >= 2:
                    # i need 2 elements on each bus at least
                    action = action_space({"change_bus": {"substations_id": [(sub_id, indx)]}})
                    res.append(action)
        return res

    @staticmethod
    def get_all_unitary_topologies_set(action_space):
        """
        This methods allows to compute and return all the unitary topological changes that can be performed on a
        powergrid.

        The changes will be performed using the "set_bus" method. The "do nothing" action will be counted once
        per substation in the grid.

        Parameters
        ----------
        action_space: :class:`grid2op.BaseAction.ActionHelper`
            The action space used.

        Returns
        -------
        res: ``list``
            The list of all the topological actions that can be performed.

        """
        res = []
        S = [0, 1]
        for sub_id, num_el in enumerate(action_space.sub_info):
            tmp = []
            new_topo = np.full(shape=num_el, fill_value=1, dtype=np.int)
            # perform the action "set everything on bus 1"
            action = action_space({"set_bus": {"substations_id": [(sub_id, new_topo)]}})
            tmp.append(action)

            powerlines_or_id = action_space.line_or_to_sub_pos[action_space.line_or_to_subid == sub_id]
            powerlines_ex_id = action_space.line_ex_to_sub_pos[action_space.line_ex_to_subid == sub_id]
            powerlines_id = np.concatenate((powerlines_or_id, powerlines_ex_id))

            # computes all the topologies at 2 buses for this substation
            for tup in itertools.product(S, repeat=num_el - 1):
                indx = np.full(shape=num_el, fill_value=False, dtype=np.bool)
                tup = np.array((0, *tup)).astype(np.bool)  # add a zero to first element -> break symmetry
                indx[tup] = True
                if np.sum(indx) >= 2 and np.sum(~indx) >= 2:
                    # i need 2 elements on each bus at least
                    new_topo = np.full(shape=num_el, fill_value=1, dtype=np.int)
                    new_topo[~indx] = 2

                    if np.sum(indx[powerlines_id]) == 0 or np.sum(~indx[powerlines_id]) == 0:
                        # if there is a "node" without a powerline, the topology is not valid
                        continue

                    action = action_space({"set_bus": {"substations_id": [(sub_id, new_topo)]}})
                    tmp.append(action)

            if len(tmp) >= 2:
                # if i have only one single topology on this substation, it doesn't make any action
                # i cannot change the topology is there is only one.
                res += tmp

        return res

    @staticmethod
    def get_all_unitary_redispatch(action_space):
        res = []
        n_gen = len(action_space.gen_redispatchable)

        for gen_idx in range(n_gen):
            # Skip non-dispatchable generators
            if not action_space.gen_redispatchable[gen_idx]:
                continue

            # Create evenly spaced positive interval
            ramps_up = np.linspace(0.0, action_space.gen_max_ramp_up[gen_idx], num=5)
            ramps_up = ramps_up[1:] # Exclude redispatch of 0MW

            # Create evenly spaced negative interval
            ramps_down = np.linspace(-action_space.gen_max_ramp_down[gen_idx], 0.0, num=5)
            ramps_down = ramps_down[:-1] # Exclude redispatch of 0MW

            # Merge intervals
            ramps = np.append(ramps_up, ramps_down)

            # Create ramp up actions
            for ramp in ramps:
                action = action_space({"redispatch": [(gen_idx, ramp)]})
                res.append(action)

        return res
