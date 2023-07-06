# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import numpy as np
import itertools
from typing import Dict, List

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import AmbiguousAction, Grid2OpException
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

    SET_STATUS_ID = 0
    CHANGE_STATUS_ID = 1
    SET_BUS_ID = 2
    CHANGE_BUS_ID = 3
    REDISPATCHING_ID = 4
    STORAGE_POWER_ID = 5
    RAISE_ALARM_ID = 6
    RAISE_ALERT_ID = 7

    ERR_MSG_WRONG_TYPE = ('The action to update using `ActionSpace` is of type "{}" '
                         '"which is not the type of action handled by this action space "'
                         '("{}")')
    
    def __init__(self, gridobj, actionClass=BaseAction, _init_grid=True):
        """
        INTERNAL USE ONLY

         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

           The :class:`grid2op.Environment.Environment` is responsible for the creation of the
           action space. Do not attempt to make one yourself.

        Parameters
        ----------
        gridobj: :class:`grid2op.Space.GridObjects`
            Representation of the underlying powergrid.

        actionClass: ``type``
            Type of action used to build :attr:`Space.SerializableSpace._template_obj`. It should derived from
            :class:`BaseAction`.

        """
        SerializableSpace.__init__(
            self, gridobj=gridobj, subtype=actionClass, _init_grid=_init_grid
        )
        self.actionClass = self.subtype
        self._template_act = self.actionClass()

    @staticmethod
    def from_dict(dict_):
        """
        INTERNAL USE ONLY

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

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
        CLS = SerializableActionSpace.init_grid(tmp)
        res = CLS(gridobj=tmp, actionClass=tmp.subtype, _init_grid=False)
        return res

    def _get_possible_action_types(self):
        rnd_types = []
        cls = type(self)
        if "set_line_status" in self.actionClass.authorized_keys:
            rnd_types.append(cls.SET_STATUS_ID)
        if "change_line_status" in self.actionClass.authorized_keys:
            rnd_types.append(cls.CHANGE_STATUS_ID)
        if "set_bus" in self.actionClass.authorized_keys:
            rnd_types.append(cls.SET_BUS_ID)
        if "change_bus" in self.actionClass.authorized_keys:
            rnd_types.append(cls.CHANGE_BUS_ID)
        if "redispatch" in self.actionClass.authorized_keys:
            rnd_types.append(cls.REDISPATCHING_ID)
        if self.n_storage > 0 and "storage_power" in self.actionClass.authorized_keys:
            rnd_types.append(cls.STORAGE_POWER_ID)
        if self.dim_alarms > 0 and "raise_alarm" in self.actionClass.authorized_keys:
            rnd_types.append(cls.RAISE_ALARM_ID)
        if self.dim_alerts > 0 and "raise_alert" in self.actionClass.authorized_keys:
            rnd_types.append(cls.RAISE_ALERT_ID)
        return rnd_types

    def supports_type(self, action_type):
        """
        Returns if the current action_space supports the current action type.

        Parameters
        ----------
        action_type: ``str``
            One of "set_line_status", "change_line_status", "set_bus", "change_bus", "redispatch",
            "storage_power", "set_storage", "curtail" or "curtail_mw"
            A string representing the action types you want to inspect.

        Returns
        -------
        ``True`` if you can use the `action_type` to create an action, ``False`` otherwise.

        Examples
        ---------

        To know if you can use the `act.set_bus` property to change the bus of an element, you can use:

        .. code-block:: python

            import grid2op
            from grid2op.Converter import ConnectivityConverter

            env = grid2op.make("rte_case14_realistic", test=True)
            can_i_use_set_bus = env.action_space.supports_type("set_bus") # this is True

            env2 = grid2op.make("educ_case14_storage", test=True)
            can_i_use_set_bus = env2.action_space.supports_type("set_bus") # this is False
            # this environment do not allow for topological changes but only action on storage units and redispatching

        """
        name_action_types = [
            "set_line_status",
            "change_line_status",
            "set_bus",
            "change_bus",
            "redispatch",
            "storage_power",
            "set_storage",
            "curtail",
            "curtail_mw",
            "raise_alarm",
            "raise_alert"
        ]
        assert action_type in name_action_types, (
            f"The action type provided should be in {name_action_types}. "
            f"You provided {action_type} which is not supported."
        )

        if action_type == "storage_power":
            return (self.n_storage > 0) and (
                "set_storage" in self.actionClass.authorized_keys
            )
        elif action_type == "set_storage":
            return (self.n_storage > 0) and (
                "set_storage" in self.actionClass.authorized_keys
            )
        elif action_type == "curtail_mw":
            return "curtail" in self.actionClass.authorized_keys
        else:
            return action_type in self.actionClass.authorized_keys

    def _sample_set_line_status(self, rnd_update=None):
        if rnd_update is None:
            rnd_update = {}
        rnd_line = self.space_prng.randint(self.n_line)
        rnd_status = self.space_prng.choice([1, -1])
        rnd_update["set_line_status"] = [(rnd_line, rnd_status)]
        return rnd_update

    def _sample_change_line_status(self, rnd_update=None):
        if rnd_update is None:
            rnd_update = {}
        rnd_line = self.space_prng.randint(self.n_line)
        rnd_update["change_line_status"] = [rnd_line]
        return rnd_update

    def _sample_set_bus(self, rnd_update=None):
        if rnd_update is None:
            rnd_update = {}
        rnd_sub = self.space_prng.randint(self.n_sub)
        sub_size = self.sub_info[rnd_sub]
        rnd_topo = self.space_prng.choice([-1, 0, 1, 2], sub_size)
        rnd_update["set_bus"] = {"substations_id": [(rnd_sub, rnd_topo)]}
        return rnd_update

    def _sample_change_bus(self, rnd_update=None):
        if rnd_update is None:
            rnd_update = {}
        rnd_sub = self.space_prng.randint(self.n_sub)
        sub_size = self.sub_info[rnd_sub]
        rnd_topo = self.space_prng.choice([0, 1], sub_size).astype(dt_bool)
        rnd_update["change_bus"] = {"substations_id": [(rnd_sub, rnd_topo)]}
        return rnd_update

    def _sample_redispatch(self, rnd_update=None):
        if rnd_update is None:
            rnd_update = {}
        gens = np.arange(self.n_gen)[self.gen_redispatchable]
        rnd_gen = self.space_prng.choice(gens)
        rd = -self.gen_max_ramp_down[rnd_gen]
        ru = self.gen_max_ramp_up[rnd_gen]
        rnd_gen_disp = (ru - rd) * self.space_prng.random() + rd
        rnd_disp = np.zeros(self.n_gen)
        rnd_disp[rnd_gen] = rnd_gen_disp
        rnd_update["redispatch"] = rnd_disp
        rnd_update["redispatch"] = rnd_update["redispatch"].astype(dt_float)
        return rnd_update

    def _sample_storage_power(self, rnd_update=None):
        if rnd_update is None:
            rnd_update = {}
        stor_unit = np.arange(self.n_storage)
        rnd_sto = self.space_prng.choice(stor_unit)
        rd = -self.storage_max_p_prod[rnd_sto]
        ru = self.storage_max_p_absorb[rnd_sto]
        rnd_sto_prod = (ru - rd) * self.space_prng.random() + rd
        res = np.zeros(self.n_gen)
        res[rnd_sto] = rnd_sto_prod
        rnd_update["storage_power"] = res
        rnd_update["storage_power"] = rnd_update["storage_power"].astype(dt_float)
        return rnd_update

    def _sample_raise_alarm(self, rnd_update=None):
        """.. warning:: 
            /!\\\\ Only valid with "l2rpn_icaps_2021" environment /!\\\\
        """
        if rnd_update is None:
            rnd_update = {}
        rnd_area = self.space_prng.randint(self.dim_alarms)
        rnd_update["raise_alarm"] = [rnd_area]
        return rnd_update

    def _sample_raise_alert(self, rnd_update=None):
        if rnd_update is None:
            rnd_update = {}
        rnd_alerted_lines = self.space_prng.choice([True, False], self.dim_alerts).astype(dt_bool)
        rnd_update["raise_alert"] = rnd_alerted_lines
        return rnd_update

    def sample(self):
        """
        A utility used to sample a new random :class:`BaseAction`.

        The sampled action is unitary: It has an impact on a single line/substation/generator.

        There is no guarantee concerning the "legality" of the action (see the description of the
        Action module for more information about illegal action).

        It will only act by doing action supported by the action space. For example, if the action space
        does not support "redispatching" then this method will NOT sample any redispatching action.

        Returns
        -------
        res: :class:`BaseAction`
            A random action sampled from the :attr:`ActionSpace.actionClass`

        Examples
        ---------
        The first usage is to sample uniform **unary** actions, you can do this with the
        following:

        .. code-block:: python

            import grid2op
            env = grid2op.make()

            # and now you can sample from the action space
            random_action = env.action_space.sample()

        *Note* that the random action can be illegal depending on the game rules defined in the
        rules :class:`grid2op.Rules`

        If for some reason you want to sample more complex actions, you can do this the following way:

        .. code-block:: python

            import grid2op
            env = grid2op.make()

            # and now you can sample from the action space
            random_action = env.action_space()
            for i in range(5):
                # my resulting action will be a complex action
                # that will be the results of applying 5 random actions
                random_action += env.action_space.sample()
            print(random_action)

        """
        rnd_act = self.actionClass()

        # get the type of actions I am allowed to perform
        rnd_types = self._get_possible_action_types()

        # Cannot sample this space, return do nothing
        if not len(rnd_types):
            return rnd_act

        # this sampling
        rnd_type = self.space_prng.choice(rnd_types)

        if rnd_type == self.SET_STATUS_ID:
            rnd_update = self._sample_set_line_status()
        elif rnd_type == self.CHANGE_STATUS_ID:
            rnd_update = self._sample_change_line_status()
        elif rnd_type == self.SET_BUS_ID:
            rnd_update = self._sample_set_bus()
        elif rnd_type == self.CHANGE_BUS_ID:
            rnd_update = self._sample_change_bus()
        elif rnd_type == self.REDISPATCHING_ID:
            rnd_update = self._sample_redispatch()
        elif rnd_type == self.STORAGE_POWER_ID:
            rnd_update = self._sample_storage_power()
        elif rnd_type == self.RAISE_ALARM_ID:
            rnd_update = self._sample_raise_alarm()
        elif rnd_type == self.RAISE_ALERT_ID:
            rnd_update = self._sample_raise_alert()
        else:
            raise Grid2OpException(
                "Impossible to sample action of type {}".format(rnd_type)
            )

        rnd_act.update(rnd_update)
        return rnd_act

    def disconnect_powerline(self, line_id=None, line_name=None, previous_action=None):
        """
        Utilities to disconnect a powerline more easily.

        Parameters
        ----------
        line_id: ``int``
            The powerline to be disconnected.

        line_name: ``str``
            Name of the powerline. Note that either line_id or line_name should be provided. If both are provided, it is
            an error, if none are provided it is an error.

        previous_action: :class:`BaseAction`
            If you want to stack up multiple actions.

        Returns
        -------
        res: :class:`BaseAction`
            The action that will disconnect the powerline.

        Notes
        ------
        If you use `previous_action` it will modify the action **in place** which means that
        `previous_action` will be modified by this method.

        Examples
        ---------
        You can use it this way:

        .. code-block:: python

            import grid2op
            env = grid2op.make()

            # and now you can disconnect line 0
            disco_line_0 = env.action_space.disconnect_powerline(line_id=0)

            # or line with name "0_4_1"
            disco_line_1 = env.action_space.disconnect_powerline(line_name="0_4_1")

            # and you can disconnect both line 2 and 3 with:
            disco_line_2 = env.action_space.disconnect_powerline(line_id=2)
            disco_line_2_and_3 = env.action_space.disconnect_powerline(line_id=3, previous_action=disco_line_2)
            print(disco_line_2_and_3)
            # be careful, "disco_line_2" is affected and is in fact equal to "disco_line_2_and_3"
            # after the last call!

        """
        if line_id is None and line_name is None:
            raise AmbiguousAction(
                'You need to provide either the "line_id" or the "line_name" of the powerline '
                "you want to disconnect"
            )
        if line_id is not None and line_name is not None:
            raise AmbiguousAction(
                'You need to provide only of the "line_id" or the "line_name" of the powerline '
                "you want to disconnect"
            )

        if line_id is None:
            line_id = np.where(self.name_line == line_name)[0]
            if not len(line_id):
                raise AmbiguousAction(
                    'Line with name "{}" is not on the grid. The powerlines names are:\n{}'
                    "".format(line_name, self.name_line)
                )
        if previous_action is None:
            res = self.actionClass()
        else:
            if not isinstance(previous_action, self.actionClass):
                raise AmbiguousAction(
                    type(self).ERR_MSG_WRONG_TYPE.format(type(previous_action), self.actionClass)
                )
            res = previous_action
        if line_id > self.n_line:
            raise AmbiguousAction(
                "You asked to disconnect powerline of id {} but this id does not exist. The "
                "grid counts only {} powerline".format(line_id, self.n_line)
            )
        res.update({"set_line_status": [(line_id, -1)]})
        return res

    def reconnect_powerline(
        self, bus_or, bus_ex, line_id=None, line_name=None, previous_action=None
    ):
        """
        Utilities to reconnect a powerline more easily.

        Note that in case "bus_or" or "bus_ex" are not the current bus to which the powerline is connected, they
        will be affected by this action.

        Notes
        ------
        This utility requires you to specify on which bus you want to connect each end
        ("*origin*" or "*extremity*") of the powerline you want to reconnect.

        If you don't want to specify them, you can set them to ``0`` and it will reconnect them
        to the last known buses to which they were connected (this is automatically done by the
        Environment since version `0.8.0`).

        If you use `previous_action` it will modify the action **in place** which means that
        `previous_action` will be modified by this method.

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
        res: :class:`BaseAction`
            The action that will reconnect the powerline.

        Examples
        ---------
        You can use it this way:

        .. code-block:: python

            import grid2op
            env = grid2op.make()

            # and now you can reconnect line 0
            reco_line_0 = env.action_space.reconnect_powerline(line_id=0, bus_or=1, bus_ex=0)

            # or line with name "0_4_1" to bus 1 on its "origin" end and bus 2 on its "extremity" end
            reco_line_1 = env.action_space.reconnect_powerline(line_name="0_4_1", bus_or=1, bus_ex=2)

            # and you can reconnect both line 2 and 3 with:
            reco_line_2 = env.action_space.reconnect_powerline(line_id=2, bus_or=1, bus_ex=2)
            reco_line_2_and_3 = env.action_space.reconnect_powerline(line_id=3,
                                                                    bus_or=0, bus_ex=1,
                                                                    previous_action=reco_line_2)
            print(reco_line_2_and_3)
            # be careful, "reco_line_2" is affected and is in fact equal to "reco_line_2_and_3"
            # after the last call!

        """
        if line_id is None and line_name is None:
            raise AmbiguousAction(
                'You need to provide either the "line_id" or the "line_name" of the powerline '
                "you want to reconnect"
            )
        if line_id is not None and line_name is not None:
            raise AmbiguousAction(
                'You need to provide only of the "line_id" or the "line_name" of the powerline '
                "you want to reconnect"
            )

        if line_id is None:
            line_id = np.where(self.name_line == line_name)[0]

        if previous_action is None:
            res = self.actionClass()
        else:
            if not isinstance(previous_action, self.actionClass):
                raise AmbiguousAction(
                    type(self).ERR_MSG_WRONG_TYPE.format(type(previous_action), self.actionClass)
                )
            res = previous_action
        if line_id > self.n_line:
            raise AmbiguousAction(
                "You asked to disconnect powerline of id {} but this id does not exist. The "
                "grid counts only {} powerline".format(line_id, self.n_line)
            )
        res.update(
            {
                "set_line_status": [(line_id, 1)],
                "set_bus": {
                    "lines_or_id": [(line_id, bus_or)],
                    "lines_ex_id": [(line_id, bus_ex)],
                },
            }
        )
        return res

    def change_bus(
        self,
        name_element,
        extremity=None,
        substation=None,
        type_element=None,
        previous_action=None,
    ):
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
        type_element: ``str``, optional
            Type of the element to look for. It is here to speed up the computation. One of "line", "gen" or "load"
        previous_action: :class:`Action`, optional
            The (optional) action to update. It should be of the same type as :attr:`ActionSpace.actionClass`

        Notes
        ------
        If you use `previous_action` it will modify the action **in place** which means that
        `previous_action` will be modified by this method.

        Returns
        -------
        res: :class:`BaseAction`
            The action with the modification implemented

        Raises
        ------
        res :class:`grid2op.Exception.AmbiguousAction`
            If *previous_action* has not the same type as :attr:`ActionSpace.actionClass`.

        Examples
        ---------
        You can use it this way:

        .. code-block:: python

            import grid2op
            env = grid2op.make()

            # change bus of element named 'gen_1_0'
            change_gen_0 = env.action_space.change_bus('gen_1_0', type_element="gen")

            # you are not forced to specify the element types
            change_load_1 = env.action_space.change_bus('load_2_1')

            # dealing with powerline, you can affect one of its extremity
            # (handy when you don't know on which substation it is located)
            change_line_8_or = env.action_space.change_bus('5_11_8', extremity="or")

            # and you can combine the action with
            change_line_14_ex = env.action_space.change_bus('12_13_14', extremity="ex")
            change_line_14_ex_load_2 = env.action_space.change_bus("load_3_2",
                                                                   previous_action=change_line_14_ex)
            print(change_line_14_ex_load_2)
            # be careful, "change_line_14_ex" is affected and is in fact equal to
            # "change_line_14_ex_load_2"
            # after the last call!

        """
        if previous_action is None:
            res = self.actionClass()
        else:
            if not isinstance(previous_action, self.actionClass):
                raise AmbiguousAction(
                    type(self).ERR_MSG_WRONG_TYPE.format(type(previous_action), self.actionClass)
                )
            res = previous_action

        dict_, to_sub_pos, my_id, my_sub_id = self._extract_dict_action(
            name_element, extremity, substation, type_element, res
        )
        arr_ = dict_["change_bus"]
        me_id_ = to_sub_pos[my_id]
        arr_[me_id_] = True
        res.update({"change_bus": {"substations_id": [(my_sub_id, arr_)]}})
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
            raise Grid2OpException(
                "It is mandatory to know on which ends you want to change the bus of the powerline"
            )
        else:
            raise Grid2OpException(
                'unknown extremity specifier "{}". Extremity should be "or" or "ex"'
                "".format(extremity)
            )
        return to_subid, to_sub_pos, to_name

    def _extract_dict_action(
        self,
        name_element,
        extremity=None,
        substation=None,
        type_element=None,
        action=None,
    ):
        to_subid = None
        to_sub_pos = None
        to_name = None

        if type_element is None:
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
                to_subid, to_sub_pos, to_name = self._extract_database_powerline(
                    extremity
                )
            else:
                AmbiguousAction(
                    'Element "{}" not found in the powergrid'.format(name_element)
                )
        elif type_element == "line":
            to_subid, to_sub_pos, to_name = self._extract_database_powerline(extremity)
        elif type_element[:3] == "gen" or type_element[:4] == "prod":
            to_subid = self.gen_to_subid
            to_sub_pos = self.gen_to_sub_pos
            to_name = self.name_gen
        elif type_element == "load":
            to_subid = self.load_to_subid
            to_sub_pos = self.load_to_sub_pos
            to_name = self.name_load
        else:
            raise AmbiguousAction(
                'unknown type_element specifier "{}". type_element should be "line" or "load" '
                'or "gen"'.format(extremity)
            )

        my_id = None
        for i, nm in enumerate(to_name):
            if nm == name_element:
                my_id = i
                break
        if my_id is None:
            raise AmbiguousAction(
                'Element "{}" not found in the powergrid'.format(name_element)
            )
        my_sub_id = to_subid[my_id]

        dict_ = action.effect_on(substation_id=my_sub_id)
        return dict_, to_sub_pos, my_id, my_sub_id

    def set_bus(
        self,
        name_element,
        new_bus,
        extremity=None,
        substation=None,
        type_element=None,
        previous_action=None,
    ):
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

        Examples
        ---------
        You can use it this way:

        .. code-block:: python

            import grid2op
            env = grid2op.make()

            # set bus of element named 'gen_1_0' to bus 2
            setbus_gen_0 = env.action_space.set_bus('gen_1_0', new_bus=2, type_element="gen")

            # are not forced to specify the element types (example with load set to bus 1)
            setbus_load_1 = env.action_space.set_bus('load_2_1', new_bus=1)

            # dealing with powerline, you can affect one of its extremity
            # (handy when you don't know on which substation it is located)
            setbus_line_8_or = env.action_space.set_bus('5_11_8', new_bus=1, extremity="or")

            # and you can combine the actions with:
            setbus_line_14_ex = env.action_space.set_bus('12_13_14', new_bus=2, extremity="ex")
            setbus_line_14_ex_load_2 = env.action_space.set_bus("load_3_2", new_bus=1,
                                                                previous_action=setbus_line_14_ex)
            print(setbus_line_14_ex_load_2)
            # be careful, "setbus_line_14_ex" is affected and is in fact equal to
            # "setbus_line_14_ex_load_2"
            # after the last call!

        """
        if previous_action is None:
            res = self.actionClass()
        else:
            res = previous_action

        dict_, to_sub_pos, my_id, my_sub_id = self._extract_dict_action(
            name_element, extremity, substation, type_element, res
        )
        dict_["set_bus"][to_sub_pos[my_id]] = new_bus
        res.update({"set_bus": {"substations_id": [(my_sub_id, dict_["set_bus"])]}})
        return res

    def get_set_line_status_vect(self):
        """
        Computes and returns a vector that can be used in the "set_status" keyword if building an :class:`BaseAction`

        Returns
        -------
        res: :class:`numpy.array`, dtype:dt_int
            A vector that doesn't affect the grid, but can be used in "set_line_status"

        """
        return self._template_act.get_set_line_status_vect()

    def get_change_line_status_vect(self):
        """
        Computes and return a vector that can be used in the "change_line_status" keyword if building an :class:`BaseAction`

        Returns
        -------
        res: :class:`numpy.array`, dtype:dt_bool
            A vector that doesn't affect the grid, but can be used in "change_line_status"

        """
        return self._template_act.get_change_line_status_vect()

    @staticmethod
    def get_all_unitary_line_set(action_space):
        """
        Return all unitary actions that "set" powerline status.

        For each powerline, there are 5 such actions:

        - disconnect it
        - connected it origin at bus 1 and extremity at bus 1
        - connected it origin at bus 1 and extremity at bus 2
        - connected it origin at bus 2 and extremity at bus 1
        - connected it origin at bus 2 and extremity at bus 2

        Parameters
        ----------
        action_space: :class:`grid2op.BaseAction.ActionSpace`
            The action space used.

        Returns
        -------
        res: ``list``
            The list of all "set" action acting on powerline status

        """
        res = []

        # powerline switch: disconnection
        for i in range(action_space.n_line):
            res.append(action_space.disconnect_powerline(line_id=i))

        # powerline switch: reconnection
        for bus_or in [1, 2]:
            for bus_ex in [1, 2]:
                for i in range(action_space.n_line):
                    act = action_space.reconnect_powerline(
                        line_id=i, bus_ex=bus_ex, bus_or=bus_or
                    )
                    res.append(act)

        return res

    @staticmethod
    def get_all_unitary_line_set_simple(action_space):
        """
        Return all unitary actions that "set" powerline status but in a 
        more simple way than :func:`SerializableActionSpace.get_all_unitary_line_set`

        For each powerline, there are 2 such actions:

        - disconnect it
        - connected it (implicitly to the last known busbar where each
          side used to be connected)

        It has the main advantages to "only" add 2 actions per powerline
        instead of 5.
        
        
        Parameters
        ----------
        action_space: :class:`grid2op.BaseAction.ActionSpace`
            The action space used.

        Returns
        -------
        res: ``list``
            The list of all "set" action acting on powerline status

        """
        res = []

        # powerline set: disconnection
        for i in range(action_space.n_line):
            res.append(action_space({"set_line_status": [(i,-1)]}))
        
        # powerline set: reconnection   
        for i in range(action_space.n_line):
            res.append(action_space({"set_line_status": [(i, +1)]}))

        return res
        
    @staticmethod
    def get_all_unitary_alarm(action_space):
        """
        .. warning::
            /!\\\\ Only valid with "l2rpn_icaps_2021" environment /!\\\\
        """
        res = []
        for i in range(action_space.dim_alarms):
            status = np.full(action_space.dim_alarms, fill_value=False, dtype=dt_bool)
            status[i] = True
            res.append(action_space({"raise_alarm": status}))
        return res

    @staticmethod
    def get_all_unitary_alert(action_space):
        """
        Return all unitary actions that raise an alert on powerlines.
        
        .. warning:: There will be one action per combination of attackable lines, so basically, if 
           you can raise alerts on 10 powerline, you will end up with 2**10 actions.
           
           If you got 22 attackable lines, then you got 2**22 actions... probably a TERRIBLE IDEA !
        """
        res = []
        possible_values = [False, True]
        if action_space.dim_alerts:
            for status in itertools.product(possible_values, repeat=type(action_space).dim_alerts):
                res.append(action_space({"raise_alert": np.array(status, dtype=dt_bool)}))
        return res

    @staticmethod
    def get_all_unitary_line_change(action_space):
        """
        Return all unitary actions that "change" powerline status.

        For each powerline, there is only one such action that consist in change its status.

        Parameters
        ----------
        action_space: :class:`grid2op.BaseAction.ActionSpace`
            The action space used.

        Returns
        -------
        res: ``list``
            The list of all "change" action acting on powerline status

        """
        res = []
        for i in range(action_space.n_line):
            status = action_space.get_change_line_status_vect()
            status[i] = True
            res.append(action_space({"change_line_status": status}))
        return res

    @staticmethod
    def get_all_unitary_topologies_change(action_space, sub_id=None):
        """
        This methods allows to compute and return all the unitary topological changes that can be performed on a
        powergrid.

        The changes will be performed using the "change_bus" method. It excludes the "do nothing" action

        Parameters
        ----------
        action_space: :class:`grid2op.BaseAction.ActionSpace`
            The action space used.

        sub_id: ``int``, optional
            The substation ID. If ``None`` it is done for all substations.

        Notes
        -----
        This might take a long time on large grid (possibly 10-15 mins for the IEEE 118 for example)

        Returns
        -------
        res: ``list``
            The list of all the topological actions that can be performed.

        Examples
        ---------
        You can use it this way:

        .. code-block:: python

            import grid2op
            env = grid2op.make()

            # all "change bus" action for all the substations
            all_change_actions = env.action_space.get_all_unitary_topologies_change(env.action_space)

            # you can only study "change_bus" action for a given substation (can dramatically improve the computation time)
            all_change_actions_sub4 = env.action_space.get_all_unitary_topologies_change(env.action_space, sub_id=4)

        """
        res = []
        S = [0, 1]
        for sub_id_, num_el in enumerate(action_space.sub_info):
            if sub_id is not None:
                if sub_id_ != sub_id:
                    continue
            already_set = set()
            # remove the "do nothing" action, which is either equivalent to not change anything
            # or to change everything
            already_set.add(tuple([1 for _ in range(num_el)]))
            already_set.add(tuple([0 for _ in range(num_el)]))
            for tup_ in itertools.product(S, repeat=num_el):
                if tup_ not in already_set:
                    indx = np.full(shape=num_el, fill_value=False, dtype=dt_bool)
                    # tup = np.array((0, *tup)).astype(dt_bool)  # add a zero to first element -> break symmetry
                    tup = np.array(tup_).astype(
                        dt_bool
                    )  # add a zero to first element -> break symmetry
                    indx[tup] = True
                    action = action_space(
                        {"change_bus": {"substations_id": [(sub_id_, indx)]}}
                    )
                    already_set.add(tup_)
                    already_set.add(tuple([1 - el for el in tup_]))
                    res.append(action)
                # otherwise, the change has already beend added (NB by symmetry , if there are A, B, C and D in
                # a substation, changing A,B or changing C,D always has the same effect.
        return res

    @staticmethod
    def get_all_unitary_topologies_set(action_space, sub_id=None):
        """
        This methods allows to compute and return all the unitary topological changes that can be performed on a
        powergrid.

        The changes will be performed using the "set_bus" method. The "do nothing" action will be counted once
        per substation in the grid.

        Parameters
        ----------
        action_space: :class:`grid2op.BaseAction.ActionHelper`
            The action space used.

        sub_id: ``int``, optional
            The substation ID. If ``None`` it is done for all substations.

        Notes
        -----
        This might take a long time on large grid (possibly 10-15 mins for the IEEE 118 for example)

        Returns
        -------
        res: ``list``
            The list of all the topological actions that can be performed.

        Examples
        ---------
        You can use it this way:

        .. code-block:: python

            import grid2op
            env = grid2op.make()

            # all "set_bus" actions
            all_change_actions = env.action_space.get_all_unitary_topologies_set(env.action_space)

            # you can only study "set_bus" action for a given substation (can dramatically improve the computation time)
            all_change_actions_sub4 = env.action_space.get_all_unitary_topologies_set(env.action_space, sub_id=4)

        """
        res = []
        S = [0, 1]
        for sub_id_, num_el in enumerate(action_space.sub_info):
            tmp = []
            if sub_id is not None:
                if sub_id_ != sub_id:
                    continue

            new_topo = np.full(shape=num_el, fill_value=1, dtype=dt_int)
            # perform the action "set everything on bus 1"
            action = action_space(
                {"set_bus": {"substations_id": [(sub_id_, new_topo)]}}
            )
            tmp.append(action)

            powerlines_or_id = action_space.line_or_to_sub_pos[
                action_space.line_or_to_subid == sub_id_
            ]
            powerlines_ex_id = action_space.line_ex_to_sub_pos[
                action_space.line_ex_to_subid == sub_id_
            ]
            powerlines_id = np.concatenate((powerlines_or_id, powerlines_ex_id))

            # computes all the topologies at 2 buses for this substation
            for tup in itertools.product(S, repeat=num_el - 1):
                indx = np.full(shape=num_el, fill_value=False, dtype=dt_bool)
                tup = np.array((0, *tup)).astype(
                    dt_bool
                )  # add a zero to first element -> break symmetry
                indx[tup] = True
                if np.sum(indx) >= 2 and np.sum(~indx) >= 2:
                    # i need 2 elements on each bus at least (almost all the times, except when a powerline
                    # is alone on its bus)
                    new_topo = np.full(shape=num_el, fill_value=1, dtype=dt_int)
                    new_topo[~indx] = 2

                    if (
                        np.sum(indx[powerlines_id]) == 0
                        or np.sum(~indx[powerlines_id]) == 0
                    ):
                        # if there is a "node" without a powerline, the topology is not valid
                        continue

                    action = action_space(
                        {"set_bus": {"substations_id": [(sub_id_, new_topo)]}}
                    )
                    tmp.append(action)
                else:
                    # i need to take into account the case where 1 powerline is alone on a bus too
                    if (
                        np.sum(indx[powerlines_id]) >= 1
                        and np.sum(~indx[powerlines_id]) >= 1
                    ):
                        new_topo = np.full(shape=num_el, fill_value=1, dtype=dt_int)
                        new_topo[~indx] = 2
                        action = action_space(
                            {"set_bus": {"substations_id": [(sub_id_, new_topo)]}}
                        )
                        tmp.append(action)

            if len(tmp) >= 2:
                # if i have only one single topology on this substation, it doesn't make any action
                # i cannot change the topology is there is only one.
                res += tmp

        return res

    @staticmethod
    def get_all_unitary_redispatch(
        action_space, num_down=5, num_up=5, max_ratio_value=1.0
    ):
        """
        Redispatching action are continuous action. This method is an helper to convert the continuous
        action into discrete action (by rounding).

        The number of actions is equal to num_down + num_up (by default 10) per dispatchable generator.


        This method acts as followed:

        - it will divide the interval [-gen_max_ramp_down, 0] into `num_down`, each will make
          a distinct action (then counting `num_down` different action, because 0.0 is removed)
        - it will do the same for [0, gen_maw_ramp_up]


        Parameters
        ----------
        action_space: :class:`grid2op.BaseAction.ActionHelper`
            The action space used.

        num_down: ``int``
            In how many intervals the "redispatch down" will be split

        num_up: ``int``
            In how many intervals the "redispatch up" will be split

        max_ratio_value: ``float``
            Expressed in terms of ratio of `gen_max_ramp_up` / `gen_max_ramp_down`, it gives the maximum value
            that will be used to generate the actions. For example, if `max_ratio_value=0.5`, then it will not
            generate actions that attempts to redispatch more than `0.5 * gen_max_ramp_up` (or less than
            `- 0.5 * gen_max_ramp_down`). This helps reducing the instability that can be caused by redispatching.

        Returns
        -------
        res: ``list``
            The list of all discretized redispatching actions.

        """

        res = []
        n_gen = len(action_space.gen_redispatchable)

        for gen_idx in range(n_gen):
            # Skip non-dispatchable generators
            if not action_space.gen_redispatchable[gen_idx]:
                continue

            # Create evenly spaced positive interval
            ramps_up = np.linspace(
                0.0, max_ratio_value * action_space.gen_max_ramp_up[gen_idx], num=num_up
            )
            ramps_up = ramps_up[1:]  # Exclude redispatch of 0MW

            # Create evenly spaced negative interval
            ramps_down = np.linspace(
                -max_ratio_value * action_space.gen_max_ramp_down[gen_idx],
                0.0,
                num=num_down,
            )
            ramps_down = ramps_down[:-1]  # Exclude redispatch of 0MW

            # Merge intervals
            ramps = np.append(ramps_up, ramps_down)

            # Create ramp up actions
            for ramp in ramps:
                action = action_space({"redispatch": [(gen_idx, ramp)]})
                res.append(action)

        return res

    @staticmethod
    def get_all_unitary_curtail(action_space, num_bin=10, min_value=0.5):
        """
        Curtailment action are continuous action. This method is an helper to convert the continuous
        action into discrete action (by rounding).

        The number of actions is equal to num_bin (by default 10) per renewable generator
        (remember that only renewable generator can be curtailed in grid2op).


        This method acts as followed:

        - it will divide the interval [0, 1] into `num_bin`, each will make
          a distinct action (then counting `num_bin` different action, because 0.0 is removed)


        Parameters
        ----------
        action_space: :class:`grid2op.BaseAction.ActionHelper`
            The action space used.

        num_bin: ``int``
            Number of actions for each renewable generator

        min_value: ``float``
            Between 0. and 1.: minimum value allow for the curtailment. FOr example if you set this
            value to be 0.2 then no curtailment will be done to limit the generator below 20% of its maximum capacity

        Returns
        -------
        res: ``list``
            The list of all discretized curtailment actions.
        """

        res = []
        n_gen = len(action_space.gen_renewable)

        for gen_idx in range(n_gen):
            # Skip non-renewable generators (they cannot be curtail)
            if not action_space.gen_renewable[gen_idx]:
                continue
            # Create evenly spaced interval
            ramps = np.linspace(min_value, 1.0, num=num_bin)

            # Create ramp up actions
            for ramp in ramps:
                action = action_space({"curtail": [(gen_idx, ramp)]})
                res.append(action)

        return res

    @staticmethod
    def get_all_unitary_storage(action_space, num_down=5, num_up=5):
        """
        Storage action are continuous action. This method is an helper to convert the continuous
        action into discrete action (by rounding).

        The number of actions is equal to num_down + num_up (by default 10) per storage unit.


        This method acts as followed:

        - it will divide the interval [-storage_max_p_prod, 0] into `num_down`, each will make
          a distinct action (then counting `num_down` different action, because 0.0 is removed)
        - it will do the same for [0, storage_max_p_absorb]


        Parameters
        ----------
        action_space: :class:`grid2op.BaseAction.ActionHelper`
            The action space used.

        Returns
        -------
        res: ``list``
            The list of all discretized actions on storage units.

        """

        res = []
        n_stor = action_space.n_storage

        for stor_idx in range(n_stor):

            # Create evenly spaced positive interval
            ramps_up = np.linspace(
                0.0, action_space.storage_max_p_absorb[stor_idx], num=num_up
            )
            ramps_up = ramps_up[1:]  # Exclude action of 0MW

            # Create evenly spaced negative interval
            ramps_down = np.linspace(
                -action_space.storage_max_p_prod[stor_idx], 0.0, num=num_down
            )
            ramps_down = ramps_down[:-1]  # Exclude action of 0MW

            # Merge intervals
            ramps = np.append(ramps_up, ramps_down)

            # Create ramp up actions
            for ramp in ramps:
                action = action_space({"set_storage": [(stor_idx, ramp)]})
                res.append(action)

        return res

    def _custom_deepcopy_for_copy(self, new_obj):
        super()._custom_deepcopy_for_copy(new_obj)
        # SerializableObservationSpace
        new_obj.actionClass = self.subtype
        new_obj._template_act = self.actionClass()

    def _aux_get_back_to_ref_state_curtail(self, res, obs):
        is_curtailed = obs.curtailment_limit != 1.0
        if np.any(is_curtailed):
            res["curtailment"] = []
            if not self.supports_type("curtail"):
                warnings.warn(
                    "A generator is is curtailed, but you cannot perform curtailment action. Impossible to get back to the original topology."
                )
                return

            curtail = np.full(obs.n_gen, fill_value=np.NaN)
            curtail[is_curtailed] = 1.0
            act = self.actionClass()
            act.curtail = curtail
            res["curtailment"].append(act)

    def _aux_get_back_to_ref_state_line(self, res, obs):
        disc_lines = ~obs.line_status
        if np.any(disc_lines):
            li_disc = np.where(disc_lines)[0]
            res["powerline"] = []
            for el in li_disc:
                act = self.actionClass()
                if self.supports_type("set_line_status"):
                    act.set_line_status = [(el, +1)]
                elif self.supports_type("change_line_status"):
                    act.change_line_status = [el]
                else:
                    warnings.warn(
                        "A powerline is disconnected by you cannot reconnect it with your action space. Impossible to get back to the original topology"
                    )
                    break
                res["powerline"].append(act)

    def _aux_get_back_to_ref_state_sub(self, res, obs):
        not_on_bus_1 = obs.topo_vect > 1  # disconnected lines are handled above
        if np.any(not_on_bus_1):
            res["substation"] = []
            subs_changed = type(self).grid_objects_types[
                not_on_bus_1, type(self).SUB_COL
            ]
            for sub_id in set(subs_changed):
                nb_el: int = type(self).sub_info[sub_id]
                act = self.actionClass()
                if self.supports_type("set_bus"):
                    act.sub_set_bus = [(sub_id, np.ones(nb_el, dtype=dt_int))]
                elif self.supports_type("change_bus"):
                    arr_ = np.full(nb_el, fill_value=False, dtype=dt_bool)
                    changed = obs.state_of(substation_id=sub_id)["topo_vect"] >= 1
                    arr_[changed] = True
                    act.sub_change_bus = [(sub_id, arr_)]
                else:
                    warnings.warn(
                        "A substation is is not on its original topology (all at bus 1) and your action type does not allow to change it. "
                        "Impossible to get back to the original topology."
                    )
                    break
                res["substation"].append(act)

    def _aux_get_back_to_ref_state_redisp(self, res, obs, precision=1e-5):
        # TODO this is ugly, probably slow and could definitely be optimized
        notredisp_setpoint = obs.target_dispatch != 0.0
        if np.any(notredisp_setpoint):
            need_redisp = np.where(notredisp_setpoint)[0]
            res["redispatching"] = []
            # combine generators and do not exceed ramps (up or down)
            rem = np.zeros(self.n_gen, dtype=dt_float)
            nb_ = np.zeros(self.n_gen, dtype=dt_int)
            for gen_id in need_redisp:
                if obs.target_dispatch[gen_id] > 0.0:
                    div_ = obs.target_dispatch[gen_id] / obs.gen_max_ramp_down[gen_id]
                else:
                    div_ = -obs.target_dispatch[gen_id] / obs.gen_max_ramp_up[gen_id]
                div_ = np.round(div_, precision)
                nb_[gen_id] = int(div_)
                if div_ != int(div_):
                    if obs.target_dispatch[gen_id] > 0.0:
                        rem[gen_id] = (
                            obs.target_dispatch[gen_id]
                            - obs.gen_max_ramp_down[gen_id] * nb_[gen_id]
                        )
                    else:
                        rem[gen_id] = (
                            -obs.target_dispatch[gen_id]
                            - obs.gen_max_ramp_up[gen_id] * nb_[gen_id]
                        )
                    nb_[gen_id] += 1
            # now create the proper actions
            for nb_act in range(np.max(nb_)):
                act = self.actionClass()
                if not self.supports_type("redispatch"):
                    warnings.warn(
                        "Some redispatching are set, but you cannot modify it with your action type. Impossible to get back to the original topology."
                    )
                    break
                reds = np.zeros(self.n_gen, dtype=dt_float)
                for gen_id in need_redisp:
                    if nb_act >= nb_[gen_id]:
                        # nothing to add for this generator in this case
                        continue
                    if obs.target_dispatch[gen_id] > 0.0:
                        if nb_act < nb_[gen_id] - 1 or (
                            rem[gen_id] == 0.0 and nb_act == nb_[gen_id] - 1
                        ):
                            reds[gen_id] = -obs.gen_max_ramp_down[gen_id]
                        else:
                            reds[gen_id] = -rem[gen_id]
                    else:
                        if nb_act < nb_[gen_id] - 1 or (
                            rem[gen_id] == 0.0 and nb_act == nb_[gen_id] - 1
                        ):
                            reds[gen_id] = obs.gen_max_ramp_up[gen_id]
                        else:
                            reds[gen_id] = rem[gen_id]

                act.redispatch = [
                    (gen_id, red_) for gen_id, red_ in zip(need_redisp, reds)
                ]
                res["redispatching"].append(act)

    def _aux_get_back_to_ref_state_storage(
        self, res, obs, storage_setpoint, precision=5
    ):
        # TODO this is ugly, probably slow and could definitely be optimized
        # TODO refacto with the redispatching
        notredisp_setpoint = obs.storage_charge / obs.storage_Emax != storage_setpoint
        delta_time_hour = dt_float(obs.delta_time / 60.0)
        if np.any(notredisp_setpoint):
            need_ajust = np.where(notredisp_setpoint)[0]
            res["storage"] = []
            # combine storage units and do not exceed maximum power
            rem = np.zeros(self.n_storage, dtype=dt_float)
            nb_ = np.zeros(self.n_storage, dtype=dt_int)
            current_state = obs.storage_charge - storage_setpoint * obs.storage_Emax
            for stor_id in need_ajust:
                if current_state[stor_id] > 0.0:
                    div_ = current_state[stor_id] / (
                        obs.storage_max_p_prod[stor_id] * delta_time_hour
                    )
                else:
                    div_ = -current_state[stor_id] / (
                        obs.storage_max_p_absorb[stor_id] * delta_time_hour
                    )
                div_ = np.round(div_, precision)
                nb_[stor_id] = int(div_)
                if div_ != int(div_):
                    if current_state[stor_id] > 0.0:
                        rem[stor_id] = (
                            current_state[stor_id] / delta_time_hour
                            - obs.storage_max_p_prod[stor_id] * nb_[stor_id]
                        )
                    else:
                        rem[stor_id] = (
                            -current_state[stor_id] / delta_time_hour
                            - obs.storage_max_p_absorb[stor_id] * nb_[stor_id]
                        )
                    nb_[stor_id] += 1

            # now create the proper actions
            for nb_act in range(np.max(nb_)):
                act = self.actionClass()
                if not self.supports_type("set_storage"):
                    warnings.warn(
                        "Some storage are modififed, but you cannot modify them with your action type. Impossible to get back to the original topology."
                    )
                    break
                reds = np.zeros(self.n_storage, dtype=dt_float)
                for stor_id in need_ajust:
                    if nb_act >= nb_[stor_id]:
                        # nothing to add in this case
                        continue
                    if current_state[stor_id] > 0.0:
                        if nb_act < nb_[stor_id] - 1 or (
                            rem[stor_id] == 0.0 and nb_act == nb_[stor_id] - 1
                        ):
                            reds[stor_id] = -obs.storage_max_p_prod[stor_id]
                        else:
                            reds[stor_id] = -rem[stor_id]
                    else:
                        if nb_act < nb_[stor_id] - 1 or (
                            rem[stor_id] == 0.0 and nb_act == nb_[stor_id] - 1
                        ):
                            reds[stor_id] = obs.storage_max_p_absorb[stor_id]
                        else:
                            reds[stor_id] = rem[stor_id]

                act.storage_p = [
                    (stor_id, red_) for stor_id, red_ in zip(need_ajust, reds)
                ]
                res["storage"].append(act)

    def get_back_to_ref_state(
        self,
        obs: "grid2op.Observation.BaseObservation",
        storage_setpoint=0.5,
        precision=5,
    ) -> Dict[str, List[BaseAction]]:
        """
        This function returns the list of unary actions that you can perform in order to get back to the "fully meshed" / "initial" topology.

        Parameters
        ----------
        observation:
            The current observation (the one you want to know actions to set it back ot)
        Notes
        -----
        In this context a "unary" action, is (exclusive or):

        - an action that acts on a single powerline
        - an action on a single substation
        - a redispatching action
        - a storage action

        The list might be relatively long, in the case where lots of actions are needed. Depending on the rules of the game (for example limiting the
        action on one single substation), in order to get back to this topology, multiple consecutive actions will need to be implemented.

        It is returned as a dictionnary of list. This dictionnary has 4 keys:

        - "powerline" for the list of actions needed to set back the powerlines in a proper state (connected). They can be of type "change_line" or "set_line".
        - "substation" for the list of actions needed to set back each substation in its initial state (everything connected to bus 1). They can be
          implemented as "set_bus" or "change_bus"
        - "redispatching": for the redispatching action (there can be multiple redispatching actions needed because of the ramps of the generator)
        - "storage": for action on storage units (you might need to perform multiple storage actions because of the maximum power these units can absorb / produce )
        - "curtailment": for curtailment action (usually at most one such action is needed)

        After receiving these lists, the agent has the choice for the order in which to apply these actions as well as how to best combine them (you can most
        of the time combine action of different types in grid2op.)

        .. warning::

            It does not presume anything on the availability of the objects. For examples, this funciton ignores completely the cooldowns on lines and substations.

        .. warning::

            For the storage units, it tries to set their current setpoint to `storage_setpoint` % of their storage total capacity. Applying these actions
            at different times might not fully set back the storage to this capacity in case of storage losses !

        .. warning::

            See section :ref:`action_powerline_status` for note on the powerline status. It might happen that you modify a powerline status using a "set_bus" (ie
            tagged as "susbtation" by this function).

        .. warning::

            It can raise warnings in case it's not possible, with your action space, to get back to the original / fully meshed topology

        Examples
        --------

        TODO

        """
        from grid2op.Observation.baseObservation import BaseObservation

        if not isinstance(obs, BaseObservation):
            raise AmbiguousAction(
                "You need to provide a grid2op Observation for this function to work correctly."
            )
        res = {}

        # powerline actions
        self._aux_get_back_to_ref_state_line(res, obs)
        # substations
        self._aux_get_back_to_ref_state_sub(res, obs)
        # redispatching
        self._aux_get_back_to_ref_state_redisp(res, obs, precision=precision)
        # storage
        self._aux_get_back_to_ref_state_storage(
            res, obs, storage_setpoint, precision=precision
        )
        # curtailment
        self._aux_get_back_to_ref_state_curtail(res, obs)

        return res
