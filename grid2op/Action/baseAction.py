# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np
import warnings
from typing import Tuple, Dict, Literal, Any, List
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
    
from packaging import version

from grid2op.dtypes import dt_int, dt_bool, dt_float
from grid2op.Exceptions import *
from grid2op.Space import GridObjects

# TODO time delay somewhere (eg action is implemented after xxx timestep, and not at the time where it's proposed)

# TODO have the "reverse" action, that does the opposite of an action. Will be hard but who know ? :eyes:
# TODO ie:  action + (rev_action) = do_nothing_action

# TODO consistency in names gen_p / prod_p and in general gen_* prod_*


class BaseAction(GridObjects):
    """
    This is a base class for each :class:`BaseAction` objects.
    As stated above, an action represents conveniently the modifications that will affect a powergrid.

    It is not recommended to instantiate an action from scratch. The recommended way to get an action is either by
    modifying an existing one using the method :func:`BaseAction.update` or to call and :class:`ActionSpace` object that
    has been properly set up by an :class:`grid2op.Environment`.

    BaseAction can be fully converted to and back from a numpy array with a **fixed** size.

    An action can modify the grid in multiple ways.
    It can change :

    - the production and voltage setpoint of the generator units
    - the amount of power consumed (for both active and reactive part) for load
    - disconnect powerlines
    - change the topology of the _grid.

    To be valid, an action should be convertible to a tuple of 5 elements:

    - the first element is the "injections" vector: representing the way generator units and loads are modified

        - It is, in turn, a dictionary with the following keys (optional)

            - "load_p" a vector of the same size of the load, giving the modification of the loads active consumption
            - "load_q" a vector of the same size of the load, giving the modification of the loads reactive consumption
            - "prod_p" a vector of the same size of the generators, giving the modification of the productions active
              setpoint production
            - "prod_v" a vector of the same size of the generators, giving the modification of the productions voltage
              setpoint

    - the second element is made of force line status. It is made of a vector of size :attr:`BaseAction._n_lines`
      (the number of lines in the powergrid) and is interpreted as:

            - -1 force line disconnection
            - +1 force line reconnection
            - 0 do nothing to this line

    - the third element is the switch line status vector. It is made of a vector of size :attr:`BaseAction.n_line`
      and is
      interpreted as:

        - ``True``: change the line status
        - ``False``: don't do anything

    - the fourth element set the buses to which the object is connected. It's a vector of integers with the following
      interpretation:

        - 0 -> don't change
        - -1 -> disconnect the object.
        - 1 -> connect to bus 1
        - 2 -> connect to bus 2
        - 3 -> connect to bus 3 (added in version 1.10.0)
        - etc.  (added in version 1.10.0)

    - the fifth element changes the buses to which the object is connected. It's a boolean vector interpreted as:

        - ``False``: nothing is done
        - ``True``: change the bus eg connect it to bus 1 if it was connected to bus 2 or connect it to bus 2 if it was
          connected to bus 1. NB this is only active if the system has only 2 buses per substation (that's the case for
          the L2RPN challenge).

    - the sixth element is a vector, representing the redispatching. Component of this vector is added to the
      generators active setpoint value (if set) of the first elements.

    **NB** the difference between :attr:`BaseAction._set_topo_vect` and :attr:`BaseAction._change_bus_vect` is the
    following:

        - If  a component of :attr:`BaseAction._set_topo_vect` is 1, then the object (load, generator or powerline)
          will be moved to bus 1 of the substation to which it is connected. If it is already to bus 1 nothing will be
          done.
          If it's on another bus it will connect it to bus 1. It's disconnected, it will reconnect it and connect it
          to bus 1.
        - If a component of :attr:`BaseAction._change_bus_vect` is True, then the object will be moved from one bus to
          another.
          If the object were on bus 1
          it will be moved on bus 2, and if it were on bus 2, it will be moved on bus 1. If the object were
          disconnected,
          then this does nothing.

    The conversion to the action into an understandable format by the backend is performed by the "update" method,
    that takes into account a dictionary and is responsible to convert it into this format.
    It is possible to overload this class as long as the overloaded :func:`BaseAction.__call__` operator returns the
    specified format, and the :func:`BaseAction.__init__` method has the same signature.

    This format is then digested by the backend and the powergrid is modified accordingly.

    Attributes
    ----------

    _set_line_status: :class:`numpy.ndarray`, dtype:int
        For each powerline, it gives the effect of the action on the status of it. It should be understood as:

          - -1: disconnect the powerline
          - 0: don't affect the powerline
          - +1: reconnect the powerline

    _switch_line_status: :class:`numpy.ndarray`, dtype:bool
        For each powerline, it informs whether the action will switch the status of a powerline of not. It should be
        understood as followed:

          - ``False``: the action doesn't affect the powerline
          - ``True``: the action affects the powerline. If it was connected, it will disconnect it. If it was
            disconnected, it will reconnect it.

    _dict_inj: ``dict``
        Represents the modification of the injection (productions and loads) of the power _grid. This dictionary can
        have the optional keys:

            - "load_p" to set the active load values (this is a numpy array with the same size as the number of load
              in the power _grid with Nan: don't change anything, else set the value
            - "load_q": same as above but for the load reactive values
            - "prod_p": same as above but for the generator active setpoint values. It has the size corresponding
              to the number of generators in the test case.
            - "prod_v": same as above but set the voltage setpoint of generator units.

    _set_topo_vect: :class:`numpy.ndarray`, dtype:int
        Similar to :attr:`BaseAction._set_line_status` but instead of affecting the status of powerlines, it affects the
        bus connectivity at a substation. It has the same size as the full topological vector
        (:attr:`BaseAction._dim_topo`)
        and for each element it should be understood as:

            - 0 -> don't change
            - 1 -> connect to bus 1
            - 2 -> connect to bus 2
            - -1 -> disconnect the object.

    _change_bus_vect: :class:`numpy.ndarray`, dtype:bool
         Similar to :attr:`BaseAction._switch_line_status` but it affects the topology at substations instead of the
         status of
         the powerline. It has the same size as the full topological vector (:attr:`BaseAction._dim_topo`) and each
         component should mean:

             - ``False``: the object is not affected
             - ``True``: the object will be moved to another bus. If it was on bus 1 it will be moved on bus 2, and if
               it was on bus 2 it will be moved on bus 1.

    authorized_keys: :class:`set`
        The set indicating which keys the actions can understand when calling :func:`BaseAction.update`

    _subs_impacted: :class:`numpy.ndarray`, dtype:bool
        This attributes is either not initialized (set to ``None``) or it tells, for each substation, if it is impacted
        by the action (in this case :attr:`BaseAction._subs_impacted`\[sub_id\] is ``True``) or not
        (in this case :attr:`BaseAction._subs_impacted`\[sub_id\] is ``False``)

    _lines_impacted: :class:`numpy.ndarray`, dtype:bool
        This attributes is either not initialized (set to ``None``) or it tells, for each powerline, if it is impacted
        by the action (in this case :attr:`BaseAction._lines_impacted`\[line_id\] is ``True``) or not
        (in this case :attr:`BaseAction._subs_impacted`\[line_id\] is ``False``)

    attr_list_vect: ``list``, static
        The authorized key that are processed by :func:`BaseAction.__call__` to modify the injections

    attr_list_vect_set: ``set``, static
        The authorized key that is processed by :func:`BaseAction.__call__` to modify the injections

    _redispatch: :class:`numpy.ndarray`, dtype:float
        Amount of redispatching that this action will perform. Redispatching will increase the generator's active
        setpoint
        value. This will be added to the value of the generators. The Environment will make sure that every physical
        constraint is met. This means that the agent provides a setpoint, but there is no guarantee that the setpoint
        will be achievable. Redispatching action is cumulative, this means that if at a given timestep you ask +10 MW
        on a generator, and on another you ask +10 MW then the total setpoint for this generator that the environment
        will try to implement is +20MW.

    _storage_power: :class:`numpy.ndarray`, dtype:float
        Amount of power you want each storage units to produce / absorbs. Storage units are in "loads"
        convention. This means that if you ask for a positive number, the storage unit will absorb
        power from the grid (=it will charge) and if you ask for a negative number, the storage unit
        will inject power on the grid (storage unit will discharge).

    _curtail: :class:`numpy.ndarray`, dtype:float
        For each renewable generator, allows you to give a maximum value (as ratio of Pmax, *eg* 0.5 =>
        you limit the production of this generator to 50% of its Pmax) to renewable generators.
        
        .. warning::
            In grid2op we decided that the "curtailment" type of actions consists in directly providing the 
            upper bound you the agent allowed for a given generator. It does not reflect the amount
            of MW that will be "curtailed" but will rather provide a limit on the number of
            MW a given generator can produce.

    Examples
    --------
    Here are example on how to use the action, for more information on what will be the effect of each,
    please refer to the explanatory notebooks.

    You have two main methods to build actions, as showed here:

    .. code-block:: python

        import grid2op
        env_name = "l2rpn_case14_sandbox"  # or any other name
        env = grid2op.make(env_name)

        # first method:
        action_description = {...}  # see below
        act = env.action_space(action_description)

        # second method
        act = env.action_space()
        act.PROPERTY = MODIF

    The description of action as a dictionary is the "historical" method. The method using the properties
    has been added to simplify the API.

    To connect / disconnect powerline, using the "set" action, you can:

    .. code-block:: python

        # method 1
        act = env.action_space({"set_line_status": [(line_id, new_status), (line_id, new_status), ...]})

        # method 2
        act = env.action_space()
        act.line_set_status = [(line_id, new_status), (line_id, new_status), ...]

    typically: 0 <= line_id <= env.n_line and new_status = 1 or -1

    To connect / disconnect powerline using the "change" action type, you can:

    .. code-block:: python

        # method 1
        act = env.action_space({"change_line_status": [line_id, line_id, ...]})

        # method 2
        act = env.action_space()
        act.line_change_status = [line_id, line_id, ...]

    typically: 0 <= line_id <= env.n_line

    To modify the busbar at which an element is connected you can (if using set, to use "change" instead
    replace "set_bus" in the text below by "change_bus" **eg** `nv.action_space({"change_bus": ...})`
    or `act.load_change_bus = ...` ):

    .. code-block:: python

        # method 1
        act = env.action_space({"set_bus":
                                    {"lines_or_id": [(line_id, new_bus), (line_id, new_bus), ...],
                                     "lines_ex_id": [(line_id, new_bus), (line_id, new_bus), ...],
                                     "loads_id": [(load_id, new_bus), (load_id, new_bus), ...],
                                     "generators_id": [(gen_id, new_bus), (gen_id, new_bus), ...],
                                     "storages_id": [(storage_id, new_bus), (storage_id, new_bus), ...]
                                     }
                                })

        # method 2
        act = env.action_space()
        act.line_or_set_bus = [(line_id, new_bus), (line_id, new_bus), ...]
        act.line_ex_set_bus = [(line_id, new_bus), (line_id, new_bus), ...]
        act.load_set_bus =  [(load_id, new_bus), (load_id, new_bus), ...]
        act.gen_set_bus = [(gen_id, new_bus), (gen_id, new_bus), ...]
        act.storage_set_bus = [(storage_id, new_bus), (storage_id, new_bus), ...]

    Of course you can modify one type of object at a time (you don't have to specify all "lines_or_id",
    "lines_ex_id", "loads_id", "generators_id", "storages_id"

    You can also give the topologies you want at each substations with:

    .. code-block:: python

        # method 1
        act = env.action_space({"set_bus":{
                                "substations_id": [(sub_id, topo_sub), (sub_id, topo_sub), ...]
                                }})
        # method 2
        act = env.action_space()
        act.sub_set_bus = [(sub_id, topo_sub), (sub_id, topo_sub), ...]

    In the above typically 0 <= sub_id < env.n_sub and topo_sub is a vector having the right dimension (
    so if a substation has 4 elements, then topo_sub should have 4 elements)

    It has to be noted that `act.sub_set_bus` will return a 1d vector representing the topology
    of the grid as "set" by the action, with the convention, -1 => disconnect, 0 => don't change,
    1=> set to bus 1 and 2 => set object to bus 2.


    In order to perform redispatching you can do as follow:

    .. code-block:: python

        # method 1
        act = env.action_space({"redispatch": [(gen_id, amount), (gen_id, amount), ...]})
        # method 2
        act = env.action_space()
        act.redispatch = [(gen_id, amount), (gen_id, amount), ...]

    Typically 0<= gen_id < env.n_gen and `amount` is a floating point between gen_max_ramp_down and
    gen_min_ramp_down for the generator modified.

    In order to perform action on storage units, you can:

    .. code-block:: python

        # method 1
        act = env.action_space({"set_storage": [(storage_id, amount), (storage_id, amount), ...]})

        # method 2
        act = env.action_space()
        act.set_storage = [(storage_id, amount), (storage_id, amount), ...]

    Typically `0 <= storage_id < env.n_storage` and `amount` is a floating point between the maximum
    power and minimum power the storage unit can absorb / produce.

    Finally, in order to perform curtailment action on renewable generators, you can:

    .. code-block:: python

        # method 1
        act = env.action_space({"curtail": [(gen_id, amount), (gen_id, amount), ...]})

        # method 2
        act = env.action_space()
        act.curtail = [(gen_id, amount), (gen_id, amount), ...]

    Typically `0 <= gen_id < env.n_gen` and `amount` is a floating point between the 0. and 1.
    giving the limit of power you allow each renewable generator to produce (expressed in ratio of
    Pmax). For example if `gen_id=1` and `amount=0.7` it means you limit the production of
    generator 1 to 70% of its Pmax.

    """

    authorized_keys = {
        "injection",
        "hazards",
        "maintenance",
        "set_line_status",
        "change_line_status",
        "set_bus",
        "change_bus",
        "redispatch",
        "set_storage",
        "curtail",
        "raise_alarm",
        "raise_alert",
    }

    attr_list_vect = [
        "prod_p",
        "prod_v",
        "load_p",
        "load_q",
        "_redispatch",
        "_set_line_status",
        "_switch_line_status",
        "_set_topo_vect",
        "_change_bus_vect",
        "_hazards",
        "_maintenance",
        "_storage_power",
        "_curtail",
        "_raise_alarm",
        "_raise_alert",
    ]
    attr_nan_list_set = set()

    attr_list_set = set(attr_list_vect)
    shunt_added = False

    _line_or_str = "line (origin)"
    _line_ex_str = "line (extremity)"

    ERR_ACTION_CUT = 'The action added to me will be cut, because i don\'t support modification of "{}"'
    ERR_NO_STOR_SET_BUS = 'Impossible to modify the storage bus (with "set") with this action type.'
    
    def __init__(self):
        """
        INTERNAL USE ONLY

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            **It is NOT recommended** to create an action with this method, use the action space
            of the environment :attr:`grid2op.Environment.Environment.action_space` instead.

        This is used to create an BaseAction instance. Preferably, :class:`BaseAction` should be created with
        :class:`ActionSpace`.

        IMPORTANT: Use :func:`ActionSpace.__call__` or :func:`ActionSpace.sample` to generate a valid action.

        """
        GridObjects.__init__(self)

        # False(line is disconnected) / True(line is connected)
        self._set_line_status = np.full(shape=self.n_line, fill_value=0, dtype=dt_int)
        self._switch_line_status = np.full(
            shape=self.n_line, fill_value=False, dtype=dt_bool
        )

        # injection change
        self._dict_inj = {}

        # topology changed
        self._set_topo_vect = np.full(shape=self.dim_topo, fill_value=0, dtype=dt_int)
        self._change_bus_vect = np.full(
            shape=self.dim_topo, fill_value=False, dtype=dt_bool
        )

        # add the hazards and maintenance usefull for saving.
        self._hazards = np.full(shape=self.n_line, fill_value=False, dtype=dt_bool)
        self._maintenance = np.full(shape=self.n_line, fill_value=False, dtype=dt_bool)

        # redispatching vector
        self._redispatch = np.full(shape=self.n_gen, fill_value=0.0, dtype=dt_float)

        # storage unit vector
        self._storage_power = np.full(
            shape=self.n_storage, fill_value=0.0, dtype=dt_float
        )

        # curtailment of renewable energy
        self._curtail = np.full(shape=self.n_gen, fill_value=-1.0, dtype=dt_float)

        self._vectorized = None
        self._lines_impacted = None
        self._subs_impacted = None

        # shunts
        if type(self).shunts_data_available:
            self.shunt_p = np.full(
                shape=self.n_shunt, fill_value=np.NaN, dtype=dt_float
            )
            self.shunt_q = np.full(
                shape=self.n_shunt, fill_value=np.NaN, dtype=dt_float
            )
            self.shunt_bus = np.full(shape=self.n_shunt, fill_value=0, dtype=dt_int)
        else:
            self.shunt_p = None
            self.shunt_q = None
            self.shunt_bus = None

        self._single_act = True

        self._raise_alarm = np.full(
            shape=self.dim_alarms, dtype=dt_bool, fill_value=False
        )  # TODO

        self._raise_alert = np.full(
            shape=self.dim_alerts, dtype=dt_bool, fill_value=False
        )  # TODO

        # change the stuff
        self._modif_inj = False
        self._modif_set_bus = False
        self._modif_change_bus = False
        self._modif_set_status = False
        self._modif_change_status = False
        self._modif_redispatch = False
        self._modif_storage = False
        self._modif_curtailment = False
        self._modif_alarm = False
        self._modif_alert = False

    @classmethod
    def process_shunt_satic_data(cls):
        if not cls.shunts_data_available:
            # this is really important, otherwise things from grid2op base types will be affected
            cls.attr_list_vect = copy.deepcopy(cls.attr_list_vect)
            cls.attr_list_set = copy.deepcopy(cls.attr_list_set)
            # remove the shunts from the list to vector
            for el in ["shunt_p", "shunt_q", "shunt_bus"]:
                if el in cls.attr_list_vect:
                    try:
                        cls.attr_list_vect.remove(el)
                    except ValueError:
                        pass
            cls.attr_list_set = set(cls.attr_list_vect)
        return super().process_shunt_satic_data()
    
    def copy(self) -> "BaseAction":
        # sometimes this method is used...
        return self.__deepcopy__()

    def shape(self):
        return type(self).shapes()
    
    def dtype(self):
        return type(self).dtypes()
    
    def _aux_copy(self, other):
        attr_simple = [
            "_modif_inj",
            "_modif_set_bus",
            "_modif_change_bus",
            "_modif_set_status",
            "_modif_change_status",
            "_modif_redispatch",
            "_modif_storage",
            "_modif_curtailment",
            "_modif_alarm",
            "_modif_alert",
            "_single_act",
        ]

        attr_vect = [
            "_set_line_status",
            "_switch_line_status",
            "_set_topo_vect",
            "_change_bus_vect",
            "_hazards",
            "_maintenance",
            "_redispatch",
            "_storage_power",
            "_curtail",
            "_raise_alarm",
            "_raise_alert",
        ]

        if type(self).shunts_data_available:
            attr_vect += ["shunt_p", "shunt_q", "shunt_bus"]

        for attr_nm in attr_simple:
            setattr(other, attr_nm, getattr(self, attr_nm))

        for attr_nm in attr_vect:
            getattr(other, attr_nm)[:] = getattr(self, attr_nm)

    def __copy__(self) -> "BaseAction":
        res = type(self)()

        self._aux_copy(other=res)

        # handle dict_inj
        for k, el in self._dict_inj.items():
            res._dict_inj[k] = copy.copy(el)

        # just copy
        res._vectorized = self._vectorized
        res._lines_impacted = self._lines_impacted
        res._subs_impacted = self._subs_impacted

        return res

    @classmethod
    def process_shunt_satic_data(cls):
        return super().process_shunt_satic_data()
    
    def __deepcopy__(self, memodict={}) -> "BaseAction":
        res = type(self)()

        self._aux_copy(other=res)

        # handle dict_inj
        for k, el in self._dict_inj.items():
            res._dict_inj[k] = copy.deepcopy(el, memodict)

        # just copy
        res._vectorized = copy.deepcopy(self._vectorized, memodict)
        res._lines_impacted = copy.deepcopy(self._lines_impacted, memodict)
        res._subs_impacted = copy.deepcopy(self._subs_impacted, memodict)

        return res

    def _aux_serialize_add_key_change(self, attr_nm, dict_key, res):
            tmp_ = [int(id_) for id_, val in enumerate(getattr(self, attr_nm)) if val]
            if tmp_:
                res[dict_key] = tmp_

    def _aux_serialize_add_key_set(self, attr_nm, dict_key, res):            
            tmp_ = [(int(id_), int(val)) for id_, val in enumerate(getattr(self, attr_nm)) if np.abs(val) >= 1e-7]
            if tmp_:
                res[dict_key] = tmp_
                
    def as_serializable_dict(self) -> dict:
        """
        This method returns an action as a dictionnary, that can be serialized using the "json" module.

        It can be used to store the action into a grid2op indepependant format (the default action serialization, for speed, writes actions to numpy array.
        The size of these arrays can change depending on grid2op versions, especially if some different types of actions are implemented).

        Once you have these dictionnary, you can use them to build back the action from the action space.

        .. warning::
            This function does not work correctly with version of grid2op lower (or equal to) 1.9.5 
            
        Examples
        ---------

        It can be used like:

        .. code-block:: python

            import grid2op
            env_name = "l2rpn_case14_sandbox"  # or anything else
            env = grid2op.make(env_name)

            act = env.action_space(...)

            dict_ = act.as_serializable_dict()  # you can save this dict with the json library
            act2 = env.action_space(dict_)
            act == act2

        """
        res = {}
        # bool elements
        if self._modif_alert:
            res["raise_alert"] = [
                int(id_) for id_, val in enumerate(self._raise_alert) if val
            ]
            if not res["raise_alert"]:
                del res["raise_alert"]
                
        if self._modif_alarm:
            res["raise_alarm"] = [
                int(id_) for id_, val in enumerate(self._raise_alarm) if val
            ]
            if not res["raise_alarm"]:
                del res["raise_alarm"]
                
        if self._modif_change_bus:
            res["change_bus"] = {}
            self._aux_serialize_add_key_change("load_change_bus", "loads_id", res["change_bus"])
            self._aux_serialize_add_key_change("gen_change_bus", "generators_id", res["change_bus"])
            self._aux_serialize_add_key_change("line_or_change_bus", "lines_or_id", res["change_bus"])
            self._aux_serialize_add_key_change("line_ex_change_bus", "lines_ex_id", res["change_bus"])
            if hasattr(type(self), "n_storage") and type(self).n_storage:
                self._aux_serialize_add_key_change("storage_change_bus", "storages_id", res["change_bus"])
            if not res["change_bus"]:
                del res["change_bus"]
            
        if self._modif_change_status:
            res["change_line_status"] = [
                int(id_) for id_, val in enumerate(self._switch_line_status) if val
            ]
            if not res["change_line_status"]:
                del res["change_line_status"]
            
        # int elements
        if self._modif_set_bus:
            res["set_bus"] = {}
            self._aux_serialize_add_key_set("load_set_bus", "loads_id", res["set_bus"])
            self._aux_serialize_add_key_set("gen_set_bus", "generators_id", res["set_bus"])
            self._aux_serialize_add_key_set("line_or_set_bus", "lines_or_id", res["set_bus"])
            self._aux_serialize_add_key_set("line_ex_set_bus", "lines_ex_id", res["set_bus"])
            if hasattr(type(self), "n_storage") and type(self).n_storage:
                self._aux_serialize_add_key_set("storage_set_bus", "storages_id", res["set_bus"])
            if not res["set_bus"]:
                del res["set_bus"]
            
        if self._modif_set_status:
            res["set_line_status"] = [
                (int(id_), int(val))
                for id_, val in enumerate(self._set_line_status)
                if val != 0
            ]
            if not res["set_line_status"]:
                del res["set_line_status"]
            
        # float elements
        if self._modif_redispatch:
            res["redispatch"] = [
                (int(id_), float(val))
                for id_, val in enumerate(self._redispatch)
                if np.abs(val) >= 1e-7
            ]
            if not res["redispatch"]:
                del res["redispatch"]
                
        if self._modif_storage:
            res["set_storage"] = [
                (int(id_), float(val))
                for id_, val in enumerate(self._storage_power)
                if np.abs(val) >= 1e-7
            ]
            if not res["set_storage"]:
                del res["set_storage"]
                
        if self._modif_curtailment:
            res["curtail"] = [
                (int(id_), float(val))
                for id_, val in enumerate(self._curtail)
                if np.abs(val + 1.) >= 1e-7
            ]
            if not res["curtail"]:
                del res["curtail"]

        # more advanced options
        if self._modif_inj:
            res["injection"] = {}
            for ky in ["prod_p", "prod_v", "load_p", "load_q"]:
                if ky in self._dict_inj:
                    res["injection"][ky] = [float(val) for val in self._dict_inj[ky]]
            if not res["injection"]:
                del res["injection"]

        if type(self).shunts_data_available:
            res["shunt"] = {}
            if np.isfinite(self.shunt_p).any():
                res["shunt"]["shunt_p"] = [
                    (int(sh_id), float(val)) for sh_id, val in enumerate(self.shunt_p) if np.isfinite(val)
                ]
            if np.isfinite(self.shunt_q).any():
                res["shunt"]["shunt_q"] = [
                    (int(sh_id), float(val)) for sh_id, val in enumerate(self.shunt_q) if np.isfinite(val)
                ]
            if (self.shunt_bus != 0).any():
                res["shunt"]["shunt_bus"] = [
                    (int(sh_id), int(val))
                    for sh_id, val in enumerate(self.shunt_bus)
                    if val != 0
                ]
            if not res["shunt"]:
                del res["shunt"]
        return res

    @classmethod
    def _add_shunt_data(cls):
        if cls.shunt_added is False and cls.shunts_data_available:
            cls.shunt_added = True

            cls.attr_list_vect = copy.deepcopy(cls.attr_list_vect)
            cls.attr_list_vect += ["shunt_p", "shunt_q", "shunt_bus"]

            cls.authorized_keys = copy.deepcopy(cls.authorized_keys)
            cls.authorized_keys.add("shunt")
            cls.attr_nan_list_set.add("shunt_p")
            cls.attr_nan_list_set.add("shunt_q")
            cls._update_value_set()

    def alarm_raised(self) -> np.ndarray:
        """
        INTERNAL

        .. warning:: 
            /!\\\\ Only valid with "l2rpn_icaps_2021" environment /!\\\\

        This function is used to know if the given action aimed at raising an alarm or not.

        Returns
        -------
        res: numpy array
            The indexes of the areas where the agent has raised an alarm.

        """
        return (self._raise_alarm).nonzero()[0]

    def alert_raised(self) -> np.ndarray:
        """
        INTERNAL

        This function is used to know if the given action aimed at raising an alert or not.

        Returns
        -------
        res: numpy array
            The indexes of the lines where the agent has raised an alert.

        """
        return (self._raise_alert).nonzero[0]

    @classmethod
    def _aux_process_old_compat(cls):
        super()._aux_process_old_compat()
        
        # this is really important, otherwise things from grid2op base types will be affected
        cls.authorized_keys = copy.deepcopy(cls.authorized_keys)
        cls.attr_list_vect = copy.deepcopy(cls.attr_list_vect)

        # deactivate storage
        if "set_storage" in cls.authorized_keys:
            cls.authorized_keys.remove("set_storage")
        if "_storage_power" in cls.attr_list_vect:
            cls.attr_list_vect.remove("_storage_power")
            cls.attr_list_set = set(cls.attr_list_vect)

        # remove the curtailment
        if "curtail" in cls.authorized_keys:
            cls.authorized_keys.remove("curtail")
        if "_curtail" in cls.attr_list_vect:
            cls.attr_list_vect.remove("_curtail")

    @classmethod
    def _aux_process_n_busbar_per_sub(cls):
        cls.authorized_keys = copy.deepcopy(cls.authorized_keys)
        cls.attr_list_vect = copy.deepcopy(cls.attr_list_vect)
        
        if "change_bus" in cls.authorized_keys:
            cls.authorized_keys.remove("change_bus")
        if "_change_bus_vect" in cls.attr_list_vect:
            cls.attr_list_vect.remove("_change_bus_vect")
                
    @classmethod
    def process_grid2op_compat(cls):
        super().process_grid2op_compat()
        glop_ver = cls._get_grid2op_version_as_version_obj()
        
        if cls.glop_version == cls.BEFORE_COMPAT_VERSION:
            # oldest version: no storage and no curtailment available
            cls._aux_process_old_compat()
            
        if glop_ver < version.parse("1.6.0"):
            # this feature did not exist before.
            cls.dim_alarms = 0
        
        if glop_ver < version.parse("1.9.1"):
            # this feature did not exist before.
            cls.dim_alerts = 0

        if (cls.n_busbar_per_sub >= 3) or (cls.n_busbar_per_sub == 1):
            # only relevant for grid2op >= 1.10.0
            # remove "change_bus" if it's there more than 3 buses (no sense: where to change it ???)
            # or if there are only one busbar (cannot change anything)
            # if there are only one busbar, the "set_bus" action can still be used
            # to disconnect the element, this is why it's not removed
            cls._aux_process_n_busbar_per_sub()
                
        cls.attr_list_set = copy.deepcopy(cls.attr_list_set)
        cls.attr_list_set = set(cls.attr_list_vect)
                
    def _reset_modified_flags(self):
        self._modif_inj = False
        self._modif_set_bus = False
        self._modif_change_bus = False
        self._modif_set_status = False
        self._modif_change_status = False
        self._modif_redispatch = False
        self._modif_storage = False
        self._modif_curtailment = False
        self._modif_alarm = False
        self._modif_alert = False

    def can_affect_something(self) -> bool:
        """
        This functions returns True if the current action has any chance to change the grid.

        Notes
        -----
        This does not say however if the action will indeed modify something somewhere !
        """
        return (
            self._modif_inj
            or self._modif_set_bus
            or self._modif_change_bus
            or self._modif_set_status
            or self._modif_change_status
            or self._modif_redispatch
            or self._modif_storage
            or self._modif_curtailment
            or self._modif_alarm
            or self._modif_alert
        )

    def _get_array_from_attr_name(self, attr_name):
        if hasattr(self, attr_name):
            res = super()._get_array_from_attr_name(attr_name)
        else:
            if attr_name in self._dict_inj:
                res = self._dict_inj[attr_name]
            else:
                if attr_name == "prod_p" or attr_name == "prod_v":
                    res = np.full(self.n_gen, fill_value=0.0, dtype=dt_float)
                elif attr_name == "load_p" or attr_name == "load_q":
                    res = np.full(self.n_load, fill_value=0.0, dtype=dt_float)
                else:
                    raise Grid2OpException(
                        'Impossible to find the attribute "{}" '
                        'into the BaseAction of type "{}"'.format(attr_name, type(self))
                    )
        return res

    def _post_process_from_vect(self):
        self._modif_inj = self._dict_inj != {}
        self._modif_set_bus = (self._set_topo_vect != 0).any()
        self._modif_change_bus = (self._change_bus_vect).any()
        self._modif_set_status = (self._set_line_status != 0).any()
        self._modif_change_status = (self._switch_line_status).any()
        self._modif_redispatch = (
            np.isfinite(self._redispatch) & (np.abs(self._redispatch) >= 1e-7)
        ).any()
        self._modif_storage = (np.abs(self._storage_power) >= 1e-7).any()
        self._modif_curtailment = (np.abs(self._curtail + 1.0) >= 1e-7).any()
        self._modif_alarm = self._raise_alarm.any()
        self._modif_alert = self._raise_alert.any()

    def _assign_attr_from_name(self, attr_nm, vect):
        if hasattr(self, attr_nm):
            if attr_nm not in type(self).attr_list_set:
                raise AmbiguousAction(
                    f"Impossible to modify attribute {attr_nm} with this action type."
                )
            super()._assign_attr_from_name(attr_nm, vect)
            self._post_process_from_vect()
        else:
            if np.isfinite(vect).any() and (np.abs(vect) >= 1e-7).any():
                self._dict_inj[attr_nm] = vect

    def check_space_legit(self):
        """
        This method allows to check if this method is ambiguous **per se** (defined more formally as:
        whatever the observation at time *t*, and the changes that can occur between *t* and *t+1*, this
        action will be ambiguous).

        For example, an action that try to assign something to busbar 3 will be ambiguous *per se*. An action
        that tries to dispatch a non dispatchable generator will be also ambiguous *per se*.

        However, an action that "switch" (change) the status (connected / disconnected) of a powerline can be
        ambiguous and it will not be detected here. This is because the ambiguity depends on the current state
        of the powerline:

        - if the powerline is disconnected, changing its status means reconnecting it. And we cannot reconnect a
          powerline without specifying on which bus.
        - on the contrary if the powerline is connected, changing its status means disconnecting it, which is
          always feasible.

        In case of "switch" as we see here, the action can be ambiguous, but not ambiguous *per se*. This method
        will **never** throw any error in this case.

        Raises
        -------
        :class:`grid2op.Exceptions.AmbiguousAction`
            Or any of its more precise subclasses, depending on which assumption is not met.

        """
        self._check_for_ambiguity()

    def get_set_line_status_vect(self) -> np.ndarray:
        """
        Computes and returns a vector that can be used in the :func:`BaseAction.__call__` with the keyword
        "set_status" if building an :class:`BaseAction`.

        **NB** this vector is not the internal vector of this action but corresponds to "do nothing" action.

        Returns
        -------
        res: :class:`numpy.array`, dtype:dt_int
            A vector that doesn't affect the grid, but can be used in :func:`BaseAction.__call__` with the keyword
            "set_status" if building an :class:`BaseAction`.

        """
        return np.full(shape=self.n_line, fill_value=0, dtype=dt_int)

    def get_change_line_status_vect(self) -> np.ndarray:
        """
        Computes and returns a vector that can be used in the :func:`BaseAction.__call__` with the keyword
        "set_status" if building an :class:`BaseAction`.

        **NB** this vector is not the internal vector of this action but corresponds to "do nothing" action.

        Returns
        -------
        res: :class:`numpy.array`, dtype:dt_bool
            A vector that doesn't affect the grid, but can be used in :func:`BaseAction.__call__` with the keyword
            "set_status" if building an :class:`BaseAction`.

        """
        return np.full(shape=self.n_line, fill_value=False, dtype=dt_bool)

    def __eq__(self, other) -> bool:
        """
        Test the equality of two actions.

        2 actions are said to be identical if they have the same impact on the powergrid. This is unrelated to their
        respective class. For example, if an Action is of class :class:`Action` and doesn't act on the injection, it
        can be equal to an Action of the derived class :class:`TopologyAction` (if the topological modifications are the
        same of course).

        This implies that the attributes :attr:`Action.authorized_keys` is not checked in this method.

        Note that if 2 actions don't act on the same powergrid, or on the same backend (eg number of loads, or
        generators are not the same in *self* and *other*, or they are not in the same order) then action will be
        declared as different.

        **Known issue** if two backends are different, but the description of the _grid are identical (ie all
        n_gen, n_load, n_line, sub_info, dim_topo, all vectors \*_to_subid, and \*_pos_topo_vect are
        identical) then this method will not detect the backend are different, and the action could be declared
        as identical. For now, this is only a theoretical behavior: if everything is the same, then probably, up to
        the naming convention, then the power grids are identical too.

        Parameters
        ----------
        other: :class:`BaseAction`
            An instance of class Action to which "self" will be compared.

        Returns
        -------
        res: ``bool``
            Whether the actions are equal or not.

        """
        if other is None:
            return False

        # check that the underlying grid is the same in both instances
        same_grid = type(self).same_grid_class(type(other))
        if not same_grid:
            return False

        # _grid is the same, now I test the the injections modifications are the same
        same_action = self._modif_inj == other._modif_inj
        same_action = same_action and self._dict_inj.keys() == other._dict_inj.keys()
        if not same_action:
            return False

        # all injections are the same
        for el in self._dict_inj.keys():
            me_inj = self._dict_inj[el]
            other_inj = other._dict_inj[el]
            tmp_me = np.isfinite(me_inj)
            tmp_other = np.isfinite(other_inj)
            if not np.all(tmp_me == tmp_other) or not np.all(
                me_inj[tmp_me] == other_inj[tmp_other]
            ):
                return False

        # same line status
        if (self._modif_set_status != other._modif_set_status) or not np.all(
            self._set_line_status == other._set_line_status
        ):
            return False

        if (self._modif_change_status != other._modif_change_status) or not np.all(
            self._switch_line_status == other._switch_line_status
        ):
            return False

        # redispatching is same
        if (self._modif_redispatch != other._modif_redispatch) or not np.all(
            self._redispatch == other._redispatch
        ):
            return False

        # storage is same
        me_inj = self._storage_power
        other_inj = other._storage_power
        tmp_me = np.isfinite(me_inj)
        tmp_other = np.isfinite(other_inj)
        if not np.all(tmp_me == tmp_other) or not np.all(
            me_inj[tmp_me] == other_inj[tmp_other]
        ):
            return False

        # curtailment
        if (self._modif_curtailment != other._modif_curtailment) or not np.array_equal(
            self._curtail, other._curtail
        ):
            return False

        # alarm
        if (self._modif_alarm != other._modif_alarm) or not np.array_equal(
            self._raise_alarm, other._raise_alarm
        ):
            return False
    
        # alarm
        if (self._modif_alert != other._modif_alert) or not np.array_equal(
            self._raise_alert, other._raise_alert
        ):
            return False

        # same topology changes
        if (self._modif_set_bus != other._modif_set_bus) or not np.all(
            self._set_topo_vect == other._set_topo_vect
        ):
            return False
        if (self._modif_change_bus != other._modif_change_bus) or not np.all(
            self._change_bus_vect == other._change_bus_vect
        ):
            return False

        # shunts are the same
        if type(self).shunts_data_available:
            if self.n_shunt != other.n_shunt:
                return False
            is_ok_me = np.isfinite(self.shunt_p)
            is_ok_ot = np.isfinite(other.shunt_p)
            if (is_ok_me != is_ok_ot).any():
                return False
            if not (self.shunt_p[is_ok_me] == other.shunt_p[is_ok_ot]).all():
                return False
            is_ok_me = np.isfinite(self.shunt_q)
            is_ok_ot = np.isfinite(other.shunt_q)
            if (is_ok_me != is_ok_ot).any():
                return False
            if not (self.shunt_q[is_ok_me] == other.shunt_q[is_ok_ot]).all():
                return False
            if not (self.shunt_bus == other.shunt_bus).all():
                return False

        return True

    def _dont_affect_topology(self) -> bool:
        return (
            (not self._modif_set_bus)
            and (not self._modif_change_bus)
            and (not self._modif_set_status)
            and (not self._modif_change_status)
        )

    def get_topological_impact(self, powerline_status=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gives information about the element being impacted by this action.
        **NB** The impacted elements can be used by :class:`grid2op.BaseRules` to determine whether or not an action
        is legal or not.
        **NB** The impacted are the elements that can potentially be impacted by the action. This does not mean they
        will be impacted. For examples:

        * If an action from an :class:`grid2op.BaseAgent` reconnect a powerline, but this powerline is being
          disconnected by a hazard at the same time step, then this action will not be implemented on the grid.

        However, it this powerline couldn't be reconnected for some reason (for example it was already out of order)
        the action will still be declared illegal, even if it has NOT impacted the powergrid.

        * If an action tries to disconnect a powerline already disconnected, it will "impact" this powergrid.
          This means that even if the action will do nothing, it disconnecting this powerline is against the rules,
          then the action will be illegal.
        * If an action tries to change the topology of a substation, but this substation is already at the target
          topology, the same mechanism applies. The action will "impact" the substation, even if, in the end, it
          consists of doing nothing.

        Any such "change" that would be illegal is declared as "illegal" regardless of the real impact of this action
        on the powergrid.

        Returns
        -------
        lines_impacted: :class:`numpy.ndarray`, dtype:dt_bool
            A vector with the same size as the number of powerlines in the grid (:attr:`BaseAction.n_line`) with for
            each component ``True`` if the line STATUS is impacted by the action, and ``False`` otherwise. See
            :attr:`BaseAction._lines_impacted` for more information.

        subs_impacted: :class:`numpy.ndarray`, dtype:dt_bool
            A vector with the same size as the number of substations in the grid with for each
            component ``True`` if the substation is impacted by the action, and ``False`` otherwise. See
            :attr:`BaseAction._subs_impacted` for more information.

        Examples
        --------

        You can use this function like;

        .. code-block:: python

            import grid2op
            env_name = "l2rpn_case14_sandbox"  # or any other name
            env = grid2op.make(env_name)

            # get an action
            action = env.action_space.sample()
            # inspect its impact
            lines_impacted, subs_impacted = action.get_topological_impact()

            for line_id in np.where(lines_impacted)[0]:
                print(f"The line {env.name_line[line_id]} with id {line_id} is impacted by this action")

            print(action)
        """
        if self._dont_affect_topology():
            # action is not impacting the topology
            # so it does not modified anything concerning the topology
            self._lines_impacted = np.full(
                shape=self.n_line, fill_value=False, dtype=dt_bool
            )
            self._subs_impacted = np.full(
                shape=self.sub_info.shape, fill_value=False, dtype=dt_bool
            )
            return self._lines_impacted, self._subs_impacted

        if powerline_status is None:
            isnotconnected = np.full(self.n_line, fill_value=True, dtype=dt_bool)
        else:
            isnotconnected = ~powerline_status

        self._lines_impacted = self._switch_line_status | (self._set_line_status != 0)
        self._subs_impacted = np.full(
            shape=self.sub_info.shape, fill_value=False, dtype=dt_bool
        )

        # compute the changes of the topo vector
        effective_change = self._change_bus_vect | (self._set_topo_vect != 0)

        # remove the change due to powerline only
        effective_change[
            self.line_or_pos_topo_vect[self._lines_impacted & isnotconnected]
        ] = False
        effective_change[
            self.line_ex_pos_topo_vect[self._lines_impacted & isnotconnected]
        ] = False
        
        # i can change also the status of a powerline by acting on its extremity
        # first sub case i connected the powerline by setting origin OR extremity to positive stuff
        if powerline_status is not None:
            # if we don't know the state of the grid, we don't consider
            # these "improvments": we consider a powerline is never
            # affected if its bus is modified at any of its ends.
            connect_set_or = (self._set_topo_vect[self.line_or_pos_topo_vect] > 0) & (
                isnotconnected
            )
            self._lines_impacted |= connect_set_or
            effective_change[self.line_or_pos_topo_vect[connect_set_or]] = False
            effective_change[self.line_ex_pos_topo_vect[connect_set_or]] = False
            connect_set_ex = (self._set_topo_vect[self.line_ex_pos_topo_vect] > 0) & (
                isnotconnected
            )
            self._lines_impacted |= connect_set_ex
            effective_change[self.line_or_pos_topo_vect[connect_set_ex]] = False
            effective_change[self.line_ex_pos_topo_vect[connect_set_ex]] = False
            
            # second sub case i disconnected the powerline by setting origin or extremity to negative stuff
            disco_set_or = (self._set_topo_vect[self.line_or_pos_topo_vect] < 0) & (
                powerline_status
            )
            self._lines_impacted |= disco_set_or
            effective_change[self.line_or_pos_topo_vect[disco_set_or]] = False
            effective_change[self.line_ex_pos_topo_vect[disco_set_or]] = False
            disco_set_ex = (self._set_topo_vect[self.line_ex_pos_topo_vect] < 0) & (
                powerline_status
            )
            self._lines_impacted |= disco_set_ex
            effective_change[self.line_or_pos_topo_vect[disco_set_ex]] = False
            effective_change[self.line_ex_pos_topo_vect[disco_set_ex]] = False
        
        self._subs_impacted[self._topo_vect_to_sub[effective_change]] = True
        return self._lines_impacted, self._subs_impacted

    def remove_line_status_from_topo(self,
                                     obs: "grid2op.Observation.BaseObservation" = None,
                                     check_cooldown: bool = True):
        """
        .. versionadded:: 1.8.0
        
        This function prevent an action to act on a powerline status if
        through the "set_bus" and "change_bus" part if a cooldown applies (
        see :ref:`action_powerline_status` for cases where this can apply)
        
        For example:
        
        .. code-block:: python
        
            import grid2op
            import numpy as np
            env_name = "l2rpn_icaps_2021_small"
            env = grid2op.make(env_name)
            env.set_id(0) 
            env.seed(0)
            obs = env.reset()

            act = env.action_space({"set_bus": {"substations_id": [(27, [1, -1, 2, 2, 1])]}})
            obs, reward, done, info = env.step(act)

            act_sub28 = env.action_space({"set_bus": {"substations_id": [(28, [1, 2, 2, 1, 1])]}})
            obs, reward, done, info = env.step(act_sub28)
            # >>> info["exception"] : IllegalAction('Powerline with ids [42] have been modified illegally (cooldown)')

        This is because in the second action, the powerline 42 is assigned to bus 2, so it would be reconnected, 
        which is not possible due to the cooldown.
        
        The behaviour is (for all powerlines where a cooldown applies *ie* `obs.time_before_cooldown_sub > 0`):
        
          - if this line is disconnected and is assigned to a bus 1 or 2 at a substation for
            one of its end, then this part of the action is ignored (it has not effect: bus will NOT
            be set)
          - if this line is connected and it is assigned to bus "-1" at one of its side
            (extremity or origin side) then this part of the action is ignored (bus will NOT be "set")
          - if this line is disconnected and the bus to one of its side is "changed", then this
            part is ignored: bus will NOT be changed
        
        And regardless of cooldowns it also:
        
          - if a powerline is affected to a certain bus at one of its end with `set_bus` (for example 
            `set_bus` to 1 or 2) and at the same time disconnected (`set_line_status` is -1) then
            the `set_bus` part is ignored to avoid `AmbiguousAction`
          - if a powerline is disconnect from its bus at one of its end with `set_bus` (for example 
            `set_bus` to -1) and at the same time reconnected (`set_line_status` is 1) then
            the `set_bus` part is ignored to avoid `AmbiguousAction`
          - if a powerline is affected to a certain bus at one of its end with `change_bus` (`change_bus` is 
            ``True``) and at the same time disconnected (`set_line_status` is -1) then
            the `change_bus` part is ignore to avoid `AmbiguousAction`
            
            
        .. warning::
            This modifies the action in-place, especially the "set_bus" and "change_bus" attributes.
        
        .. note::
            This function does not check the cooldowns if you specify `check_cooldown=False`
        
        .. note::
            As from version 1.9.0 you are no longer forced to provide an observation if `check_cooldown=False`
        
        .. warning::
            For grid2op equal or lower to 1.9.5 this function was bugged in some corner cases. We highly recommend
            upgrading if you use this function with these grid2op versions.
            
        Examples
        ---------
        
        To avoid the issue explained above, you can now do:
        
        .. code-block:: python
        
            import grid2op
            import numpy as np
            env_name = "l2rpn_icaps_2021_small"
            env = grid2op.make(env_name)
            env.set_id(0) 
            env.seed(0)
            obs = env.reset()

            act = env.action_space({"set_bus": {"substations_id": [(27, [1, -1, 2, 2, 1])]}})
            obs, reward, done, info = env.step(act)

            act_sub28_clean = env.action_space({"set_bus": {"substations_id": [(28, [1, 2, 2, 1, 1])]}})
            act_sub28_clean.remove_line_status_from_topo(obs)
            print(act_sub28_clean)
            # This action will:
            #     - NOT change anything to the injections
            #     - NOT perform any redispatching action
            #     - NOT modify any storage capacity
            #     - NOT perform any curtailment
            #     - NOT force any line status
            #     - NOT switch any line status
            #     - NOT switch anything in the topology
            #     - Set the bus of the following element(s):
            #         - Assign bus 1 to line (extremity) id 41 [on substation 28]
            #         - Assign bus 2 to line (origin) id 44 [on substation 28]
            #         - Assign bus 1 to line (extremity) id 57 [on substation 28]
            #         - Assign bus 1 to generator id 16 [on substation 28]
            #     - NOT raise any alarm 
            #     - NOT raise any alert

            obs, reward, done, info = env.step(act_sub28_clean)
            # >>> info["exception"] : []
        
        .. note::
            The part of the action `act_sub28_clean` that would 
            "*- Assign bus 2 to line (extremity) id 42 [on substation 28]*" has been removed because powerline
            42 is disconnected in the observation and under a cooldown.
            
        Parameters
        ----------
        obs: :class:`grid2op.Observation.BaseObservation`
            The current observation
            
        check_cooldown: `bool`, optional
            If `True` (default) will modify the action only for the powerline impacted by a cooldown.
            Otherwise will modify all the powerlines.
            
            
        """
        if not check_cooldown:
            line_under_cooldown = np.full(self.n_line, fill_value=True, dtype=dt_bool)
            if obs is None:
                connected = np.full(self.n_line, fill_value=True, dtype=dt_bool)
                disconnected = np.full(self.n_line, fill_value=True, dtype=dt_bool)
            else:
                connected = obs.line_status
                disconnected = ~obs.line_status
        else:
            line_under_cooldown = obs.time_before_cooldown_line > 0
            connected = obs.line_status
            disconnected = ~obs.line_status
            
        cls = type(self)
        
        # remove the "set" part that would cause a reconnection
        mask_reco = np.full(cls.dim_topo, fill_value=False)
        if check_cooldown:
            reco_or_ = np.full(cls.n_line, fill_value=False)
            reco_or_[(self._set_topo_vect[cls.line_or_pos_topo_vect] > 0) & 
                    disconnected & line_under_cooldown] = True
            mask_reco[cls.line_or_pos_topo_vect] = reco_or_
            
            reco_ex_ = np.full(cls.n_line, fill_value=False)
            reco_ex_[(self._set_topo_vect[cls.line_ex_pos_topo_vect] > 0) & 
                    disconnected & line_under_cooldown] = True
            mask_reco[cls.line_ex_pos_topo_vect] = reco_ex_
        
        # do not reconnect powerline that will be disconnected
        reco_or_and_disco_ = np.full(cls.n_line, fill_value=False)
        reco_or_and_disco_[(self._set_topo_vect[cls.line_or_pos_topo_vect] > 0) & (self._set_topo_vect[cls.line_ex_pos_topo_vect] < 0)] = True
        reco_or_and_disco_[(self._set_topo_vect[cls.line_or_pos_topo_vect] > 0) & (self._set_line_status < 0)] = True
        mask_reco[cls.line_or_pos_topo_vect] |= reco_or_and_disco_

        reco_ex_and_disco_ = np.full(cls.n_line, fill_value=False)
        reco_ex_and_disco_[(self._set_topo_vect[cls.line_ex_pos_topo_vect] > 0) & (self._set_topo_vect[cls.line_or_pos_topo_vect] < 0)] = True
        reco_ex_and_disco_[(self._set_topo_vect[cls.line_ex_pos_topo_vect] > 0) & (self._set_line_status < 0)] = True
        mask_reco[cls.line_ex_pos_topo_vect] |= reco_ex_and_disco_
        # and now remove the change from the set_bus
        self._set_topo_vect[mask_reco] = 0

        
        # remove the "set" that would cause a disconnection
        mask_disco = np.full(cls.dim_topo, fill_value=False)
        if check_cooldown:
            disco_or_ = np.full(cls.n_line, fill_value=False)
            disco_or_[(self._set_topo_vect[cls.line_or_pos_topo_vect] < 0) & 
                    connected & line_under_cooldown] = True
            mask_disco[cls.line_or_pos_topo_vect] = disco_or_
            
            disco_ex_ = np.full(cls.n_line, fill_value=False)
            disco_ex_[(self._set_topo_vect[cls.line_ex_pos_topo_vect] < 0) & 
                    connected & line_under_cooldown] = True
            mask_disco[cls.line_ex_pos_topo_vect] = disco_ex_
            
        disco_or_and_reco_ = np.full(cls.n_line, fill_value=False)
        disco_or_and_reco_[(self._set_topo_vect[cls.line_or_pos_topo_vect] < 0) & (self._set_line_status > 0)] = True
        mask_disco[cls.line_or_pos_topo_vect] |= disco_or_and_reco_

        disco_ex_and_reco_ = np.full(cls.n_line, fill_value=False)
        disco_ex_and_reco_[(self._set_topo_vect[cls.line_ex_pos_topo_vect] < 0) & (self._set_line_status > 0)] = True
        mask_disco[cls.line_ex_pos_topo_vect] |= disco_ex_and_reco_
        self._set_topo_vect[mask_disco] = 0
        
        # remove the "change" part when powerlines is disconnected
        mask_disco = np.full(cls.dim_topo, fill_value=False)
        if check_cooldown:
            reco_or_ = np.full(cls.n_line, fill_value=False)
            reco_or_[self._change_bus_vect[cls.line_or_pos_topo_vect] & 
                    disconnected & line_under_cooldown] = True
            mask_disco[cls.line_or_pos_topo_vect] = reco_or_
            
            reco_ex_ = np.full(cls.n_line, fill_value=False)
            reco_ex_[self._change_bus_vect[cls.line_ex_pos_topo_vect] & 
                    disconnected & line_under_cooldown] = True
            mask_disco[cls.line_ex_pos_topo_vect] = reco_ex_
            
        # do not change bus of powerline that will be disconnected
        reco_or_and_disco_ = np.full(cls.n_line, fill_value=False)
        reco_or_and_disco_[(self._change_bus_vect[cls.line_or_pos_topo_vect]) & (self._set_line_status < 0)] = True
        mask_disco[cls.line_or_pos_topo_vect] |= reco_or_and_disco_
        reco_ex_and_disco_ = np.full(cls.n_line, fill_value=False)
        reco_ex_and_disco_[(self._change_bus_vect[cls.line_ex_pos_topo_vect]) & (self._set_line_status < 0)] = True
        mask_disco[cls.line_ex_pos_topo_vect] |= reco_ex_and_disco_

        # "erase" the change_bus for concerned powerlines
        self._change_bus_vect[mask_disco] = False
        
        return self
        
    def reset(self):
        """
        INTERNAL USE ONLY

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Reset the action to the "do nothing" state.

        """
        # False(line is disconnected) / True(line is connected)
        self._set_line_status[:] = 0
        self._switch_line_status[:] = False

        # injection change
        self._dict_inj = {}

        # topology changed
        self._set_topo_vect[:] = 0
        self._change_bus_vect[:] = False

        # add the hazards and maintenance usefull for saving.
        self._hazards[:] = False
        self._maintenance[:] = False

        # redispatching vector
        self._redispatch[:] = 0.0

        # storage
        self._storage_power[:] = 0.0

        # storage
        self._curtail[:] = -1.0

        self._vectorized = None
        self._lines_impacted = None
        self._subs_impacted = None

        # shunts
        if type(self).shunts_data_available:
            self.shunt_p[:] = np.NaN
            self.shunt_q[:] = np.NaN
            self.shunt_bus[:] = 0

        # alarm
        self._raise_alarm[:] = False        
        
        # alert
        self._raise_alert[:] = False

        self._reset_modified_flags()

    def _assign_iadd_or_warn(self, attr_name, new_value):
        if attr_name not in self.attr_list_set:
            old_value = getattr(self, attr_name)
            new_is_finite = np.isfinite(new_value)
            old_is_finite = np.isfinite(old_value)
            new_finite = new_value[new_is_finite | old_is_finite]
            old_finite = old_value[new_is_finite | old_is_finite]
            if (new_finite != old_finite).any():
                warnings.warn(
                    type(self).ERR_ACTION_CUT.format(attr_name)
                )
        else:
            getattr(self, attr_name)[:] = new_value
            
    def _aux_iadd_inj(self, other):
        for el in self.attr_list_vect:
            if el in other._dict_inj:
                if el not in self._dict_inj:
                    self._dict_inj[el] = other._dict_inj[el]
                else:
                    val = other._dict_inj[el]
                    ok_ind = np.isfinite(val)
                    self._dict_inj[el][ok_ind] = val[ok_ind]
        # warning if the action cannot be added
        for el in other._dict_inj:
            if not el in self.attr_list_set:
                warnings.warn(
                    type(self).ERR_ACTION_CUT.format(el)
                )
    
    def _aux_iadd_redisp(self, other):
        redispatching = other._redispatch
        if (np.abs(redispatching) >= 1e-7).any():
            if "_redispatch" not in self.attr_list_set:
                warnings.warn(
                    type(self).ERR_ACTION_CUT.format("_redispatch")
                )
            else:
                ok_ind = np.isfinite(redispatching)
                self._redispatch[ok_ind] += redispatching[ok_ind]
    
    def _aux_iadd_curtail(self, other):
        curtailment = other._curtail
        ok_ind = np.isfinite(curtailment) & (np.abs(curtailment + 1.0) >= 1e-7)
        if ok_ind.any():
            if "_curtail" not in self.attr_list_set:
                warnings.warn(
                    type(self).ERR_ACTION_CUT.format("_curtail")
                )
            else:
                # new curtailment of the results should be
                # the curtailment of rhs, only when rhs acts
                # on curtailment
                self._curtail[ok_ind] = curtailment[ok_ind]
    
    def _aux_iadd_storage(self, other):
        set_storage = other._storage_power
        ok_ind = np.isfinite(set_storage) & (np.abs(set_storage) >= 1e-7).any()
        if ok_ind.any():
            if "_storage_power" not in self.attr_list_set:
                warnings.warn(
                    type(self).ERR_ACTION_CUT.format("_storage_power")
                )
            else:
                self._storage_power[ok_ind] += set_storage[ok_ind]
                
    def _aux_iadd_modif_flags(self, other):
        self._modif_change_bus = self._modif_change_bus or other._modif_change_bus
        self._modif_set_bus = self._modif_set_bus or other._modif_set_bus
        self._modif_change_status = (
            self._modif_change_status or other._modif_change_status
        )
        self._modif_set_status = self._modif_set_status or other._modif_set_status
        self._modif_inj = self._modif_inj or other._modif_inj
        self._modif_redispatch = self._modif_redispatch or other._modif_redispatch
        self._modif_storage = self._modif_storage or other._modif_storage
        self._modif_curtailment = self._modif_curtailment or other._modif_curtailment
        self._modif_alarm = self._modif_alarm or other._modif_alarm
        self._modif_alert = self._modif_alert or other._modif_alert
        
    def _aux_iadd_shunt(self, other):
        if not type(other).shunts_data_available:
            warnings.warn("Trying to add an action that does not support "
                          "shunt with an action that does.")
            return
        
        val = other.shunt_p
        ok_ind = np.isfinite(val)
        shunt_p = 1.0 * self.shunt_p
        shunt_p[ok_ind] = val[ok_ind]
        self._assign_iadd_or_warn("shunt_p", shunt_p)

        val = other.shunt_q
        ok_ind = np.isfinite(val)
        shunt_q = 1.0 * self.shunt_q
        shunt_q[ok_ind] = val[ok_ind]
        self._assign_iadd_or_warn("shunt_q", shunt_q)

        val = other.shunt_bus
        ok_ind = val != 0
        shunt_bus = 1 * self.shunt_bus
        shunt_bus[ok_ind] = val[ok_ind]
        self._assign_iadd_or_warn("shunt_bus", shunt_bus)
    
    def _aux_iadd_set_change_status(self, other):
        other_set = other._set_line_status
        other_change = other._switch_line_status
        me_set = 1 * self._set_line_status
        me_change = copy.deepcopy(self._switch_line_status)

        # i change, but so does the other, i do nothing
        canceled_change = other_change & me_change
        # i dont change, the other change, i change
        update_change = other_change & ~me_change
        # Defered apply to prevent conflicts
        me_change[canceled_change] = False
        me_change[update_change] = True

        # i change, but the other set, it's erased
        me_change[other_set != 0 & me_change] = False

        # i set, but the other change, set to the opposite
        inverted_set = other_change & (me_set != 0)
        # so change +1 becomes -1 and -1 becomes +1
        me_set[inverted_set] *= -1
        # Has been inverted, cancel change
        me_change[inverted_set] = False

        # i set, the other set
        me_set[other_set != 0] = other_set[other_set != 0]

        self._assign_iadd_or_warn("_set_line_status", me_set)
        self._assign_iadd_or_warn("_switch_line_status", me_change)
        
    def _aux_iadd_set_change_bus(self, other):
        other_set = other._set_topo_vect
        other_change = other._change_bus_vect
        me_set = 1 * self._set_topo_vect
        me_change = copy.deepcopy(self._change_bus_vect)

        # i change, but so does the other, i do nothing
        canceled_change = other_change & me_change
        # i dont change, the other change, i change
        update_change = other_change & ~me_change
        # Defered apply to prevent conflicts
        me_change[canceled_change] = False
        me_change[update_change] = True

        # i change, but the other set, it's erased
        me_change[other_set != 0 & me_change] = False

        # i set, but the other change, set to the opposite
        inverted_set = other_change & (me_set > 0)
        # so change +1 becomes +2 and +2 becomes +1
        me_set[inverted_set] -= 1  # 1 becomes 0 and 2 becomes 1
        me_set[inverted_set] *= -1  # 1 is 0 and 2 becomes -1
        me_set[inverted_set] += 2  # 1 is 2 and 2 becomes 1
        # Has been inverted, cancel change
        me_change[inverted_set] = False

        # i set, the other set
        me_set[other_set != 0] = other_set[other_set != 0]

        self._assign_iadd_or_warn("_set_topo_vect", me_set)
        self._assign_iadd_or_warn("_change_bus_vect", me_change)
        
    def __iadd__(self, other: Self):
        """
        Add an action to this one.

        Adding an action to myself is equivalent to perform myself, and then perform other (but at the
        same step)

        Parameters
        ----------
        other: :class:`BaseAction`

        Examples
        --------

        .. code-block:: python

            import grid2op
            env_name = "l2rpn_case14_sandbox"  # or any other name
            env = grid2op.make(env_name)

            act1 = env.action_space()
            act1.set_bus = ...  # for example
            print("before += :")
            print(act1)

            act2 = env.action_space()
            act2.redispatch = ... # for example
            print(act2)

            act1 += act 2
            print("after += ")
            print(act1)

        """

        # deal with injections
        self._aux_iadd_inj(other)
                
        # redispatching
        self._aux_iadd_redisp(other)
        
        # storage
        self._aux_iadd_storage(other)

        # curtailment
        self._aux_iadd_curtail(other)

        # set and change status
        self._aux_iadd_set_change_status(other)

        # set and change bus
        self._aux_iadd_set_change_bus(other)

        # shunts
        if type(self).shunts_data_available:
            self._aux_iadd_shunt(other)

        # alarm feature
        self._raise_alarm[other._raise_alarm] = True

        # line alert feature
        self._raise_alert[other._raise_alert] = True


        # the modif flags
        self._aux_iadd_modif_flags(other)
        return self

    def __add__(self, other)  -> "BaseAction":
        """
        Implements the `+` operator for the action using the `+=` definition.

        This function is not commutative !

        Notes
        -------
        Be careful if two actions do not share the same type (for example you want to add act1
        of type :class:`TopologyAction` to act2 of type :class:`DispatchAction`) the results of
        `act1 + act2` might differ from what you expect.

        The result will always of the same type as act1. In the above case, it means that the `dispatch`
        part of `act2`will be ignored (because it is ignored in :class:`TopologyAction`).

        This is why we recommend to using this class directly with the :class:`PlayableAction` or
        from action directly generated with `env.action_space()`
        """
        res = type(self)()
        res += self
        res += other
        return res

    def __call__(self) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        INTERNAL USE ONLY

         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This method is used to return the effect of the current action in a format understandable by the backend.
        This format is detailed below.

        This function must also integrate the redispatching strategy for the BaseAction.

        It also performs a check of whether or not an action is "Ambiguous", eg an action that reconnect a powerline
        but doesn't specify on which bus to reconnect it is said to be ambiguous.

        If this :func:`BaseAction.__call__` is overloaded, the call of :func:`BaseAction._check_for_ambiguity` must be
        ensured by this the derived class.

        Returns
        -------
        dict_injection: :class:`dict`
            This dictionnary is :attr:`BaseAction._dict_inj`

        set_line_status: :class:`numpy.ndarray`, dtype:int
            This array is :attr:`BaseAction._set_line_status`

        switch_line_status: :class:`numpy.ndarray`, dtype:bool
            This array is :attr:`BaseAction._switch_line_status`

        set_topo_vect: :class:`numpy.ndarray`, dtype:int
            This array is :attr:`BaseAction._set_topo_vect`

        change_bus_vect: :class:`numpy.ndarray`, dtype:bool
            This array is :attr:`BaseAction._change_bus_vect`

        redispatch: :class:`numpy.ndarray`, dtype:float
            This array, that has the same size as the number of generators indicates for each generator the amount of
            redispatching performed by the action.

        storage_power: :class:`numpy.ndarray`, dtype:float
            Indicates, for all storage units, what is the production / absorbtion setpoint

        curtailment: :class:`numpy.ndarray`, dtype:float
            Indicates, for all generators, which curtailment is applied (if any)

        shunts: ``dict``
            A dictionary containing the shunts data, with keys: "shunt_p", "shunt_q" and "shunt_bus" and the
            convention, for "shun_p" and "shunt_q" that Nan means "don't change" and for shunt_bus: -1 => disconnect
            0 don't change, and 1 / 2 connect to bus 1 / 2

        Raises
        -------
        :class:`grid2op.Exceptions.AmbiguousAction`
            Or one of its derivate class.

        """

        self._check_for_ambiguity()

        dict_inj = self._dict_inj
        set_line_status = self._set_line_status
        switch_line_status = self._switch_line_status
        set_topo_vect = self._set_topo_vect
        change_bus_vect = self._change_bus_vect
        redispatch = self._redispatch
        storage_power = self._storage_power
        # remark: curtailment is handled by an algorithm in the environment, so don't need to be returned here
        shunts = {}
        if type(self).shunts_data_available:
            shunts["shunt_p"] = self.shunt_p
            shunts["shunt_q"] = self.shunt_q
            shunts["shunt_bus"] = self.shunt_bus
        # other remark: alarm and alert are not handled in the backend, this is why it does not appear here !
        return (
            dict_inj,
            set_line_status,
            switch_line_status,
            set_topo_vect,
            change_bus_vect,
            redispatch,
            storage_power,
            shunts,
        )

    def _digest_shunt(self, dict_):
        if not type(self).shunts_data_available:
            return

        if "shunt" in dict_:
            ddict_ = dict_["shunt"]

            key_shunt_reco = {"set_bus", "shunt_p", "shunt_q", "shunt_bus"}
            for k in ddict_:
                if k not in key_shunt_reco:
                    warn = "The key {} is not recognized by BaseAction when trying to modify the shunt.".format(
                        k
                    )
                    warn += " Recognized keys are {}".format(sorted(key_shunt_reco))
                    warnings.warn(warn)
            for key_n, vect_self in zip(
                ["shunt_bus", "shunt_p", "shunt_q", "set_bus"],
                [self.shunt_bus, self.shunt_p, self.shunt_q, self.shunt_bus],
            ):
                if key_n in ddict_:
                    tmp = ddict_[key_n]
                    if isinstance(tmp, np.ndarray):
                        # complete shunt vector is provided
                        vect_self[:] = tmp
                    elif isinstance(tmp, list):
                        # expected a list: (id shunt, new bus)
                        cls = type(self)
                        for (sh_id, new_bus) in tmp:
                            if sh_id < 0:
                                raise AmbiguousAction(
                                    "Invalid shunt id {}. Shunt id should be positive".format(
                                        sh_id
                                    )
                                )
                            if sh_id >= cls.n_shunt:
                                raise AmbiguousAction(
                                    "Invalid shunt id {}. Shunt id should be less than the number "
                                    "of shunt {}".format(sh_id, cls.n_shunt)
                                )
                            if key_n == "shunt_bus" or key_n == "set_bus":
                                if new_bus <= -2:
                                    raise IllegalAction(
                                        f"Cannot ask for a shunt bus <= -2, found {new_bus} for shunt id {sh_id}"
                                    )
                                elif new_bus > cls.n_busbar_per_sub:
                                    raise IllegalAction(
                                        f"Cannot ask for a shunt bus > {cls.n_busbar_per_sub} "
                                        f"the maximum number of busbar per substations"
                                        f", found {new_bus} for shunt id {sh_id}"
                                    )
                                
                            vect_self[sh_id] = new_bus
                    elif tmp is None:
                        pass
                    else:
                        raise AmbiguousAction(
                            "Invalid way to modify {} for shunts. It should be a numpy array or a "
                            "dictionary.".format(key_n)
                        )

    def _digest_injection(self, dict_):
        # I update the action
        if "injection" in dict_:
            if dict_["injection"] is not None:
                tmp_d = dict_["injection"]
                self._modif_inj = True
                for k in tmp_d:
                    if k in self.attr_list_set:
                        self._dict_inj[k] = np.array(tmp_d[k]).astype(dt_float)
                        # TODO check the size based on the input data !
                    else:
                        warn = (
                            "The key {} is not recognized by BaseAction when trying to modify the injections."
                            "".format(k)
                        )
                        warnings.warn(warn)

    def _digest_setbus(self, dict_):
        if "set_bus" in dict_:
            self._modif_set_bus = True
            if dict_["set_bus"] is None:
                # no real action has been made
                return
            
            if isinstance(dict_["set_bus"], dict):
                ddict_ = dict_["set_bus"]
                handled = False
                if "loads_id" in ddict_:
                    self.load_set_bus = ddict_["loads_id"]
                    handled = True
                if "generators_id" in ddict_:
                    self.gen_set_bus = ddict_["generators_id"]
                    handled = True
                if "lines_or_id" in ddict_:
                    self.line_or_set_bus = ddict_["lines_or_id"]
                    handled = True
                if "lines_ex_id" in ddict_:
                    self.line_ex_set_bus = ddict_["lines_ex_id"]
                    handled = True
                if "storages_id" in ddict_:
                    self.storage_set_bus = ddict_["storages_id"]
                    handled = True
                if "substations_id" in ddict_:
                    self.sub_set_bus = ddict_["substations_id"]
                    handled = True
                if ddict_ == {}:
                    handled = True
                    # weird way to do nothing but hey, how am I to judge ?
                if not handled:
                    msg = 'Invalid way to set the topology. When dict_["set_bus"] is a dictionary it should have'
                    msg += (
                        ' at least one of "loads_id", "generators_id", "lines_or_id", '
                    )
                    msg += '"lines_ex_id" or "substations_id" or "storages_id"'
                    msg += " as keys. None where found. Current used keys are: "
                    msg += "{}".format(sorted(ddict_.keys()))
                    raise AmbiguousAction(msg)
            else:
                self.set_bus = dict_["set_bus"]

    def _digest_change_bus(self, dict_):
        if "change_bus" in dict_:
            self._modif_change_bus = True
            if dict_["change_bus"] is None:
                # no real action has been made
                return

            if isinstance(dict_["change_bus"], dict):
                ddict_ = dict_["change_bus"]
                handled = False
                if "loads_id" in ddict_:
                    self.load_change_bus = ddict_["loads_id"]
                    handled = True
                if "generators_id" in ddict_:
                    self.gen_change_bus = ddict_["generators_id"]
                    handled = True
                if "lines_or_id" in ddict_:
                    self.line_or_change_bus = ddict_["lines_or_id"]
                    handled = True
                if "lines_ex_id" in ddict_:
                    self.line_ex_change_bus = ddict_["lines_ex_id"]
                    handled = True
                if "storages_id" in ddict_:
                    self.storage_change_bus = ddict_["storages_id"]
                    handled = True
                if "substations_id" in ddict_:
                    self.sub_change_bus = ddict_["substations_id"]
                    handled = True
                if ddict_ == {}:
                    handled = True
                    # weird way to do nothing but hey, how am I to judge ?
                if not handled:
                    msg = 'Invalid way to change the topology. When dict_["set_bus"] is a dictionary it should have'
                    msg += (
                        ' at least one of "loads_id", "generators_id", "lines_or_id", '
                    )
                    msg += '"lines_ex_id" or "substations_id" or "storages_id"'
                    msg += " as keys. None where found. Current used keys are: "
                    msg += "{}".format(sorted(ddict_.keys()))
                    raise AmbiguousAction(msg)
            else:
                self.change_bus = dict_["change_bus"]

    def _digest_set_status(self, dict_):
        if "set_line_status" in dict_:
            # this action can both disconnect or reconnect a powerlines
            self.line_set_status = dict_["set_line_status"]

    def _digest_hazards(self, dict_):
        if "hazards" in dict_:
            # set the values of the power lines to "disconnected" for element being "False"
            # does nothing to the others
            # an hazard will never reconnect a powerline
            if dict_["hazards"] is not None:
                self._modif_set_status = True
                tmp = dict_["hazards"]
                try:
                    tmp = np.array(tmp)
                except Exception as exc_:
                    raise AmbiguousAction(
                        f'You ask to perform hazard on powerlines, this can only be done if "hazards" can be casted '
                        f"into a numpy ndarray with error {exc_}"
                    )
                if np.issubdtype(tmp.dtype, np.dtype(bool).type):
                    if len(tmp) != self.n_line:
                        raise InvalidNumberOfLines(
                            'This "hazards" action acts on {} lines while there are {} in the _grid'.format(
                                len(tmp), self.n_line
                            )
                        )
                elif not np.issubdtype(tmp.dtype, np.dtype(int).type):
                    raise AmbiguousAction(
                        "You can only ask hazards with int or boolean numpy array vector."
                    )

                self._set_line_status[tmp] = -1
                self._hazards[tmp] = True
                # force ignore of any topological actions
                self._ignore_topo_action_if_disconnection(tmp)

    def _digest_maintenance(self, dict_):
        if "maintenance" in dict_:
            # set the values of the power lines to "disconnected" for element being "False"
            # does nothing to the others
            # a _maintenance operation will never reconnect a powerline
            if dict_["maintenance"] is not None:
                self._modif_set_status = True
                tmp = dict_["maintenance"]
                try:
                    tmp = np.array(tmp)
                except Exception as exc_:
                    raise AmbiguousAction(
                        f'You ask to perform maintenance on powerlines, this can only be done if "maintenance" can '
                        f"be casted into a numpy ndarray with error {exc_}"
                    )
                if np.issubdtype(tmp.dtype, np.dtype(bool).type):
                    if len(tmp) != self.n_line:
                        raise InvalidNumberOfLines(
                            'This "maintenance" action acts on {} lines while there are {} in the _grid'.format(
                                len(tmp), self.n_line
                            )
                        )
                elif not np.issubdtype(tmp.dtype, np.dtype(int).type):
                    raise AmbiguousAction(
                        "You can only ask to perform lines maintenance with int or boolean numpy array vector."
                    )
                self._set_line_status[tmp] = -1
                self._maintenance[tmp] = True
                self._ignore_topo_action_if_disconnection(tmp)

    def _digest_change_status(self, dict_):
        if "change_line_status" in dict_:
            # the action will switch the status of the powerline
            # for each element equal to 1 in this dict_["change_line_status"]
            # if the status is "disconnected" it will be transformed into "connected"
            # and if the status is "connected" it will be switched to "disconnected"
            # Lines with "0" in this vector are not impacted.
            if dict_["change_line_status"] is not None:
                self.line_change_status = dict_["change_line_status"]

    def _digest_redispatching(self, dict_):
        if "redispatch" in dict_:
            self.redispatch = dict_["redispatch"]

    def _digest_storage(self, dict_):
        if "set_storage" in dict_:
            self.storage_p = dict_["set_storage"]

    def _digest_curtailment(self, dict_):
        if "curtail" in dict_:
            self.curtail = dict_["curtail"]

    def _digest_alarm(self, dict_):
        """
        .. warning:: 
            /!\\\\ Only valid with "l2rpn_icaps_2021" environment /!\\\\"""
        if "raise_alarm" in dict_:
            self.raise_alarm = dict_["raise_alarm"]

    
    def _digest_alert(self, dict_):
        if "raise_alert" in dict_:
            self.raise_alert = dict_["raise_alert"]

    def _reset_vect(self):
        """
        INTERNAL USE ONLY

         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Need to be called when update is called !

        """
        self._vectorized = None
        self._subs_impacted = None
        self._lines_impacted = None

    def update(self, dict_):
        """
        Update the action with a comprehensible format specified by a dictionary.

        Preferably, if a key of the argument *dict_* is not found in :attr:`Action.authorized_keys` it should throw a
        warning. This argument will be completely ignored.

        This method also reset the attributes :attr:`Action._vectorized` :attr:`Action._lines_impacted` and
        :attr:`Action._subs_impacted` to ``None`` regardless of the argument in input.

        If an action consists of "reconnecting" a powerline, and this same powerline is affected by maintenance or a
        hazard, it will be erased without any warning. "hazards" and "maintenance" have the priority. This is a
        requirement for all proper :class:`Action` subclass.

        Parameters
        ----------
        dict_: :class:`dict`
            If it's ``None`` or empty it does nothing. Otherwise, it can contain the following (optional) keys:

            - "*injection*" if the action will modify the injections (generator setpoint/load value - active or
              reactive) of the powergrid. It has optionally one of the following keys:

                    - "load_p": to set the active load values (this is a numpy array with the same size as the number of
                      load in the power _grid with Nan: don't change anything, else set the value
                    - "load_q": same as above but for the load reactive values
                    - "prod_p": same as above but for the generator active setpoint values. It has the size
                      corresponding to the number of generators in the test case.
                    - "prod_v": same as above but set the voltage setpoint of generator units.

            - "*hazards*": represents the hazards that the line might suffer (boolean vector) False: no hazard, nothing
              is done, True: a hazard, the powerline is disconnected
            - "*maintenance*": represents the maintenance operation performed on each powerline (boolean vector) False:
              no maintenance, nothing is done, True: a maintenance is scheduled, the powerline is disconnected
            - "*set_line_status*": a vector (int or float) to set the status of the powerline status (connected /
              disconnected) with the following interpretation:

                - 0: nothing is changed,
                - -1: disconnect the powerline,
                - +1: reconnect the powerline. If an action consists in "reconnecting" a powerline, and this same
                  powerline is affected by a maintenance or a hazard, it will be erased without any warning. "hazards"
                  and "maintenance" have the priority.

            - "change_line_status": a vector (bool) to change the status of the powerline. This vector should be
              interpreted as:

                - ``False``: do nothing
                - ``True``: change the status of the powerline: disconnect it if it was connected, connect it if it was
                  disconnected

            - "set_bus": (numpy int vector or dictionary) will set the buses to which the objects are connected. It
              follows a similar interpretation than the line status vector:

                - 0 -> don't change anything
                - +1 -> set to bus 1,
                - +2 -> set to bus 2
                - +3 -> set to bus 3 (grid2op >= 1.10.0)
                - etc.
                - -1: You can use this method to disconnect an object by setting the value to -1.

            - "change_bus": (numpy bool vector or dictionary) will change the bus to which the object is connected.
              True will
              change it (eg switch it from bus 1 to bus 2 or from bus 2 to bus 1). NB this is only active if the system
              has only 2 buses per substation.
              .. versionchanged:: 1.10.0
                This feature is deactivated if `act.n_busbar_per_sub >= 3` or `act.n_busbar_per_sub == 1`

            - "redispatch": the best use of this is to specify either the numpy array of the redispatch vector you want
              to apply (that should have the size of the number of generators on the grid) or to specify a list of
              tuple, each tuple being 2 elements: first the generator ID, second the amount of redispatching,
              for example `[(1, -23), (12, +17)]`

            - "set_storage": the best use of this is to specify either the numpy array of the storage units vector
              you want
              to apply (that should have the size of the number of storage units on the grid) or to specify a list of
              tuple, each tuple being 2 elements: first the storage ID, second the amount of power you want to
              produce / absorb, for example `[(1, -23), (12, +17)]`

            **NB** the difference between "set_bus" and "change_bus" is the following:

              - If "set_bus" is 1, then the object (load, generator or powerline) will be moved to bus 1 of the
                substation to which it is connected. If it is already to bus 1 nothing will be done. If it's on another
                bus it will connect it to bus 1. It's disconnected, it will reconnect it and connect it to bus 1.
              - If "change_bus" is True, then objects will be moved from one bus to another. If the object were on bus 1
                then it will be moved on bus 2, and if it were on bus 2, it will be moved on bus 1. If the object is
                disconnected then the action is ambiguous, and calling it will throw an AmbiguousAction exception.
                
            - "curtail" : TODO
            - "raise_alarm" : TODO
            - "raise_alert": TODO

            **NB**: CHANGES: you can reconnect a powerline without specifying on each bus you reconnect it at both its
            ends. In that case the last known bus id for each its end is used.

            **NB**: if for a given powerline, both switch_line_status and set_line_status is set, the action will not
            be usable.
            This will lead to an :class:`grid2op.Exception.AmbiguousAction` exception.

            **NB**: The length of vectors provided here is NOT check in this function. This method can be "chained" and
            only on the final action, when used, eg. in the Backend, is checked.

            **NB**: If a powerline is disconnected, on maintenance, or suffer an outage, the associated "set_bus" will
            be ignored.
            Disconnection has the priority on anything. This priority is given because, in case of hazard, the hazard
            has the priority over the possible actions.

        Examples
        --------
        Here are short examples on how to update an action *eg.* how to create a valid :class:`Action` object that
        be used to modify a :class:`grid2op.Backend.Backend`.

        In all the following examples, we suppose that a valid grid2op environment is created, for example with:
        
        .. code-block:: python

            import grid2op
            from grid2op.Action import BaseAction
            env_name = "l2rpn_case14_sandbox"
            # create a simple environment
            # and make sure every type of action can be used.
            env = grid2op.make(env_name, action_class=BaseAction)

        *Example 1*: modify the load active values to set them all to 1. You can replace "load_p" by "load_q",
        "prod_p" or "prod_v" to change the load reactive value, the generator active setpoint or the generator
        voltage magnitude setpoint.

        .. code-block:: python

            new_load = np.ones(env.action_space.n_load)
            modify_load_active_value = env.action_space({"injection": {"load_p": new_load}})
            print(modify_load_active_value)

        *Example 2*: disconnect the powerline of id 1:

        .. code-block:: python

            disconnect_powerline = env.action_space({"set_line_status": [(1, -1)]})
            print(disconnect_powerline)
            # there is a shortcut to do that:
            disconnect_powerline2 = env.disconnect_powerline(line_id=1)

        *Example 3*: force the reconnection of the powerline of id 5 by connected it to bus 1 on its origin side and
        bus 2 on its extremity side.

        .. code-block:: python

            reconnect_powerline = env.action_space({"set_line_status": [(5, 1)],
                                                    "set_bus": {"lines_or_id": [(5, 1)]},
                                                    "set_bus": {"lines_ex_id": [(5, 2)]}
                                                     })
            print(reconnect_powerline)
            # and the shorter method:
            reconnect_powerline = env.action.space.reconnect_powerline(line_id=5, bus_or=1, bus_ex=2)

        *Example 4*: change the bus to which load 4 is connected:

        .. code-block:: python

            change_load_bus = env.action_space({"set_bus": {"loads_id": [(4, 1)]} })
            print(change_load_bus)

        *Example 5*: reconfigure completely substation 5, and connect the first 3 elements to bus 1 and the last 3
        elements to bus 2

        .. code-block:: python

            sub_id = 5
            target_topology = np.ones(env.sub_info[sub_id], dtype=dt_int)
            target_topology[3:] = 2
            reconfig_sub = env.action_space({"set_bus": {"substations_id": [(sub_id, target_topology)] } })
            print(reconfig_sub)

        *Example 6*: apply redispatching of +17.42 MW at generator with id 23 and -27.8 at generator with id 1

        .. code-block:: python

            redisp_act = env.action_space({"redispatch": [(23, +17.42), (23, -27.8)]})
            print(redisp_act)

        *Example 7*: apply an action on a storage unit: have the storage unit of id 0 produce 1.5MW

        .. code-block:: python

            storage_act = env.action_space({"set_storage": [(0, -1.5)]})
            print(storage_act)

        *Example 8*: apply a action of type curtailment: limit the production to a renewable energy unit
        (in the example the generator with id 2)
        at 80% of its maximum capacity

        .. code-block:: python

            renewable_energy_source = 2
            storage_act = env.action_space({"curtail": [(renewable_energy_source, 0.8)]})
            print(storage_act)

        Returns
        -------
        self: :class:`BaseAction`
            Return the modified instance. This is handy to chain modifications if needed.

        """
        self._reset_vect()

        if dict_ is not None:
            for kk in dict_.keys():
                if kk not in self.authorized_keys:
                    warn = 'The key "{}" used to update an action will be ignored. Valid keys are {}'
                    warn = warn.format(kk, self.authorized_keys)
                    warnings.warn(warn)

            self._digest_shunt(dict_)
            self._digest_injection(dict_)
            self._digest_redispatching(dict_)
            self._digest_storage(dict_)  # ADDED for battery
            self._digest_curtailment(dict_)  # ADDED for curtailment
            self._digest_setbus(dict_)
            self._digest_change_bus(dict_)
            self._digest_set_status(dict_)
            self._digest_hazards(dict_)
            self._digest_maintenance(dict_)
            self._digest_change_status(dict_)
            self._digest_alarm(dict_)
            self._digest_alert(dict_)

        return self

    def is_ambiguous(self) -> Tuple[bool, AmbiguousAction]:
        """
        Says if the action, as defined is ambiguous *per se* or not.

        See definition of :func:`BaseAction.check_space_legit` for more details about *ambiguity per se*.

        Returns
        -------
        res: ``True`` if the action is ambiguous, ``False`` otherwise.

        info: ``dict`` or not
            More information about the error. If the action is not ambiguous, it values to ``None``
        """
        try:
            self._check_for_ambiguity()
            res = False
            info = None
        except AmbiguousAction as exc_:
            info = exc_
            res = True
        return res, info

    def _check_for_correct_modif_flags(self):
        if self._dict_inj:
            if not self._modif_inj:
                raise AmbiguousAction(
                    "A action on the injection is performed while the appropriate flag is not "
                    "set. Please use the official grid2op action API to modify the injections."
                )
            if "injection" not in self.authorized_keys:
                raise IllegalAction("You illegally act on the injection")
        if self._change_bus_vect.any():
            if not self._modif_change_bus:
                raise AmbiguousAction(
                    "A action of type change_bus is performed while the appropriate flag is not "
                    "set. Please use the official grid2op action API to modify the bus using "
                    "'change'."
                )
            if "change_bus" not in self.authorized_keys:
                raise IllegalAction("You illegally act on the bus (using change)")
        if (self._set_topo_vect != 0).any():
            if not self._modif_set_bus:
                raise AmbiguousAction(
                    "A action of type set_bus is performed while the appropriate flag is not "
                    "set. Please use the official grid2op action API to modify the bus using "
                    "'set'."
                )
            if "set_bus" not in self.authorized_keys:
                raise IllegalAction("You illegally act on the bus (using set)")

        if (self._set_line_status != 0).any():
            if not self._modif_set_status:
                raise AmbiguousAction(
                    "A action of type line_set_status is performed while the appropriate flag is not "
                    "set. Please use the official grid2op action API to modify the status of "
                    "powerline using "
                    "'set'."
                )
            if "set_line_status" not in self.authorized_keys:
                raise IllegalAction(
                    "You illegally act on the powerline status (using set)"
                )

        if (self._switch_line_status).any():
            if not self._modif_change_status:
                raise AmbiguousAction(
                    "A action of type line_change_status is performed while the appropriate flag "
                    "is not "
                    "set. Please use the official grid2op action API to modify the status of "
                    "powerlines using 'change'."
                )
            if "change_line_status" not in self.authorized_keys:
                raise IllegalAction(
                    "You illegally act on the powerline status (using change)"
                )

        if (np.abs(self._redispatch) >= 1e-7).any():
            if not self._modif_redispatch:
                raise AmbiguousAction(
                    "A action of type redispatch is performed while the appropriate flag "
                    "is not "
                    "set. Please use the official grid2op action API to perform redispatching "
                    "action."
                )
            if "redispatch" not in self.authorized_keys:
                raise IllegalAction("You illegally act on the redispatching")

        if (np.abs(self._storage_power) >= 1e-7).any():
            if not self._modif_storage:
                raise AmbiguousAction(
                    "A action on the storage unit is performed while the appropriate flag "
                    "is not "
                    "set. Please use the official grid2op action API to perform "
                    "action on storage unit."
                )
            if "set_storage" not in self.authorized_keys:
                raise IllegalAction("You illegally act on the storage unit")

        if (np.abs(self._curtail + 1.0) >= 1e-7).any():
            if not self._modif_curtailment:
                raise AmbiguousAction(
                    "A curtailment is performed while the action is not supposed to have done so. "
                    "Please use the official grid2op action API to perform curtailment action."
                )
            if "curtail" not in self.authorized_keys:
                raise IllegalAction("You illegally act on the curtailment")

        if (self._raise_alarm).any():
            if not self._modif_alarm:
                raise AmbiguousAction(
                    "Incorrect way to raise some alarm, the appropriate flag is not "
                    "modified properly."
                )
            if "raise_alarm" not in self.authorized_keys:
                raise IllegalAction("You illegally send an alarm.")

        if (self._raise_alert).any():
            if not self._modif_alert:
                raise AmbiguousActionRaiseAlert(
                    "Incorrect way to raise some alert, the appropriate flag is not "
                    "modified properly."
                )
            if "raise_alert" not in self.authorized_keys:
                raise IllegalAction("You illegally send an alert.")

    def _check_for_ambiguity(self):
        """
        This method checks if an action is ambiguous or not. If the instance is ambiguous, an
        :class:`grid2op.Exceptions.AmbiguousAction` is raised.

        An action can be ambiguous in the following context:

          - It incorrectly affects the injections:

            - :code:`self._dict_inj["load_p"]` doesn't have the same size as the number of loads on the _grid.
            - :code:`self._dict_inj["load_q"]` doesn't have the same size as the number of loads on the _grid.
            - :code:`self._dict_inj["prod_p"]` doesn't have the same size as the number of loads on the _grid.
            - :code:`self._dict_inj["prod_v"]` doesn't have the same size as the number of loads on the _grid.

          - It affects the powerline in an incorrect manner:

            - :code:`self._switch_line_status` has not the same size as the number of powerlines
            - :code:`self._set_line_status` has not the same size as the number of powerlines
            - the status of some powerline is both *changed* (:code:`self._switch_line_status[i] == True` for some *i*)
              and *set* (:code:`self._set_line_status[i]` for the same *i* is not 0)
            - a powerline is both connected at one end (ex. its origin is set to bus 1) and disconnected at another
              (its extremity is set to bus -1)

          - It has an ambiguous behavior concerning the topology of some substations

            - the state of some bus for some element is both *changed* (:code:`self._change_bus_vect[i] = True` for
              some *i*) and *set* (:code:`self._set_topo_vect[i]` for the same *i* is not 0)
            - :code:`self._set_topo_vect` has not the same dimension as the number of elements on the powergrid
            - :code:`self._change_bus_vect` has not the same dimension as the number of elements on the powergrid

          - For redispatching, Ambiguous actions can come from:

            - Some redispatching action is active, yet
              :attr:`grid2op.Space.GridObjects.redispatching_unit_commitment_availble` is set to ``False``
            - the length of the redispatching vector :attr:`BaseAction._redispatching` is not compatible with the number
              of generators.
            - some redispatching are above the maximum ramp up :attr:`grid2op.Space.GridObjects.gen_max_ramp_up`
            - some redispatching are below the maximum ramp down :attr:`grid2op.Space.GridObjects.gen_max_ramp_down`
            - the redispatching action affect non dispatchable generators
            - the redispatching and the production setpoint, if added, are above pmax for at least a generator
            - the redispatching and the production setpoint, if added, are below pmin for at least a generator

        In case of need to overload this method, it is advise to still call this one from the base :class:`BaseAction`
        with ":code:`super()._check_for_ambiguity()`" or ":code:`BaseAction._check_for_ambiguity(self)`".

        Raises
        -------
        :class:`grid2op.Exceptions.AmbiguousAction`
            Or any of its more precise subclasses, depending on which assumption is not met.


        """
        # check that the correct flags are properly computed
        self._check_for_correct_modif_flags()
        cls = type(self)
        
        if (
            self._modif_change_status
            and self._modif_set_status
            and (self._set_line_status[self._switch_line_status] != 0).any()
        ):
            raise InvalidLineStatus(
                "You asked to change the status (connected / disconnected) of a powerline by"
                ' using the keyword "change_status" and set this same line state in '
                '"set_status" '
                '(or "hazard" or "maintenance"). This ambiguous behaviour is not supported'
            )
        # check size
        if self._modif_inj:
            if "load_p" in self._dict_inj:
                if len(self._dict_inj["load_p"]) != cls.n_load:
                    raise InvalidNumberOfLoads(
                        "This action acts on {} loads while there are {} "
                        "in the _grid".format(
                            len(self._dict_inj["load_p"]), cls.n_load
                        )
                    )
            if "load_q" in self._dict_inj:
                if len(self._dict_inj["load_q"]) != cls.n_load:
                    raise InvalidNumberOfLoads(
                        "This action acts on {} loads while there are {} in "
                        "the _grid".format(len(self._dict_inj["load_q"]), cls.n_load)
                    )
            if "prod_p" in self._dict_inj:
                if len(self._dict_inj["prod_p"]) != cls.n_gen:
                    raise InvalidNumberOfGenerators(
                        "This action acts on {} generators while there are {} in "
                        "the _grid".format(len(self._dict_inj["prod_p"]), cls.n_gen)
                    )
            if "prod_v" in self._dict_inj:
                if len(self._dict_inj["prod_v"]) != cls.n_gen:
                    raise InvalidNumberOfGenerators(
                        "This action acts on {} generators while there are {} in "
                        "the _grid".format(len(self._dict_inj["prod_v"]), cls.n_gen)
                    )

        if len(self._switch_line_status) != cls.n_line:
            raise InvalidNumberOfLines(
                "This action acts on {} lines while there are {} in "
                "the _grid".format(len(self._switch_line_status), cls.n_line)
            )

        if len(self._set_topo_vect) != cls.dim_topo:
            raise InvalidNumberOfObjectEnds(
                "This action acts on {} ends of object while there are {} "
                "in the _grid".format(len(self._set_topo_vect), cls.dim_topo)
            )
        if len(self._change_bus_vect) != cls.dim_topo:
            raise InvalidNumberOfObjectEnds(
                "This action acts on {} ends of object while there are {} "
                "in the _grid".format(len(self._change_bus_vect), cls.dim_topo)
            )

        if len(self._redispatch) != cls.n_gen:
            raise InvalidNumberOfGenerators(
                "This action acts on {} generators (redispatching= while "
                "there are {} in the grid".format(len(self._redispatch), cls.n_gen)
            )

        # redispatching specific check
        if self._modif_redispatch:
            if "redispatch" not in cls.authorized_keys:
                raise AmbiguousAction(
                    'Action of type "redispatch" are not supported by this action type'
                )
            if not self.redispatching_unit_commitment_availble:
                raise UnitCommitorRedispachingNotAvailable(
                    "Impossible to use a redispatching action in this "
                    "environment. Please set up the proper costs for generator"
                )

            if (np.abs(self._redispatch[~cls.gen_redispatchable]) >= 1e-7).any():
                raise InvalidRedispatching(
                    "Trying to apply a redispatching action on a non redispatchable generator"
                )

            if self._single_act:
                if (self._redispatch > cls.gen_max_ramp_up).any():
                    raise InvalidRedispatching(
                        "Some redispatching amount are above the maximum ramp up"
                    )
                if (-self._redispatch > cls.gen_max_ramp_down).any():
                    raise InvalidRedispatching(
                        "Some redispatching amount are bellow the maximum ramp down"
                    )

                if "prod_p" in self._dict_inj:
                    new_p = self._dict_inj["prod_p"]
                    tmp_p = new_p + self._redispatch
                    indx_ok = np.isfinite(new_p)
                    if (tmp_p[indx_ok] > cls.gen_pmax[indx_ok]).any():
                        raise InvalidRedispatching(
                            "Some redispatching amount, cumulated with the production setpoint, "
                            "are above pmax for some generator."
                        )
                    if (tmp_p[indx_ok] < cls.gen_pmin[indx_ok]).any():
                        raise InvalidRedispatching(
                            "Some redispatching amount, cumulated with the production setpoint, "
                            "are below pmin for some generator."
                        )

        # storage specific checks:
        self._is_storage_ambiguous()

        # curtailment specific checks:
        self._is_curtailment_ambiguous()

        # topological action
        if (
            self._modif_set_bus
            and self._modif_change_bus
            and (self._set_topo_vect[self._change_bus_vect] != 0).any()
        ):
            raise InvalidBusStatus(
                "You asked to change the bus of an object with"
                ' using the keyword "change_bus" and set this same object state in "set_bus"'
                ". This ambiguous behaviour is not supported"
            )
        if self._modif_set_bus and (self._set_topo_vect < -1).any():
            raise InvalidBusStatus(
                "Invalid set_bus. Buses should be either -1 (disconnect), 0 (change nothing),"
                "1 (assign this object to bus one) or 2 (assign this object to bus"
                "2). A negative number has been found."
            )
        if self._modif_set_bus and (self._set_topo_vect > cls.n_busbar_per_sub).any():
            raise InvalidBusStatus(
                "Invalid set_bus. Buses should be either -1 (disconnect), 0 (change nothing),"
                "1 (assign this object to bus one) or 2 (assign this object to bus"
                "2). A number higher than 2 has been found: substations with more than 2 busbars"
                "are not supported by grid2op at the moment. Do not hesitate to fill a feature "
                "request on github if you need this feature."
            )

        if False:
            # TODO find an elegant way to disable that
            # now it's possible.
            for q_id, status in enumerate(self._set_line_status):
                if status == 1:
                    # i reconnect a powerline, i need to check that it's connected on both ends
                    if (
                        self._set_topo_vect[self.line_or_pos_topo_vect[q_id]] == 0
                        or self._set_topo_vect[self.line_ex_pos_topo_vect[q_id]] == 0
                    ):

                        raise InvalidLineStatus(
                            "You ask to reconnect powerline {} yet didn't tell on"
                            " which bus.".format(q_id)
                        )

        if self._modif_set_bus:
            disco_or = self._set_topo_vect[cls.line_or_pos_topo_vect] == -1
            if (self._set_topo_vect[cls.line_ex_pos_topo_vect][disco_or] > 0).any():
                raise InvalidLineStatus(
                    "A powerline is connected (set to a bus at extremity side) and "
                    "disconnected (set to bus -1 at origin side)"
                )
            disco_ex = self._set_topo_vect[cls.line_ex_pos_topo_vect] == -1
            if (self._set_topo_vect[cls.line_or_pos_topo_vect][disco_ex] > 0).any():
                raise InvalidLineStatus(
                    "A powerline is connected (set to a bus at origin side) and "
                    "disconnected (set to bus -1 at extremity side)"
                )

        # if i disconnected of a line, but i modify also the bus where it's connected
        if self._modif_set_bus or self._modif_change_bus:
            idx = self._set_line_status == -1
            id_disc = (idx).nonzero()[0]
            
            idx2 = self._set_line_status == 1
            id_reco = (idx2).nonzero()[0]

        if self._modif_set_bus:
            if "set_bus" not in cls.authorized_keys:
                raise AmbiguousAction(
                    'Action of type "set_bus" are not supported by this action type'
                )
            if (
                self._set_topo_vect[cls.line_or_pos_topo_vect[id_disc]] > 0
            ).any() or (self._set_topo_vect[cls.line_ex_pos_topo_vect[id_disc]] > 0).any():
                raise InvalidLineStatus(
                    "You ask to disconnect a powerline but also to connect it "
                    "to a certain bus."
                )
                
            if (
                self._set_topo_vect[cls.line_or_pos_topo_vect[id_reco]] == -1
            ).any() or (self._set_topo_vect[cls.line_ex_pos_topo_vect[id_reco]] == -1).any():
                raise InvalidLineStatus(
                    "You ask to reconnect a powerline but also to disconnect it "
                    "from a certain bus."
                )
        if self._modif_change_bus:
            if "change_bus" not in cls.authorized_keys:
                raise AmbiguousAction(
                    'Action of type "change_bus" are not supported by this action type'
                )
            if (
                self._change_bus_vect[cls.line_or_pos_topo_vect[id_disc]] > 0
            ).any() or (self._change_bus_vect[cls.line_ex_pos_topo_vect[id_disc]] > 0).any():
                raise InvalidLineStatus(
                    "You ask to disconnect a powerline but also to change its bus."
                )

            if (
                self._change_bus_vect[
                    cls.line_or_pos_topo_vect[self._set_line_status == 1]
                ]
            ).any():
                raise InvalidLineStatus(
                    "You ask to connect an origin powerline but also to *change* the bus  to which "
                    "it  is connected. This is ambiguous. You must *set* this bus instead."
                )
            if (
                self._change_bus_vect[
                    cls.line_ex_pos_topo_vect[self._set_line_status == 1]
                ]
            ).any():
                raise InvalidLineStatus(
                    "You ask to connect an extremity powerline but also to *change* the bus  to "
                    "which it is connected. This is ambiguous. You must *set* this bus instead."
                )

        if cls.shunts_data_available:
            if self.shunt_p.shape[0] != cls.n_shunt:
                raise IncorrectNumberOfElements(
                    "Incorrect number of shunt (for shunt_p) in your action."
                )
            if self.shunt_q.shape[0] != cls.n_shunt:
                raise IncorrectNumberOfElements(
                    "Incorrect number of shunt (for shunt_q) in your action."
                )
            if self.shunt_bus.shape[0] != cls.n_shunt:
                raise IncorrectNumberOfElements(
                    "Incorrect number of shunt (for shunt_bus) in your action."
                )
            if cls.n_shunt > 0:
                if np.max(self.shunt_bus) > cls.n_busbar_per_sub:
                    raise AmbiguousAction(
                        "Some shunt is connected to a bus greater than 2"
                    )
                if np.min(self.shunt_bus) < -1:
                    raise AmbiguousAction(
                        "Some shunt is connected to a bus smaller than -1"
                    )
        else:
            # shunt is not available
            if self.shunt_p is not None:
                raise AmbiguousAction(
                    "Attempt to modify a shunt (shunt_p) while shunt data is not handled by backend"
                )
            if self.shunt_q is not None:
                raise AmbiguousAction(
                    "Attempt to modify a shunt (shunt_q) while shunt data is not handled by backend"
                )
            if self.shunt_bus is not None:
                raise AmbiguousAction(
                    "Attempt to modify a shunt (shunt_bus) while shunt data is not handled "
                    "by backend"
                )

        if self._modif_alarm:
            if self._raise_alarm.shape[0] != cls.dim_alarms:
                raise AmbiguousAction(
                    f"Wrong number of alarm raised: {self._raise_alarm.shape[0]} raised, expecting "
                    f"{cls.dim_alarms}"
                )
        else:
            if self._raise_alarm.any():
                raise AmbiguousAction(
                    f"Unrecognize alarm action: an action acts on the alarm, yet it's not tagged "
                    f"as doing so. Expect wrong behaviour."
                )

        if self._modif_alert:
            if self._raise_alert.shape[0] != cls.dim_alerts:
                raise AmbiguousActionRaiseAlert(
                    f"Wrong number of alert raised: {self._raise_alert.shape[0]} raised, expecting "
                    f"{cls.dim_alerts}"
                )
        else:
            if self._raise_alert.any():
                raise AmbiguousActionRaiseAlert(
                    "Unrecognize alert action: an action acts on the alert, yet it's not tagged "
                    "as doing so. Expect wrong behaviour."
                )

    def _is_storage_ambiguous(self):
        """check if storage actions are ambiguous"""
        cls = type(self)
        if self._modif_storage:
            if "set_storage" not in cls.authorized_keys:
                raise AmbiguousAction(
                    'Action of type "set_storage" are not supported by this action type'
                )
            if cls.n_storage == 0:
                raise InvalidStorage(
                    "Attempt to modify a storage unit while there is none on the grid"
                )
            if self._storage_power.shape[0] != cls.n_storage:
                raise InvalidStorage(
                    "self._storage_power.shape[0] != self.n_storage: wrong number of storage "
                    "units affected"
                )
            if (self._storage_power < -cls.storage_max_p_prod).any():
                where_bug = (self._storage_power < -cls.storage_max_p_prod).nonzero()[0]
                raise InvalidStorage(
                    f"you asked a storage unit to absorb more than what it can: "
                    f"self._storage_power[{where_bug}] < -self.storage_max_p_prod[{where_bug}]."
                )
            if (self._storage_power > cls.storage_max_p_absorb).any():
                where_bug = (self._storage_power > cls.storage_max_p_absorb).nonzero()[0]
                raise InvalidStorage(
                    f"you asked a storage unit to produce more than what it can: "
                    f"self._storage_power[{where_bug}] > self.storage_max_p_absorb[{where_bug}]."
                )

        if "_storage_power" not in cls.attr_list_set:
            if (self._set_topo_vect[cls.storage_pos_topo_vect] > 0).any():
                raise InvalidStorage("Attempt to modify bus (set) of a storage unit")
            if (self._change_bus_vect[cls.storage_pos_topo_vect]).any():
                raise InvalidStorage("Attempt to modify bus (change) of a storage unit")

    def _is_curtailment_ambiguous(self):
        """check if curtailment action is ambiguous"""
        cls = type(self)
        if self._modif_curtailment:
            if "curtail" not in cls.authorized_keys:
                raise AmbiguousAction(
                    'Action of type "curtail" are not supported by this action type'
                )

            if not cls.redispatching_unit_commitment_availble:
                raise UnitCommitorRedispachingNotAvailable(
                    "Impossible to use a redispatching action in this "
                    "environment. Please set up the proper costs for generator. "
                    "This also means curtailment feature is not available."
                )

            if self._curtail.shape[0] != cls.n_gen:
                raise InvalidCurtailment(
                    "self._curtail.shape[0] != self.n_gen: wrong number of generator "
                    "units affected"
                )

            if ((self._curtail < 0.0) & (np.abs(self._curtail + 1.0) >= 1e-7)).any():
                where_bug = ((self._curtail < 0.0) & (np.abs(self._curtail + 1.0) >= 1e-7)).nonzero()[0]
                raise InvalidCurtailment(
                    f"you asked to perform a negative curtailment: "
                    f"self._curtail[{where_bug}] < 0. "
                    f"Curtailment should be a real number between 0.0 and 1.0"
                )
            if (self._curtail > 1.0).any():
                where_bug = (self._curtail > 1.0).nonzero()[0]
                raise InvalidCurtailment(
                    f"you asked a storage unit to produce more than what it can: "
                    f"self._curtail[{where_bug}] > 1. "
                    f"Curtailment should be a real number between 0.0 and 1.0"
                )
            if (np.abs(self._curtail[~cls.gen_renewable] +1.0) >= 1e-7).any():
                raise InvalidCurtailment(
                    "Trying to apply a curtailment on a non renewable generator"
                )

    def _ignore_topo_action_if_disconnection(self, sel_):
        # force ignore of any topological actions
        self._set_topo_vect[np.array(self.line_or_pos_topo_vect[sel_])] = 0
        self._change_bus_vect[np.array(self.line_or_pos_topo_vect[sel_])] = False
        self._set_topo_vect[np.array(self.line_ex_pos_topo_vect[sel_])] = 0
        self._change_bus_vect[np.array(self.line_ex_pos_topo_vect[sel_])] = False

    def _aux_obj_caract(self, id_, with_name, xxx_pos_topo_vect, objt_type, xxx_subid, name_xxx):
        for l_id, id_in_topo in enumerate(xxx_pos_topo_vect):
            if id_in_topo == id_:
                obj_id = l_id
                obj_name = name_xxx[l_id]
                substation_id = xxx_subid[obj_id]
                if not with_name:
                    return obj_id, objt_type, substation_id 
                return obj_id, objt_type, substation_id, obj_name
        return None
        
    def _aux_obj_caract_from_topo_id_load(self, cls, id_, with_name):
        return self._aux_obj_caract(id_, with_name, cls.load_pos_topo_vect, "load", cls.load_to_subid, cls.name_load)

    def _aux_obj_caract_from_topo_id_gen(self, cls, id_, with_name):
        return self._aux_obj_caract(id_, with_name, cls.gen_pos_topo_vect,
                                    "generator", cls.gen_to_subid, cls.name_gen)
        
    def _aux_obj_caract_from_topo_id_lor(self, cls, id_, with_name):
        return self._aux_obj_caract(id_, with_name, cls.line_or_pos_topo_vect,
                                    self._line_or_str, cls.line_or_to_subid, cls.name_line)
        
    def _aux_obj_caract_from_topo_id_lex(self, cls, id_, with_name):
        return self._aux_obj_caract(id_, with_name, cls.line_ex_pos_topo_vect,
                                    self._line_ex_str, cls.line_ex_to_subid, cls.name_line)
        
    def _aux_obj_caract_from_topo_storage(self, cls, id_, with_name):
        return self._aux_obj_caract(id_, with_name, cls.storage_pos_topo_vect,
                                    "storage", cls.storage_to_subid, cls.name_storage)
        
    def _obj_caract_from_topo_id(self, id_, with_name=False):
        # TODO refactor this with gridobj.topo_vect_element
        cls = type(self)
        tmp = self._aux_obj_caract_from_topo_id_load(cls, id_, with_name)
        if tmp is not None:
            return tmp
        tmp = self._aux_obj_caract_from_topo_id_gen(cls, id_, with_name)
        if tmp is not None:
            return tmp
        tmp = self._aux_obj_caract_from_topo_id_lor(cls, id_, with_name)
        if tmp is not None:
            return tmp
        tmp = self._aux_obj_caract_from_topo_id_lex(cls, id_, with_name)
        if tmp is not None:
            return tmp
        tmp = self._aux_obj_caract_from_topo_storage(cls, id_, with_name)
        if tmp is not None:
            return tmp
        raise Grid2OpException(f"Unknown element in topovect with id {id_}")

    def __str__(self) -> str:
        """
        This utility allows printing in a human-readable format what objects will be impacted by the action.

        Returns
        -------
        str: :class:`str`
            The string representation of an :class:`BaseAction` in a human-readable format.

        Examples
        ---------

        It is simply the "print" function:

        .. code-block:: python

            action = env.action_space(...)
            print(action)

        """
        res = ["This action will:"]
        impact = self.impact_on_objects()

        # injections
        injection_impact = impact["injection"]
        if injection_impact["changed"]:
            for change in injection_impact["impacted"]:
                res.append("\t - Set {} to {}".format(change["set"], change["to"]))
        else:
            res.append("\t - NOT change anything to the injections")

        # redispatch
        if self._modif_redispatch:
            res.append(
                "\t - Modify the generators with redispatching in the following way:"
            )
            for gen_idx in range(self.n_gen):
                if np.abs(self._redispatch[gen_idx]) >= 1e-7:
                    gen_name = self.name_gen[gen_idx]
                    r_amount = self._redispatch[gen_idx]
                    res.append(
                        '\t \t - Redispatch "{}" of {:.2f} MW'.format(
                            gen_name, r_amount
                        )
                    )
        else:
            res.append("\t - NOT perform any redispatching action")

        # storage
        if self._modif_storage:
            res.append("\t - Modify the storage units in the following way:")
            for stor_idx in range(self.n_storage):
                amount_ = self._storage_power[stor_idx]
                if np.isfinite(amount_) and np.abs(amount_) >= 1e-7:
                    name_ = self.name_storage[stor_idx]
                    res.append(
                        '\t \t - Ask unit "{}" to {} {:.2f} MW (setpoint: {:.2f} MW)'
                        "".format(
                            name_,
                            "absorb" if amount_ > 0.0 else "produce",
                            np.abs(amount_),
                            amount_,
                        )
                    )
        else:
            res.append("\t - NOT modify any storage capacity")

        # curtailment
        if self._modif_curtailment:
            res.append("\t - Perform the following curtailment:")
            for gen_idx in range(self.n_gen):
                amount_ = self._curtail[gen_idx]
                if np.isfinite(amount_) and np.abs(amount_ + 1.0) >= 1e-7:
                    name_ = self.name_gen[gen_idx]
                    res.append(
                        '\t \t - Limit unit "{}" to {:.1f}% of its Pmax (setpoint: {:.3f})'
                        "".format(name_, 100.0 * amount_, amount_)
                    )
        else:
            res.append("\t - NOT perform any curtailment")

        # force line status
        force_line_impact = impact["force_line"]
        if force_line_impact["changed"]:
            reconnections = force_line_impact["reconnections"]
            if reconnections["count"] > 0:
                res.append(
                    "\t - Force reconnection of {} powerlines ({})".format(
                        reconnections["count"], reconnections["powerlines"]
                    )
                )

            disconnections = force_line_impact["disconnections"]
            if disconnections["count"] > 0:
                res.append(
                    "\t - Force disconnection of {} powerlines ({})".format(
                        disconnections["count"], disconnections["powerlines"]
                    )
                )
        else:
            res.append("\t - NOT force any line status")

        # swtich line status
        swith_line_impact = impact["switch_line"]
        if swith_line_impact["changed"]:
            res.append(
                "\t - Switch status of {} powerlines ({})".format(
                    swith_line_impact["count"], swith_line_impact["powerlines"]
                )
            )
        else:
            res.append("\t - NOT switch any line status")

        # topology
        bus_switch_impact = impact["topology"]["bus_switch"]
        if len(bus_switch_impact) > 0:
            res.append("\t - Change the bus of the following element(s):")
            for switch in bus_switch_impact:
                res.append(
                    "\t \t - Switch bus of {} id {} [on substation {}]".format(
                        switch["object_type"], switch["object_id"], switch["substation"]
                    )
                )
        else:
            res.append("\t - NOT switch anything in the topology")

        assigned_bus_impact = impact["topology"]["assigned_bus"]
        disconnect_bus_impact = impact["topology"]["disconnect_bus"]
        if len(assigned_bus_impact) > 0 or len(disconnect_bus_impact) > 0:
            if assigned_bus_impact:
                res.append("\t - Set the bus of the following element(s):")
            for assigned in assigned_bus_impact:
                res.append(
                    "\t \t - Assign bus {} to {} id {} [on substation {}]".format(
                        assigned["bus"],
                        assigned["object_type"],
                        assigned["object_id"],
                        assigned["substation"],
                    )
                )
            if disconnect_bus_impact:
                res.append("\t - Disconnect the following element(s):")
            for disconnected in disconnect_bus_impact:
                res.append(
                    "\t \t - Disconnect {} id {} [on substation {}]".format(
                        disconnected["object_type"],
                        disconnected["object_id"],
                        disconnected["substation"],
                    )
                )
        else:
            res.append("\t - NOT force any particular bus configuration")

        my_cls = type(self)
        if my_cls.dim_alarms > 0:
            if self._modif_alarm:
                li_area = np.array(my_cls.alarms_area_names)[
                    (self._raise_alarm).nonzero()[0]
                ]
                if len(li_area) == 1:
                    area_str = ": " + li_area[0]
                else:
                    area_str = "s: \n\t \t - " + "\n\t \t - ".join(li_area)
                res.append(f"\t - Raise an alarm on area" f"{area_str}")
            else:
                res.append("\t - Not raise any alarm")
        
        if my_cls.dim_alerts > 0:
            if self._modif_alert:
                i_alert = (self._raise_alert).nonzero()[0]
                li_line = np.array(my_cls.alertable_line_names)[i_alert]
                if len(li_line) == 1:
                    line_str = f": {i_alert[0]} (on line {li_line[0]})"
                else:
                    line_str = "s: \n\t \t - " + "\n\t \t - ".join(
                        [f": {i} (on line {l})" for i,l in zip(i_alert,li_line)])
                res.append(f"\t - Raise alert(s) {line_str}")
            else:
                res.append("\t - Not raise any alert")
        return "\n".join(res)

    def impact_on_objects(self) -> dict:
        """
        This will return a dictionary which contains details on objects that will be impacted by the action.

        Returns
        -------
        dict: :class:`dict`
            The dictionary representation of an action impact on objects with keys, "has_impact", "injection",
            "force_line", "switch_line", "topology", "redispatch", "storage", "curtailment".

        """
        # handles actions on injections
        has_impact = False

        inject_detail = {"changed": False, "count": 0, "impacted": []}
        for k in ["load_p", "prod_p", "load_q", "prod_v"]:
            if k in self._dict_inj:
                inject_detail["changed"] = True
                has_impact = True
                inject_detail["count"] += 1
                inject_detail["impacted"].append({"set": k, "to": self._dict_inj[k]})

        # handles actions on force line status
        force_line_status = {
            "changed": False,
            "reconnections": {"count": 0, "powerlines": []},
            "disconnections": {"count": 0, "powerlines": []},
        }
        if (self._set_line_status == 1).any():
            force_line_status["changed"] = True
            has_impact = True
            force_line_status["reconnections"]["count"] = (
                self._set_line_status == 1
            ).sum()
            force_line_status["reconnections"]["powerlines"] = (
                (self._set_line_status == 1).nonzero()[0])

        if (self._set_line_status == -1).any():
            force_line_status["changed"] = True
            has_impact = True
            force_line_status["disconnections"]["count"] = (
                self._set_line_status == -1
            ).sum()
            force_line_status["disconnections"]["powerlines"] = (
                (self._set_line_status == -1).nonzero()[0]
            )

        # handles action on swtich line status
        switch_line_status = {"changed": False, "count": 0, "powerlines": []}
        if self._switch_line_status.sum():
            switch_line_status["changed"] = True
            has_impact = True
            switch_line_status["count"] = self._switch_line_status.sum()
            switch_line_status["powerlines"] = (self._switch_line_status).nonzero()[0]

        topology = {
            "changed": False,
            "bus_switch": [],
            "assigned_bus": [],
            "disconnect_bus": [],
        }
        # handles topology
        if self._change_bus_vect.any():
            for id_, k in enumerate(self._change_bus_vect):
                if k:
                    obj_id, objt_type, substation_id = self._obj_caract_from_topo_id(
                        id_
                    )
                    topology["bus_switch"].append(
                        {
                            "bus": k,
                            "object_type": objt_type,
                            "object_id": obj_id,
                            "substation": substation_id,
                        }
                    )
            topology["changed"] = True
            has_impact = True

        if (self._set_topo_vect != 0).any():
            for id_, k in enumerate(self._set_topo_vect):
                if k > 0:
                    obj_id, objt_type, substation_id = self._obj_caract_from_topo_id(
                        id_
                    )
                    topology["assigned_bus"].append(
                        {
                            "bus": k,
                            "object_type": objt_type,
                            "object_id": obj_id,
                            "substation": substation_id,
                        }
                    )

                if k < 0:
                    obj_id, objt_type, substation_id = self._obj_caract_from_topo_id(
                        id_
                    )
                    topology["disconnect_bus"].append(
                        {
                            "bus": k,
                            "object_type": objt_type,
                            "object_id": obj_id,
                            "substation": substation_id,
                        }
                    )
            topology["changed"] = True
            has_impact = True

        # handle redispatching
        redispatch = {"changed": False, "generators": []}
        if (np.abs(self._redispatch) >= 1e-7).any():
            for gen_idx in range(self.n_gen):
                if np.abs(self._redispatch[gen_idx]) >= 1e-7:
                    gen_name = self.name_gen[gen_idx]
                    r_amount = self._redispatch[gen_idx]
                    redispatch["generators"].append(
                        {"gen_id": gen_idx, "gen_name": gen_name, "amount": r_amount}
                    )
            redispatch["changed"] = True
            has_impact = True

        storage = {"changed": False, "capacities": []}
        if self._modif_storage:
            for str_idx in range(self.n_storage):
                tmp = self._storage_power[str_idx]
                if np.isfinite(tmp):
                    name_ = self.name_storage[str_idx]
                    new_capacity = tmp
                    storage["capacities"].append(
                        {
                            "storage_id": str_idx,
                            "storage_name": name_,
                            "new_capacity": new_capacity,
                        }
                    )
            storage["changed"] = True
            has_impact = True

        curtailment = {"changed": False, "limit": []}
        if self._modif_curtailment:
            for gen_idx in range(self.n_gen):
                tmp = self._curtail[gen_idx]
                if np.isfinite(tmp) and np.abs(tmp + 1.) >= 1e-7:
                    name_ = self.name_gen[gen_idx]
                    new_max = tmp
                    curtailment["limit"].append(
                        {
                            "generator_id": gen_idx,
                            "generator_name": name_,
                            "amount": new_max,
                        }
                    )
            curtailment["changed"] = True
            has_impact = True

        return {
            "has_impact": has_impact,
            "injection": inject_detail,
            "force_line": force_line_status,
            "switch_line": switch_line_status,
            "topology": topology,
            "redispatch": redispatch,
            "storage": storage,
            "curtailment": curtailment,
        }

    def _aux_as_dict_set_line(self, res):
        res["set_line_status"] = {}
        res["set_line_status"]["nb_connected"] = (self._set_line_status == 1).sum()
        res["set_line_status"]["nb_disconnected"] = (
            self._set_line_status == -1
        ).sum()
        res["set_line_status"]["connected_id"] = (
            (self._set_line_status == 1).nonzero()[0]
        )
        res["set_line_status"]["disconnected_id"] = (
            (self._set_line_status == -1).nonzero()[0]
        )
        
    def _aux_as_dict_change_line(self, res):
        res["change_line_status"] = {}
        res["change_line_status"]["nb_changed"] = self._switch_line_status.sum()
        res["change_line_status"]["changed_id"] = (
            self._switch_line_status.nonzero()[0]
        )
    
    def _aux_as_dict_change_bus(self, res):
        res["change_bus_vect"] = {}
        res["change_bus_vect"]["nb_modif_objects"] = self._change_bus_vect.sum()
        all_subs = set()
        for id_, k in enumerate(self._change_bus_vect):
            if k:
                obj_id, objt_type, substation_id, nm_  = self._obj_caract_from_topo_id(
                    id_, with_name=True
                )
                sub_id = "{}".format(substation_id)
                if not sub_id in res["change_bus_vect"]:
                    res["change_bus_vect"][sub_id] = {}
                res["change_bus_vect"][sub_id][nm_] = {
                    "type": objt_type,
                    "id": obj_id,
                }
                all_subs.add(sub_id)

        res["change_bus_vect"]["nb_modif_subs"] = len(all_subs)
        res["change_bus_vect"]["modif_subs_id"] = sorted(all_subs)
    
    def _aux_as_dict_set_bus(self, res):
        res["set_bus_vect"] = {}
        res["set_bus_vect"]["nb_modif_objects"] = (self._set_topo_vect != 0).sum()
        all_subs = set()
        for id_, k in enumerate(self._set_topo_vect):
            if k != 0:
                obj_id, objt_type, substation_id, nm_ = self._obj_caract_from_topo_id(
                    id_, with_name=True
                )
                sub_id = "{}".format(substation_id)
                if not sub_id in res["set_bus_vect"]:
                    res["set_bus_vect"][sub_id] = {}
                res["set_bus_vect"][sub_id][nm_] = {
                    "type": objt_type,
                    "id": obj_id,
                    "new_bus": k,
                }
                all_subs.add(sub_id)

        res["set_bus_vect"]["nb_modif_subs"] = len(all_subs)
        res["set_bus_vect"]["modif_subs_id"] = sorted(all_subs)
    
    def _aux_as_dict_shunt(self, res): 
        tmp = {}
        if np.any(np.isfinite(self.shunt_p)):
            tmp["shunt_p"] = 1.0 * self.shunt_p
        if np.any(np.isfinite(self.shunt_q)):
            tmp["shunt_q"] = 1.0 * self.shunt_q
        if np.any(self.shunt_bus != 0):
            tmp["shunt_bus"] = 1.0 * self.shunt_bus
        if tmp:
            res["shunt"] = tmp
        
    def as_dict(self) -> Dict[Literal["load_p", "load_q", "prod_p", "prod_v",
                                      "change_line_status", "set_line_status",
                                      "change_bus_vect", "set_bus_vect",
                                      "redispatch", "storage_power", "curtailment"],
                              Any]:
        """
        Represent an action "as a" dictionary. This dictionary is useful to further inspect on which elements
        the actions had an impact. It is not recommended to use it as a way to serialize actions. The "do nothing"
        action should always be represented by an empty dictionary.

        The following keys (all optional) are present in the results:

          * `load_p`: if the action modifies the active loads.
          * `load_q`: if the action modifies the reactive loads.
          * `prod_p`: if the action modifies the active productions of generators.
          * `prod_v`: if the action modifies the voltage setpoint of generators.
          * `set_line_status` if the action tries to **set** the status of some powerlines. If present, this is a
            a dictionary with keys:

              * `nb_connected`: number of powerlines that are reconnected
              * `nb_disconnected`: number of powerlines that are disconnected
              * `connected_id`: the id of the powerlines reconnected
              * `disconnected_id`: the ids of the powerlines disconnected

          * `change_line_status`: if the action tries to **change** the status of some powerlines. If present, this
            is a dictionary with keys:

              * `nb_changed`: number of powerlines having their status changed
              * `changed_id`: the ids of the powerlines that are changed

          * `change_bus_vect`: if the action tries to **change** the topology of some substations. If present, this
            is a dictionary with keys:

              * `nb_modif_subs`: number of substations impacted by the action
              * `modif_subs_id`: ids of the substations impacted by the action
              * `change_bus_vect`: details the objects that are modified. It is itself a dictionary that represents for
                each impacted substations (keys) the modification of the objects connected to it.

          * `set_bus_vect`: if the action tries to **set** the topology of some substations. If present, this is a
            dictionary with keys:

              * `nb_modif_subs`: number of substations impacted by the action
              * `modif_subs_id`: the ids of the substations impacted by the action
              * `set_bus_vect`: details the objects that are modified. It is also a dictionary that represents for
                each impacted substations (keys) how the elements connected to it are impacted (their "new" bus)

          * `hazards` if the action is composed of some hazards. In this case, it's simply the index of the powerlines
            that are disconnected because of them.
          * `nb_hazards` the number of hazards the "action" implemented (eg number of powerlines disconnected because of
            hazards.
          * `maintenance` if the action is composed of some maintenance. In this case, it's simply the index of the
            powerlines that are affected by maintenance operation at this time step.
            that are disconnected because of them.
          * `nb_maintenance` the number of maintenance the "action" implemented eg the number of powerlines
            disconnected because of maintenance operations.
          * `redispatch` the redispatching action (if any). It gives, for each generator (all generator, not just the
            dispatchable one) the amount of power redispatched in this action.
          * `storage_power`: the setpoint for production / consumption for all storage units
          * `curtailment`: the curtailment performed on all generator
          * `shunt` :
          
        Returns
        -------
        res: ``dict``
            The action represented as a dictionary. See above for a description of it.

        """
        res = {}

        # saving the injections
        for k in ["load_p", "prod_p", "load_q", "prod_v"]:
            if k in self._dict_inj:
                res[k] = 1.0 * self._dict_inj[k]

        # handles actions on force line status
        if (self._set_line_status != 0).any():
            self._aux_as_dict_set_line(res)

        # handles action on swtich line status
        if self._switch_line_status.sum():
            self._aux_as_dict_change_line(res)

        # handles topology change
        if (self._change_bus_vect).any():
            self._aux_as_dict_change_bus(res)

        # handles topology set
        if (self._set_topo_vect!= 0).any():
            self._aux_as_dict_set_bus(res)

        if self._hazards.any():
            res["hazards"] = self._hazards.nonzero()[0]
            res["nb_hazards"] = self._hazards.sum()

        if self._maintenance.any():
            res["maintenance"] = self._maintenance.nonzero()[0]
            res["nb_maintenance"] = self._maintenance.sum()

        if (np.abs(self._redispatch) >= 1e-7).any():
            res["redispatch"] = 1.0 * self._redispatch

        if self._modif_storage:
            res["storage_power"] = 1.0 * self._storage_power

        if self._modif_curtailment:
            res["curtailment"] = 1.0 * self._curtail
            
        if type(self).shunts_data_available:
            self._aux_as_dict_shunt(res)
        return res

    def get_types(self) -> Tuple[bool, bool, bool, bool, bool, bool, bool]:
        """
        Shorthand to get the type of an action. The type of an action is among:

        - "injection": does this action modifies load or generator active values
        - "voltage": does this action modifies the generator voltage setpoint or the shunts
        - "topology": does this action modifies the topology of the grid (*ie* set or switch some buses)
        - "line": does this action modifies the line status
        - "redispatching" does this action modifies the redispatching
        - "storage" does this action impact the production / consumption of storage units
        - "curtailment" does this action impact the non renewable generators through curtailment

        Notes
        ------

        A single action can be of multiple types.

        The `do nothing` has no type at all (all flags are ``False``)

        If a line only set / change the status of a powerline then it does not count as a topological
        modification.

        If the bus to which a storage unit is connected is modified, but there is no setpoint for
        the production / consumption of any storage units, then the action is **NOT** taged as
        an action on the storage units.

        Returns
        -------
        injection: ``bool``
            Does it affect load or generator active value
        voltage: ``bool``
            Does it affect the voltage
        topology: ``bool``
            Does it affect the topology (line status change / switch are **NOT** counted as topology)
        line: ``bool``
            Does it affect the line status (line status change / switch are **NOT** counted as topology)
        redispatching: ``bool``
            Does it performs (explicitly) any redispatching
        storage: ``bool``
            Does it performs (explicitly) any action on the storage production / consumption
        curtailment: ``bool``
            Does it performs (explicitly) any action on renewable generator

        """
        injection = "load_p" in self._dict_inj or "prod_p" in self._dict_inj
        voltage = "prod_v" in self._dict_inj
        if type(self).shunts_data_available:
            voltage = voltage or np.isfinite(self.shunt_p).any()
            voltage = voltage or np.isfinite(self.shunt_q).any()
            voltage = voltage or (self.shunt_bus != 0).any()

        lines_impacted, subs_impacted = self.get_topological_impact()
        topology = subs_impacted.any()
        line = lines_impacted.any()
        redispatching = (np.abs(self._redispatch) >= 1e-7).any()
        storage = self._modif_storage
        curtailment = self._modif_curtailment
        return injection, voltage, topology, line, redispatching, storage, curtailment

    def _aux_effect_on_load(self, load_id):
        if load_id >= self.n_load:
            raise Grid2OpException(
                f"There are only {self.n_load} loads on the grid. Cannot check impact on "
                f"`load_id={load_id}`"
            )
        if load_id < 0:
            raise Grid2OpException(f"`load_id` should be positive.")
        res = {"new_p": np.NaN, "new_q": np.NaN, "change_bus": False, "set_bus": 0}
        if "load_p" in self._dict_inj:
            res["new_p"] = self._dict_inj["load_p"][load_id]
        if "load_q" in self._dict_inj:
            res["new_q"] = self._dict_inj["load_q"][load_id]
        my_id = self.load_pos_topo_vect[load_id]
        res["change_bus"] = self._change_bus_vect[my_id]
        res["set_bus"] = self._set_topo_vect[my_id]
        return res

    def _aux_effect_on_gen(self, gen_id):
        if gen_id >= self.n_gen:
            raise Grid2OpException(
                f"There are only {self.n_gen} gens on the grid. Cannot check impact on "
                f"`gen_id={gen_id}`"
            )
        if gen_id < 0:
            raise Grid2OpException(f"`gen_id` should be positive.")
        res = {"new_p": np.NaN, "new_v": np.NaN, "set_bus": 0, "change_bus": False}
        if "prod_p" in self._dict_inj:
            res["new_p"] = self._dict_inj["prod_p"][gen_id]
        if "prod_v" in self._dict_inj:
            res["new_v"] = self._dict_inj["prod_v"][gen_id]
        my_id = self.gen_pos_topo_vect[gen_id]
        res["change_bus"] = self._change_bus_vect[my_id]
        res["set_bus"] = self._set_topo_vect[my_id]
        res["redispatch"] = self._redispatch[gen_id]
        res["curtailment"] = self._curtail[gen_id]
        return res

    def _aux_effect_on_line(self, line_id):
        if line_id >= self.n_line:
            raise Grid2OpException(
                f"There are only {self.n_line} powerlines on the grid. Cannot check impact on "
                f"`line_id={line_id}`"
            )
        if line_id < 0:
            raise Grid2OpException(f"`line_id` should be positive.")
        res = {}
        # origin topology
        my_id = self.line_or_pos_topo_vect[line_id]
        res["change_bus_or"] = self._change_bus_vect[my_id]
        res["set_bus_or"] = self._set_topo_vect[my_id]
        # extremity topology
        my_id = self.line_ex_pos_topo_vect[line_id]
        res["change_bus_ex"] = self._change_bus_vect[my_id]
        res["set_bus_ex"] = self._set_topo_vect[my_id]
        # status
        res["set_line_status"] = self._set_line_status[line_id]
        res["change_line_status"] = self._switch_line_status[line_id]
        return res

    def _aux_effect_on_storage(self, storage_id):
        if storage_id >= self.n_storage:
            raise Grid2OpException(
                f"There are only {self.n_storage} storage units on the grid. "
                f"Cannot check impact on "
                f"`storage_id={storage_id}`"
            )
        if storage_id < 0:
            raise Grid2OpException(f"`storage_id` should be positive.")
        res = {"power": np.NaN, "set_bus": 0, "change_bus": False}
        my_id = self.storage_pos_topo_vect[storage_id]
        res["change_bus"] = self._change_bus_vect[my_id]
        res["set_bus"] = self._set_topo_vect[my_id]
        res["power"] = self._storage_power[storage_id]
        return res

    def _aux_effect_on_substation(self, substation_id):
        cls = type(self)
        if substation_id >= cls.n_sub:
            raise Grid2OpException(
                f"There are only {cls.n_sub} substations on the grid. "
                f"Cannot check impact on "
                f"`substation_id={substation_id}`"
            )
        if substation_id < 0:
            raise Grid2OpException(f"`substation_id` should be positive.")

        res = {}
        beg_ = int(cls.sub_info[:substation_id].sum())
        end_ = int(beg_ + cls.sub_info[substation_id])
        res["change_bus"] = self._change_bus_vect[beg_:end_]
        res["set_bus"] = self._set_topo_vect[beg_:end_]
        return res

    def effect_on(
        self,
        _sentinel=None,
        load_id=None,
        gen_id=None,
        line_id=None,
        substation_id=None,
        storage_id=None,
    ) -> dict:
        """
        Return the effect of this action on a unique given load, generator unit, powerline or substation.
        Only one of load, gen, line or substation should be filled.

        The query of these objects can only be done by id here (ie by giving the integer of the object in the backed).
        The :class:`ActionSpace` has some utilities to access them by name too.

        Parameters
        ----------
        _sentinel: ``None``
            Used to prevent positional parameters. Internal, **do not use**.

        load_id: ``int``
            The ID of the load we want to inspect

        gen_id: ``int``
            The ID of the generator we want to inspect

        line_id: ``int``
            The ID of the powerline we want to inspect

        substation_id: ``int``
            The ID of the substation we want to inspect

        storage_id: ``int``
            The ID of the storage unit we want to inspect

        Returns
        -------
        res: :class:`dict`
            A dictionary with keys and value depending on which object needs to be inspected:

            - if a load is inspected, then the keys are:

                - "new_p" the new load active value (or NaN if it doesn't change),
                - "new_q" the new load reactive value (or Nan if nothing has changed from this point of view)
                - "set_bus" the new bus where the load will be moved (int: id of the bus, 0 no change, -1 disconnected)
                - "change_bus" whether or not this load will be moved from one bus to another (for example is an action
                  asked it to go from bus 1 to bus 2)

            - if a generator is inspected, then the keys are:

                - "new_p" the new generator active setpoint value (or NaN if it doesn't change),
                - "new_v" the new generator voltage setpoint value (or Nan if nothing has changed from this point of
                  view)
                - "set_bus" the new bus where the load will be moved (int: id of the bus, 0 no change, -1 disconnected)
                - "change_bus" whether or not this load will be moved from one bus to another (for example is an action
                  asked it to go from bus 1 to bus 2)
                - "redispatch" the amount of power redispatched for this generator.
                - "curtailment": the amount of curtailment on this generator

            - if a powerline is inspected then the keys are:

                - "change_bus_or": whether or not the origin side will be moved from one bus to another
                - "change_bus_ex": whether or not the extremity side will be moved from one bus to another
                - "set_bus_or": the new bus where the origin will be moved
                - "set_bus_ex": the new bus where the extremity will be moved
                - "set_line_status": the new status of the power line
                - "change_line_status": whether or not to switch the status of the powerline

            - if a substation is inspected, it returns the topology to this substation in a dictionary with keys:

                - "change_bus"
                - "set_bus"

            - if a storage unit is inspected, it returns a dictionary with:

                - "change_bus"
                - "set_bus"
                - "power" : the power you want to produce  / absorb with the storage unit ( if < 0 the power is
                  produced, if > 0 then power is absorbed)

        NB the difference between "set_bus" and "change_bus" is the following:

            - If "set_bus" is 1, then the object (load, generator or powerline) will be moved to bus 1 of the substation
              to which it is connected. If it is already to bus 1 nothing will be done. If it's on another bus it will
              connect it to bus 1. It's disconnected, it will reconnect it and connect it to bus 1.
            - If "change_bus" is True, then the object will be moved from one bus to another. If the object were on
              bus 1
              then it will be moved on bus 2, and if it were on bus 2, it will be moved on bus 1. If the object were
              disconnected, then it will be connected to the affected bus.

        Raises
        ------
        :class:`grid2op.Exception.Grid2OpException`
            If _sentinel is modified, or if none of the arguments are set or alternatively if 2 or more of the
            parameters are being set.

        """
        EXCEPT_TOO_MUCH_ELEMENTS = (
            "You can only the inspect the effect of an action on one single element"
        )
        if _sentinel is not None:
            raise Grid2OpException(
                "action.effect_on should only be called with named argument."
            )

        if (
            load_id is None
            and gen_id is None
            and line_id is None
            and storage_id is None
            and substation_id is None
        ):
            raise Grid2OpException(
                "You ask the effect of an action on something, without provided anything"
            )

        if load_id is not None:
            if (
                gen_id is not None
                or line_id is not None
                or storage_id is not None
                or substation_id is not None
            ):
                raise Grid2OpException(EXCEPT_TOO_MUCH_ELEMENTS)
            res = self._aux_effect_on_load(load_id)

        elif gen_id is not None:
            if (
                line_id is not None
                or storage_id is not None
                or substation_id is not None
            ):
                raise Grid2OpException(EXCEPT_TOO_MUCH_ELEMENTS)
            res = self._aux_effect_on_gen(gen_id)

        elif line_id is not None:
            if storage_id is not None or substation_id is not None:
                raise Grid2OpException(EXCEPT_TOO_MUCH_ELEMENTS)
            res = self._aux_effect_on_line(line_id)

        elif storage_id is not None:
            if substation_id is not None:
                raise Grid2OpException(
                    "You can only the inspect the effect of an action on one single element"
                )
            res = self._aux_effect_on_storage(storage_id)

        else:
            res = self._aux_effect_on_substation(substation_id)
        return res

    def get_storage_modif(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve the modification that will be performed on all the storage unit

        Returns
        -------
        storage_power: ``np.ndarray``
            New storage power target (Nan = not modified, otherwise the setpoint given) [in MW]
        storage_set_bus: ``np.ndarray``
            New bus of the storage units, affected with "set_bus" command (0 = not affected, -1 = disconnected)
        storage_change_bus: ``np.ndarray``
            New bus of the storage units, affected with "change_bus" command

        """
        cls = type(self)
        storage_power = 1.0 * self._storage_power
        storage_set_bus = 1 * self._set_topo_vect[cls.storage_pos_topo_vect]
        storage_change_bus = copy.deepcopy(
            self._change_bus_vect[cls.storage_pos_topo_vect]
        )
        return storage_power, storage_set_bus, storage_change_bus

    def get_load_modif(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve the modification that will be performed on all the loads

        Returns
        -------
        load_p: ``np.ndarray``
            New load p (Nan = not modified) [in MW]
        load_q: ``np.ndarray``
            New load q (Nan = not modified) [in MVaR]
        load_set_bus: ``np.ndarray``
            New bus of the loads, affected with "set_bus" command
        load_change_bus: ``np.ndarray``
            New bus of the loads, affected with "change_bus" command
        """
        cls = type(self)
        load_p = np.full(cls.n_load, fill_value=np.NaN, dtype=dt_float)
        if "load_p" in self._dict_inj:
            load_p[:] = self._dict_inj["load_p"]
        load_q = 1.0 * load_p
        if "load_q" in self._dict_inj:
            load_q[:] = self._dict_inj["load_q"]
        load_set_bus = 1 * self._set_topo_vect[cls.load_pos_topo_vect]
        load_change_bus = copy.deepcopy(self._change_bus_vect[cls.load_pos_topo_vect])
        return load_p, load_q, load_set_bus, load_change_bus

    def get_gen_modif(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve the modification that will be performed on all the generators

        TODO add curtailment and redispatching

        Returns
        -------
        gen_p: ``np.ndarray``
            New gen p (Nan = not modified) [in MW]
        gen_v: ``np.ndarray``
            New gen v setpoint (Nan = not modified) [in kV]
        gen_set_bus: ``np.ndarray``
            New bus of the generators, affected with "set_bus" command
        gen_change_bus: ``np.ndarray``
            New bus of the generators, affected with "change_bus" command

        """
        cls = type(self)
        gen_p = np.full(cls.n_gen, fill_value=np.NaN, dtype=dt_float)
        if "prod_p" in self._dict_inj:
            gen_p[:] = self._dict_inj["prod_p"]
        gen_v = 1.0 * gen_p
        if "prod_v" in self._dict_inj:
            gen_v[:] = self._dict_inj["prod_v"]
        gen_set_bus = 1 * self._set_topo_vect[cls.gen_pos_topo_vect]
        gen_change_bus = copy.deepcopy(self._change_bus_vect[cls.gen_pos_topo_vect])
        return gen_p, gen_v, gen_set_bus, gen_change_bus

    # TODO do the get_line_modif, get_line_or_modif and get_line_ex_modif

    def _aux_affect_object_int(
        self,
        values,
        name_el,
        nb_els,
        name_els,
        inner_vect,
        outer_vect,
        min_val=-1,
        max_val=2,
    ):
        """
        NB : this do not set the _modif_set_bus attribute. It is expected to be set in the property setter.
        This is not set here, because it's recursive and if it fails at a point, it would be set for nothing

        values: the new values to set
        name_el: "load"
        nb_els: self.n_load
        inner_vect: self.load_pos_topo_vect
        name_els: self.name_load
        outer_vect: self._set_topo_vect

        will modify outer_vect[inner_vect]
        """
        if isinstance(values, tuple):
            # i provide a tuple: load_id, new_bus
            if len(values) != 2:
                raise IllegalAction(
                    f"when set with tuple, this tuple should have size 2 and be: {name_el}_id, new_bus "
                    f"eg. (3, {max_val})"
                )
            el_id, new_bus = values
            try:
                new_bus = int(new_bus)
            except Exception as exc_:
                raise IllegalAction(
                    f'new_bus should be convertible to integer. Error was : "{exc_}"'
                )

            if new_bus < min_val:
                raise IllegalAction(
                    f"new_bus should be between {min_val} and {max_val}"
                )
            if new_bus > max_val:
                raise IllegalAction(
                    f"new_bus should be between {min_val} and {max_val}"
                )

            if isinstance(el_id, (float, dt_float, np.float64)):
                raise IllegalAction(
                    f"{name_el}_id should be integers you provided float!"
                )
            if isinstance(el_id, (bool, dt_bool)):
                raise IllegalAction(
                    f"{name_el}_id should be integers you provided bool!"
                )
            if isinstance(el_id, str):
                raise IllegalAction(
                    f"{name_el}_id should be integers you provided string "
                    f"(hint: you can use a dictionary to set the bus by name eg. "
                    f"act.{name_el}_set_bus = {{act.name_{name_el}[0] : 1, act.name_{name_el}[1] : "
                    f"{max_val}}} )!"
                )

            try:
                el_id = int(el_id)
            except Exception as exc_:
                raise IllegalAction(
                    f'{name_el}_id should be convertible to integer. Error was : "{exc_}"'
                )
            if el_id < 0:
                raise IllegalAction(
                    f"Impossible to set the bus of a {name_el} with negative id"
                )
            if el_id >= nb_els:
                raise IllegalAction(
                    f"Impossible to set a {name_el} id {el_id} because there are only "
                    f"{nb_els} on the grid (and in python id starts at 0)"
                )
            outer_vect[inner_vect[el_id]] = new_bus
            return
        elif isinstance(values, np.ndarray):
            if (
                isinstance(values.dtype, float)
                or values.dtype == dt_float
                or values.dtype == np.float64
            ):
                raise IllegalAction(
                    f"{name_el}_id should be integers you provided float!"
                )
            if isinstance(values.dtype, bool) or values.dtype == dt_bool:
                raise IllegalAction(
                    f"{name_el}_id should be integers you provided boolean!"
                )

            try:
                values = values.astype(dt_int)
            except Exception as exc_:
                raise IllegalAction(
                    f'{name_el}_id should be convertible to integer. Error was : "{exc_}"'
                )
            if (values < min_val).any():
                raise IllegalAction(
                    f"new_bus should be between {min_val} and {max_val}, found a value < {min_val}"
                )
            if (values > max_val).any():
                raise IllegalAction(
                    f"new_bus should be between {min_val} and {max_val}, found a value  > {max_val}"
                )
            outer_vect[inner_vect] = values
            return
        elif isinstance(values, list):
            # 2 cases: list of tuple, or list (convertible to numpy array)
            if len(values) == nb_els:
                # 2 cases: either i set all loads in the form [(0,..), (1,..), (2,...)]
                # or i should have converted the list to np array
                if isinstance(values[0], tuple):
                    # list of tuple, handled below
                    # TODO can be somewhat "hacked" if the type of the object on the list is not always the same
                    pass
                else:
                    # get back to case where it's a full vector
                    values = np.array(values)
                    self._aux_affect_object_int(
                        values,
                        name_el,
                        nb_els,
                        name_els,
                        inner_vect=inner_vect,
                        outer_vect=outer_vect,
                        min_val=min_val,
                        max_val=max_val,
                    )
                    return

            # expected list of tuple, each tuple is a pair with load_id, new_load_bus: example: [(0, 1), (2,2)]
            for el in values:
                if len(el) != 2:
                    raise IllegalAction(
                        f"If input is a list, it should be a  list of pair (el_id, new_bus) "
                        f"eg. [(0, {max_val}), (2, {min_val})]"
                    )
                el_id, new_bus = el
                if isinstance(el_id, str) and name_els is not None:
                    tmp = (name_els == el_id).nonzero()[0]
                    if len(tmp) == 0:
                        raise IllegalAction(f"No known {name_el} with name {el_id}")
                    el_id = tmp[0]
                self._aux_affect_object_int(
                    (el_id, new_bus),
                    name_el,
                    nb_els,
                    name_els,
                    inner_vect=inner_vect,
                    outer_vect=outer_vect,
                    min_val=min_val,
                    max_val=max_val,
                )
        elif isinstance(values, dict):
            # 2 cases: either key = load_id and value = new_bus or key = load_name and value = new bus
            for key, new_bus in values.items():
                if isinstance(key, str) and name_els is not None:
                    tmp = (name_els == key).nonzero()[0]
                    if len(tmp) == 0:
                        raise IllegalAction(f"No known {name_el} with name {key}")
                    key = tmp[0]
                self._aux_affect_object_int(
                    (key, new_bus),
                    name_el,
                    nb_els,
                    name_els,
                    inner_vect=inner_vect,
                    outer_vect=outer_vect,
                    min_val=min_val,
                    max_val=max_val,
                )
        else:
            raise IllegalAction(
                f"Impossible to modify the {name_el} bus with inputs {values}. "
                f"Please see the documentation."
            )

    @property
    def load_set_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which the action **set** the loads.

        .. versionchanged:: 1.10.0
            From grid2op version 1.10.0 it is possible (under some cirumstances, depending on how
            the environment is created) to set the busbar to a number >= 3, depending on the value
            of `type(act).n_busbar_per_sub`.
            
        Returns
        -------
        res:
            A vector of integer, of size `act.n_gen` indicating what type of action is performed for
            each load units with the convention :

            * 0 the action do not action on this load
            * -1 the action disconnect the load
            * 1 the action set the load to busbar 1
            * 2 the action set the load to busbar 2
            * 3 the action set the load to busbar 3 (grid2op >= 1.10.0)
            * etc. (grid2op >= 1.10.0)

        Examples
        --------

        Please refer to the documentation of :attr:`BaseAction.gen_set_bus` for more information.

        .. note:: 
            Be careful not to mix "change" and "set". For "change" you only need to provide the ID of the elements
            you want to change, for "set" you need to provide the ID **AND** where you want to set them.

        """
        res = self.set_bus[self.load_pos_topo_vect]
        res.flags.writeable = False
        return res

    @load_set_bus.setter
    def load_set_bus(self, values):
        cls = type(self)
        if "set_bus" not in cls.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the load bus (with "set") with this action type.'
            )
        orig_ = self.load_set_bus
        try:
            self._aux_affect_object_int(
                values,
                "load",
                cls.n_load,
                cls.name_load,
                cls.load_pos_topo_vect,
                self._set_topo_vect,
                max_val=cls.n_busbar_per_sub
            )
            self._modif_set_bus = True
        except Exception as exc_:
            self._aux_affect_object_int(
                orig_,
                "load",
                cls.n_load,
                cls.name_load,
                cls.load_pos_topo_vect,
                self._set_topo_vect,
                max_val=cls.n_busbar_per_sub
            )
            raise IllegalAction(
                f"Impossible to modify the load bus with your input. Please consult the documentation. "
                f'The error was "{exc_}"'
            )

    @property
    def gen_set_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which the action **set** the generator units.

        .. versionchanged:: 1.10.0
            From grid2op version 1.10.0 it is possible (under some cirumstances, depending on how
            the environment is created) to set the busbar to a number >= 3, depending on the value
            of `type(act).n_busbar_per_sub`.
            
        Returns
        -------
        res:
            A vector of integer, of size `act.n_gen` indicating what type of action is performed for
            each generator units with the convention :

            * 0 the action do not action on this generator
            * -1 the action disconnect the generator
            * 1 the action set the generator to busbar 1
            * 2 the action set the generator to busbar 2
            * 3 the action set the generator to busbar 3 (grid2op >= 1.10.0)
            * etc. (grid2op >= 1.10.0)

        Examples
        --------

        To retrieve the impact of the action on the generator, you can do:

        .. code-block:: python

            gen_buses = act.gen_set_bus

        To modify these buses with **set** you can do:

        .. code-block:: python

            # create an environment where i can modify everything
            import numpy as np
            import grid2op
            from grid2op.Action import CompleteAction
            env = grid2op.make("educ_case14_storage", test=True, action_class=CompleteAction)

            # create an action
            act = env.action_space()

            # method 1 : provide the full vector
            act.gen_set_bus = np.ones(act.n_gen, dtype=int)

            # method 2: provide the index of the unit you want to modify
            act.gen_set_bus = (1, 2)

            # method 3: provide a list of the units you want to modify
            act.gen_set_bus = [(1, 2), (0, -1)]

            # method 4: change the storage unit by their name with a dictionary
            act.gen_set_bus = {"gen_1_0": 2}

        .. note:: The "rule of thumb" to modify an object using "set" method it to provide always
            the ID of an object AND its value. The ID should be an integer (or a name in some cases)
            and the value an integer representing on which busbar to put the new element.

        Notes
        -----
        It is a "property", you don't have to use parenthesis to access it:

        .. code-block:: python

            # valid code
            gen_buses = act.gen_set_bus

            # invalid code, it will crash, do not run
            gen_buses = act.gen_set_bus()
            # end do not run

        And neither should you uses parenthesis to modify it:

        .. code-block:: python

            # valid code
            act.gen_set_bus = [(1, 2), (0, -1)]

            # invalid code, it will crash, do not run
            act.gen_set_bus([(1, 2), (0, -1)])
            # end do not run

        Property cannot be set "directly", you have to use the `act.XXX = ...` syntax. For example:

        .. code-block:: python

            # valid code
            act.gen_set_bus = [(1, 2), (0, -1)]

            # invalid code, it will raise an error, and even if it did not it would have not effect
            # do not run
            act.gen_set_bus[1] = 2
            # end do not run

        .. note:: 
            Be careful not to mix "change" and "set". For "change" you only need to provide the ID of the elements
            you want to change, for "set" you need to provide the ID **AND** where you want to set them.

        """
        res = self.set_bus[self.gen_pos_topo_vect]
        res.flags.writeable = False
        return res

    @gen_set_bus.setter
    def gen_set_bus(self, values):
        cls = type(self)
        if "set_bus" not in cls.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the gen bus (with "set") with this action type.'
            )
        orig_ = self.gen_set_bus
        try:
            self._aux_affect_object_int(
                values,
                "gen",
                cls.n_gen,
                cls.name_gen,
                cls.gen_pos_topo_vect,
                self._set_topo_vect,
                max_val=cls.n_busbar_per_sub
            )
            self._modif_set_bus = True
        except Exception as exc_:
            self._aux_affect_object_int(
                orig_,
                "gen",
                cls.n_gen,
                cls.name_gen,
                cls.gen_pos_topo_vect,
                self._set_topo_vect,
                max_val=cls.n_busbar_per_sub
            )
            raise IllegalAction(
                f"Impossible to modify the gen bus with your input. Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def storage_set_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which the action **set** the storage units.

        .. versionchanged:: 1.10.0
            From grid2op version 1.10.0 it is possible (under some cirumstances, depending on how
            the environment is created) to set the busbar to a number >= 3, depending on the value
            of `type(act).n_busbar_per_sub`.
            
        Returns
        -------
        res:
            A vector of integer, of size `act.n_gen` indicating what type of action is performed for
            each storage unit with the convention :

            * 0 the action do not action on this storage unit
            * -1 the action disconnect the storage unit
            * 1 the action set the storage unit to busbar 1
            * 2 the action set the storage unit to busbar 2
            * 3 the action set the storage unit to busbar 3 (grid2op >= 1.10.0)
            * etc. (grid2op >= 1.10.0)

        Examples
        --------

        Please refer to the documentation of :attr:`BaseAction.gen_set_bus` for more information.

        .. note:: 
            Be careful not to mix "change" and "set". For "change" you only need to provide the ID of the elements
            you want to change, for "set" you need to provide the ID **AND** where you want to set them.

        """
        if "set_storage" not in self.authorized_keys:
            raise IllegalAction(type(self).ERR_NO_STOR_SET_BUS)
        res = self.set_bus[self.storage_pos_topo_vect]
        res.flags.writeable = False
        return res

    @storage_set_bus.setter
    def storage_set_bus(self, values):
        cls = type(self)
        if "set_bus" not in cls.authorized_keys:
            raise IllegalAction(cls.ERR_NO_STOR_SET_BUS)
        if "set_storage" not in cls.authorized_keys:
            raise IllegalAction(cls.ERR_NO_STOR_SET_BUS)
        orig_ = self.storage_set_bus
        try:
            self._aux_affect_object_int(
                values,
                "storage",
                cls.n_storage,
                cls.name_storage,
                cls.storage_pos_topo_vect,
                self._set_topo_vect,
                max_val=cls.n_busbar_per_sub
            )
            self._modif_set_bus = True
        except Exception as exc_:
            self._aux_affect_object_int(
                orig_,
                "storage",
                cls.n_storage,
                cls.name_storage,
                cls.storage_pos_topo_vect,
                self._set_topo_vect,
                max_val=cls.n_busbar_per_sub
            )
            raise IllegalAction(
                f"Impossible to modify the storage bus with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def line_or_set_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which the action **set** the lines (origin side).

        .. versionchanged:: 1.10.0
            From grid2op version 1.10.0 it is possible (under some cirumstances, depending on how
            the environment is created) to set the busbar to a number >= 3, depending on the value
            of `type(act).n_busbar_per_sub`.
            
        Returns
        -------
        res:
            A vector of integer, of size `act.n_gen` indicating what type of action is performed for
            each lines (origin side) with the convention :

            * 0 the action do not action on this line (origin side)
            * -1 the action disconnect the line (origin side)
            * 1 the action set the line (origin side) to busbar 1
            * 2 the action set the line (origin side) to busbar 2
            * 3 the action set the line (origin side) to busbar 3 (grid2op >= 1.10.0)
            * etc.

        Examples
        --------

        Please refer to the documentation of :attr:`BaseAction.gen_set_bus` for more information.

        .. note:: 
            Be careful not to mix "change" and "set". For "change" you only need to provide the ID of the elements
            you want to change, for "set" you need to provide the ID **AND** where you want to set them.

        """
        res = self.set_bus[self.line_or_pos_topo_vect]
        res.flags.writeable = False
        return res

    @line_or_set_bus.setter
    def line_or_set_bus(self, values):
        cls = type(self)
        if "set_bus" not in cls.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the line (origin) bus (with "set") with this action type.'
            )
        orig_ = self.line_or_set_bus
        try:
            self._aux_affect_object_int(
                values,
                self._line_or_str,
                self.n_line,
                self.name_line,
                self.line_or_pos_topo_vect,
                self._set_topo_vect,
                max_val=cls.n_busbar_per_sub
            )
            self._modif_set_bus = True
        except Exception as exc_:
            self._aux_affect_object_int(
                orig_,
                self._line_or_str,
                cls.n_line,
                cls.name_line,
                cls.line_or_pos_topo_vect,
                self._set_topo_vect,
                max_val=cls.n_busbar_per_sub
            )
            raise IllegalAction(
                f"Impossible to modify the line origin bus with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def line_ex_set_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which the extremity side of each powerline is **set**.

        It behaves similarly as :attr:`BaseAction.gen_set_bus`. See the help there for more information.
        """
        res = self.set_bus[self.line_ex_pos_topo_vect]
        res.flags.writeable = False
        return res

    @line_ex_set_bus.setter
    def line_ex_set_bus(self, values):
        cls = type(self)
        if "set_bus" not in cls.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the line (ex) bus (with "set") with this action type.'
            )
        orig_ = self.line_ex_set_bus
        try:
            self._aux_affect_object_int(
                values,
                self._line_ex_str,
                cls.n_line,
                cls.name_line,
                cls.line_ex_pos_topo_vect,
                self._set_topo_vect,
                max_val=cls.n_busbar_per_sub
            )
            self._modif_set_bus = True
        except Exception as exc_:
            self._aux_affect_object_int(
                orig_,
                self._line_ex_str,
                cls.n_line,
                cls.name_line,
                cls.line_ex_pos_topo_vect,
                self._set_topo_vect,
                max_val=cls.n_busbar_per_sub
            )
            raise IllegalAction(
                f"Impossible to modify the line extrmity bus with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def set_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which any element is **set**.

        It behaves similarly as :attr:`BaseAction.gen_set_bus` and can be use to modify any elements type
        as opposed to the more specific :attr:`BaseAction.gen_set_bus`, :attr:`BaseAction.load_set_bus`,
        :attr:`BaseAction.line_or_set_bus`, :attr:`BaseAction.line_ex_set_bus` or
        :attr:`BaseAction.storage_set_bus` that are specific to a certain type of objects.

        Notes
        -----

        For performance reasons, it do not allow to modify the elements by there names.

        The order of each elements are given in the :attr:`grid2op.Space.GridObjects.gen_pos_topo_vect`,
        :attr:`grid2op.Space.GridObjects.load_pos_topo_vect`,
        :attr:`grid2op.Space.GridObjects.line_or_pos_topo_vect`,
        :attr:`grid2op.Space.GridObjects.line_ex_pos_topo_vect` or
        :attr:`grid2op.Space.GridObjects.storage_pos_topo_vect`

        For example:

        .. code-block:: python

            act.set_bus = [(0,1), (1, -1), (3, 2)]

        Will:

          * set to bus 1 the (unique) element for which \*_pos_topo_vect is 1
          * disconnect the (unique) element for which \*_pos_topo_vect is 2
          * set to bus 2 the (unique) element for which \*_pos_topo_vect is 3

        You can use the documentation page :ref:`modeled-elements-module` for more information about which
        element correspond to what component of this vector.

        """
        if "set_bus" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the bus (with "set") with this action type.'
            )
        res = 1 * self._set_topo_vect
        res.flags.writeable = False
        return res

    @set_bus.setter
    def set_bus(self, values):
        cls = type(self)
        if "set_bus" not in cls.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the bus (with "set") with this action type.'
            )
        orig_ = self.set_bus
        try:
            self._aux_affect_object_int(
                values,
                "",
                cls.dim_topo,
                None,
                np.arange(cls.dim_topo),
                self._set_topo_vect,
                max_val=cls.n_busbar_per_sub
            )
            self._modif_set_bus = True
        except Exception as exc_:
            self._aux_affect_object_int(
                orig_,
                "",
                cls.dim_topo,
                None,
                np.arange(cls.dim_topo),
                self._set_topo_vect,
                max_val=cls.n_busbar_per_sub
            )
            raise IllegalAction(
                f"Impossible to modify the bus with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def line_set_status(self) -> np.ndarray:
        """
        Property to set the status of the powerline.

        It behave similarly than :attr:`BaseAction.gen_set_bus` but with the following convention:

        * 0 still means it is not affected
        * +1 means that we force the connection on a powerline
        * -1 means we force the disconnection of a powerline

        Notes
        -----

        Setting a status of a powerline to +2 will raise an error.

        Examples
        ---------

        For example:

        .. code-block:: python

            act.line_set_status = [(0,1), (1, -1), (3, 1)]

        Will force the reconnection of line id 0 and 1 and force disconnection of line id 1.

        """
        if "set_line_status" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the status of powerlines (with "set") with this action type.'
            )
        res = 1 * self._set_line_status
        res.flags.writeable = False
        return res

    @line_set_status.setter
    def line_set_status(self, values):
        if "set_line_status" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the status of powerlines (with "set") with this action type.'
            )
        orig_ = 1 * self._set_line_status
        try:
            self._aux_affect_object_int(
                values,
                "line status",
                self.n_line,
                self.name_line,
                np.arange(self.n_line),
                self._set_line_status,
                max_val=1,
            )
            self._modif_set_status = True
        except Exception as exc_:
            self._aux_affect_object_int(
                orig_,
                "line status",
                self.n_line,
                self.name_line,
                np.arange(self.n_line),
                self._set_line_status,
                max_val=1,
            )
            raise IllegalAction(
                f"Impossible to modify the line status with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def set_line_status(self) -> np.ndarray:
        """another name for :func:`BaseAction.line_set_status`"""
        return self.line_set_status

    @set_line_status.setter
    def set_line_status(self, values):
        self.line_set_status = values

    @property
    def change_line_status(self) -> np.ndarray:
        """another name for :func:`BaseAction.change_line_status`"""
        return self.line_change_status

    @change_line_status.setter
    def change_line_status(self, values):
        self.line_change_status = values

    def _aux_affect_object_bool(
        self, values, name_el, nb_els, name_els, inner_vect, outer_vect
    ):
        """
        NB : this do not set the _modif_set_bus attribute. It is expected to be set in the property setter.
        This is not set here, because it's recursive and if it fails at a point, it would be set for nothing

        values: the new values to set
        name_el: "load"
        nb_els: self.n_load
        inner_vect: self.load_pos_topo_vect
        name_els: self.name_load
        outer_vect: self._change_bus_vect

        will modify outer_vect[inner_vect]
        """
        if isinstance(values, bool):
            # to make it explicit, tuple modifications are deactivated
            raise IllegalAction(
                f"Impossible to change a {name_el} with a tuple input. Accepted inputs are:"
                f"int, list of int, list of string, array of int, array of bool, set of int,"
                f"set of string"
            )
        elif isinstance(values, float):
            # to make it explicit, tuple modifications are deactivated
            raise IllegalAction(
                f"Impossible to change a {name_el} with a tuple input. Accepted inputs are:"
                f"int, list of int, list of string, array of int, array of bool, set of int,"
                f"set of string"
            )
        elif isinstance(values, (int, dt_int, np.int64)):
            # i provide an int: load_id
            try:
                el_id = int(values)
            except Exception as exc_:
                raise IllegalAction(
                    f'{name_el}_id should be convertible to integer. Error was : "{exc_}"'
                )
            if el_id < 0:
                raise IllegalAction(
                    f"Impossible to change a negative {name_el} with negative id"
                )
            if el_id >= nb_els:
                raise IllegalAction(
                    f"Impossible to change a {name_el} id {el_id} because there are only "
                    f"{nb_els} on the grid (and in python id starts at 0)"
                )
            outer_vect[inner_vect[el_id]] = not outer_vect[inner_vect[el_id]]
            return
        elif isinstance(values, tuple):
            # to make it explicit, tuple modifications are deactivated
            raise IllegalAction(
                f"Impossible to change a {name_el} with a tuple input. Accepted inputs are:"
                f"int, list of int, list of string, array of int, array of bool, set of int,"
                f"set of string"
            )
        elif isinstance(values, np.ndarray):
            # either the int id i need to change or the full value.
            if (
                isinstance(values.dtype, bool)
                or values.dtype == dt_bool
                or values.dtype == bool
            ):
                # so i change by giving the full vector
                if values.shape[0] != nb_els:
                    raise IllegalAction(
                        f"If provided with bool array, the number of components of the vector"
                        f"should match the total number of {name_el}. You provided a vector "
                        f"with size {values.shape[0]} and there are {nb_els} {name_el} "
                        f"on the grid."
                    )
                outer_vect[inner_vect[values]] = ~outer_vect[inner_vect[values]]
                return

            # this is the case where i give the integers i want to change
            try:
                values = values.astype(dt_int)
            except Exception as exc_:
                raise IllegalAction(
                    f'{name_el}_id should be convertible to integer. Error was : "{exc_}"'
                )
            if (values < 0).any():
                raise IllegalAction(
                    f"Impossible to change a negative {name_el} with negative id"
                )
            if (values > nb_els).any():
                raise IllegalAction(
                    f"Impossible to change a {name_el} id because there are only "
                    f"{nb_els} on the grid and you wanted to change an element with an "
                    f"id > {nb_els} (in python id starts at 0)"
                )
            outer_vect[inner_vect[values]] = ~outer_vect[inner_vect[values]]
            return
        elif isinstance(values, list):
            # 1 case only: list of int
            # (note: i cannot convert to numpy array other I could mix types...)
            for el_id_or_name in values:
                if isinstance(el_id_or_name, str):
                    tmp = (name_els == el_id_or_name).nonzero()[0]
                    if len(tmp) == 0:
                        raise IllegalAction(
                            f'No known {name_el} with name "{el_id_or_name}"'
                        )
                    el_id = tmp[0]
                elif isinstance(el_id_or_name, (bool, dt_bool)):
                    # somehow python considers bool are int...
                    raise IllegalAction(
                        f"If a list is provided, it is only valid with integer found "
                        f"{type(el_id_or_name)}."
                    )
                elif isinstance(el_id_or_name, (int, dt_int, np.int64)):
                    el_id = el_id_or_name
                else:
                    raise IllegalAction(
                        f"If a list is provided, it is only valid with integer found "
                        f"{type(el_id_or_name)}."
                    )
                el_id = int(el_id)
                self._aux_affect_object_bool(
                    el_id,
                    name_el,
                    nb_els,
                    name_els,
                    inner_vect=inner_vect,
                    outer_vect=outer_vect,
                )
        elif isinstance(values, set):
            # 2 cases: either set of load_id or set of load_name
            values = list(values)
            self._aux_affect_object_bool(
                values,
                name_el,
                nb_els,
                name_els,
                inner_vect=inner_vect,
                outer_vect=outer_vect,
            )
        else:
            raise IllegalAction(
                f"Impossible to modify the {name_el} with inputs {values}. "
                f"Please see the documentation."
            )

    @property
    def change_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which any element is **change**.

        It behaves similarly as :attr:`BaseAction.gen_change_bus` and can be use to modify any elements type
        as opposed to the more specific :attr:`BaseAction.gen_change_bus`, :attr:`BaseAction.load_change_bus`,
        :attr:`BaseAction.line_or_change_bus`, :attr:`BaseAction.line_ex_change_bus` or
        :attr:`BaseAction.storage_change_bus` that are specific to a certain type of objects.

        Notes
        -----

        For performance reasons, it do not allow to modify the elements by there names.

        The order of each elements are given in the :attr:`grid2op.Space.GridObjects.gen_pos_topo_vect`,
        :attr:`grid2op.Space.GridObjects.load_pos_topo_vect`,
        :attr:`grid2op.Space.GridObjects.line_or_pos_topo_vect`,
        :attr:`grid2op.Space.GridObjects.line_ex_pos_topo_vect` or
        :attr:`grid2op.Space.GridObjects.storage_pos_topo_vect`

        For example:

        .. code-block:: python

            act.set_bus [0, 1, 3]

        Will:

          * change the bus of the (unique) element for which \*_pos_topo_vect is 1
          * change the bus of (unique) element for which \*_pos_topo_vect is 2
          * change the bus of (unique) element for which \*_pos_topo_vect is 3

        You can use the documentation page :ref:`modeled-elements-module` for more information about which
        element correspond to what component of this "vector".

        """
        res = copy.deepcopy(self._change_bus_vect)
        res.flags.writeable = False
        return res

    @change_bus.setter
    def change_bus(self, values):
        orig_ = self.change_bus
        try:
            self._aux_affect_object_bool(
                values,
                "",
                self.dim_topo,
                None,
                np.arange(self.dim_topo),
                self._change_bus_vect,
            )
            self._modif_change_bus = True
        except Exception as exc_:
            self._aux_affect_object_bool(
                orig_,
                "",
                self.dim_topo,
                None,
                np.arange(self.dim_topo),
                self._change_bus_vect,
            )
            raise IllegalAction(
                f"Impossible to modify the bus with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def load_change_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which the loads is **changed**.

        It behaves similarly as :attr:`BaseAction.gen_change_bus`. See the help there for more information.
        """
        res = self.change_bus[self.load_pos_topo_vect]
        res.flags.writeable = False
        return res

    @load_change_bus.setter
    def load_change_bus(self, values):
        if "change_bus" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the load bus (with "change") with this action type.'
            )
        orig_ = self.load_change_bus
        try:
            self._aux_affect_object_bool(
                values,
                "load",
                self.n_load,
                self.name_load,
                self.load_pos_topo_vect,
                self._change_bus_vect,
            )
            self._modif_change_bus = True
        except Exception as exc_:
            self._change_bus_vect[self.load_pos_topo_vect] = orig_
            raise IllegalAction(
                f"Impossible to modify the load bus with your input. Please consult the documentation. "
                f'The error was "{exc_}"'
            )

    @property
    def gen_change_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which the action **change** the generator units.

        Returns
        -------
        res:
            A vector of bool, of size `act.n_gen` indicating what type of action is performed for
            each generator units with the convention :

            * ``False`` this generator is not affected by any "change" action
            * ``True`` this generator bus is not affected by any "change" action. If it was
              on bus 1, it will be moved to bus 2, if it was on bus 2 it will be moved to bus 1 (
              and if it was disconnected it will stay disconnected)

        Examples
        --------

        To retrieve the impact of the action on the storage unit, you can do:

        .. code-block:: python

            gen_buses = act.gen_change_bus

        To modify these buses you can do:

        .. code-block:: python

            # create an environment where i can modify everything
            import numpy as np
            import grid2op
            from grid2op.Action import CompleteAction
            env = grid2op.make("educ_case14_storage", test=True, action_class=CompleteAction)

            # create an action
            act = env.action_space()

            # method 1 : provide the full vector
            act.gen_change_bus = np.ones(act.n_gen, dtype=bool)

            # method 2: provide the index of the unit you want to modify
            act.gen_change_bus = 1

            # method 3: provide a list of the units you want to modify
            act.gen_change_bus = [1, 2]

            # method 4: change the storage unit by their name with a set
            act.gen_change_bus = {"gen_1_0"}

        .. note:: The "rule of thumb" to modify an object using "change" method it to provide always
            the ID of an object. The ID should be an integer (or a name in some cases). It does not
            make any sense to provide a "value" associated to an ID: either you change it, or not.

        Notes
        -----
        It is a "property", you don't have to use parenthesis to access it:

        .. code-block:: python

            # valid code
            gen_buses = act.gen_change_bus

            # invalid code, it will crash, do not run
            gen_buses = act.gen_change_bus()
            # end do not run

        And neither should you uses parenthesis to modify it:

        .. code-block:: python

            # valid code
            act.gen_change_bus = [1, 2, 3]

            # invalid code, it will crash, do not run
            act.gen_change_bus([1, 2, 3])
            # end do not run

        Property cannot be set "directly", you have to use the `act.XXX = ..` syntax. For example:

        .. code-block:: python

            # valid code
            act.gen_change_bus = [1, 3, 4]

            # invalid code, it will raise an error, and even if it did not it would have not effect
            # do not run
            act.gen_change_bus[1] = True
            # end do not run

        .. note:: Be careful not to mix "change" and "set". For "change" you only need to provide the ID of the elements
            you want to change, for "set" you need to provide the ID **AND** where you want to set them.

        """
        res = self.change_bus[self.gen_pos_topo_vect]
        res.flags.writeable = False
        return res

    @gen_change_bus.setter
    def gen_change_bus(self, values):
        if "change_bus" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the gen bus (with "change") with this action type.'
            )
        orig_ = self.gen_change_bus
        try:
            self._aux_affect_object_bool(
                values,
                "gen",
                self.n_gen,
                self.name_gen,
                self.gen_pos_topo_vect,
                self._change_bus_vect,
            )
            self._modif_change_bus = True
        except Exception as exc_:
            self._change_bus_vect[self.gen_pos_topo_vect] = orig_
            raise IllegalAction(
                f"Impossible to modify the gen bus with your input. Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def storage_change_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which the storage units are **changed**.

        It behaves similarly as :attr:`BaseAction.gen_change_bus`. See the help there for more information.
        """
        res = self.change_bus[self.storage_pos_topo_vect]
        res.flags.writeable = False
        return res

    @storage_change_bus.setter
    def storage_change_bus(self, values):
        if "change_bus" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the storage bus (with "change") with this action type.'
            )
        if "set_storage" not in self.authorized_keys:
            raise IllegalAction(
                "Impossible to modify the storage units with this action type."
            )
        orig_ = self.storage_change_bus
        try:
            self._aux_affect_object_bool(
                values,
                "storage",
                self.n_storage,
                self.name_storage,
                self.storage_pos_topo_vect,
                self._change_bus_vect,
            )
            self._modif_change_bus = True
        except Exception as exc_:
            self._change_bus_vect[self.storage_pos_topo_vect] = orig_
            raise IllegalAction(
                f"Impossible to modify the storage bus with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def line_or_change_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which the origin side of powerlines are **changed**.

        It behaves similarly as :attr:`BaseAction.gen_change_bus`. See the help there for more information.
        """
        res = self.change_bus[self.line_or_pos_topo_vect]
        res.flags.writeable = False
        return res

    @line_or_change_bus.setter
    def line_or_change_bus(self, values):
        if "change_bus" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the line (origin) bus (with "change") with this action type.'
            )
        orig_ = self.line_or_change_bus
        try:
            self._aux_affect_object_bool(
                values,
                self._line_or_str,
                self.n_line,
                self.name_line,
                self.line_or_pos_topo_vect,
                self._change_bus_vect,
            )
            self._modif_change_bus = True
        except Exception as exc_:
            self._change_bus_vect[self.line_or_pos_topo_vect] = orig_
            raise IllegalAction(
                f"Impossible to modify the line origin bus with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def line_ex_change_bus(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the busbars at which the extremity side of powerlines are **changed**.

        It behaves similarly as :attr:`BaseAction.gen_change_bus`. See the help there for more information.
        """
        res = self.change_bus[self.line_ex_pos_topo_vect]
        res.flags.writeable = False
        return res

    @line_ex_change_bus.setter
    def line_ex_change_bus(self, values):
        if "change_bus" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the line (ex) bus (with "change") with this action type.'
            )
        orig_ = self.line_ex_change_bus
        try:
            self._aux_affect_object_bool(
                values,
                self._line_ex_str,
                self.n_line,
                self.name_line,
                self.line_ex_pos_topo_vect,
                self._change_bus_vect,
            )
            self._modif_change_bus = True
        except Exception as exc_:
            self._change_bus_vect[self.line_ex_pos_topo_vect] = orig_
            raise IllegalAction(
                f"Impossible to modify the line extrmity bus with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def line_change_status(self) -> np.ndarray:
        """
        Property to set the status of the powerline.

        It behave similarly than :attr:`BaseAction.gen_change_bus` but with the following convention:

        * ``False`` will not affect the powerline
        * ``True`` will change the status of the powerline. If it was connected, it will attempt to
          disconnect it, if it was disconnected, it will attempt to reconnect it.

        """
        res = copy.deepcopy(self._switch_line_status)
        res.flags.writeable = False
        return res

    @line_change_status.setter
    def line_change_status(self, values):
        if "change_line_status" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the status of powerlines (with "change") with this action type.'
            )
        orig_ = 1 * self._switch_line_status
        try:
            self._aux_affect_object_bool(
                values,
                "line status",
                self.n_line,
                self.name_line,
                np.arange(self.n_line),
                self._switch_line_status,
            )
            self._modif_change_status = True
        except Exception as exc_:
            self._switch_line_status[:] = orig_
            raise IllegalAction(
                f"Impossible to modify the line status with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def raise_alarm(self) -> np.ndarray:
        """
        .. warning::
            /!\\\\ Only valid with "l2rpn_icaps_2021" environment /!\\\\
                
        Property to raise alarm.

        If you set it to ``True`` an alarm is raised for the given area, otherwise None are raised.

        Notes
        -----
        In order to be able to "cancel" an alarm properly, if you set "two consecutive alarm" on the same area
        it will behave as if you had set none:

        .. code-block:: python

            import grid2op
            env_name = "l2rpn_icaps_2021"  # chose an environment that supports the alarm feature
            env = grid2op.make(env_name)
            act = env.action_space()

            act.raise_alarm = [0]
            # this act will raise an alarm on the area 0

            act.raise_alarm = [0]
            # this second call will "cancel" the alarm for convenience

        This might be counter intuitive

        """
        res = copy.deepcopy(self._raise_alarm)
        res.flags.writeable = False
        return res

    @raise_alarm.setter
    def raise_alarm(self, values):
        """
        .. warning::
            /!\\\\ Only valid with "l2rpn_icaps_2021" environment /!\\\\
        """
        if "raise_alarm" not in self.authorized_keys:
            raise IllegalAction("Impossible to send alarms with this action type.")
        orig_ = copy.deepcopy(self._raise_alarm)
        try:
            self._aux_affect_object_bool(
                values,
                "raise alarm",
                self.dim_alarms,
                self.alarms_area_names,
                np.arange(self.dim_alarms),
                self._raise_alarm,
            )
            self._modif_alarm = True
        except Exception as exc_:
            self._raise_alarm[:] = orig_
            raise IllegalAction(
                f"Impossible to modify the alarm with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def raise_alert(self) -> np.ndarray:
        """
        Property to raise alert.

        If you set it to ``True`` an alert is raised for the given line, otherwise no alert is raised.

        Notes
        -----
        
        .. code-block:: python

            import grid2op
            env_name = "l2rpn_idf_2023"  # chose an environment that supports the alert feature
            env = grid2op.make(env_name)
            act = env.action_space()

            act.raise_alert = [0]
            # this act will raise an alert on the powerline attackable 0 (powerline concerned will be action.alertable_line_ids[0])

        """
        res = copy.deepcopy(self._raise_alert)
        res.flags.writeable = False
        return res

    @raise_alert.setter
    def raise_alert(self, values):
        if "raise_alert" not in self.authorized_keys:
            raise IllegalAction("Impossible to send alerts with this action type.")
        orig_ = copy.deepcopy(self._raise_alert)
        try:
            self._aux_affect_object_bool(
                values,
                "raise alert",
                self.dim_alerts,
                self.alertable_line_names,
                np.arange(self.dim_alerts),
                self._raise_alert,
            )
            self._modif_alert = True
        except Exception as exc_:
            self._raise_alert[:] = orig_
            raise IllegalAction(
                f"Impossible to modify the alert with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    def _aux_affect_object_float(
        self,
        values,
        name_el,
        nb_els,
        name_els,
        inner_vect,
        outer_vect,
    ):
        """
        INTERNAL USE ONLY

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        NB : this do not set the _modif_set_bus attribute. It is expected to be set in the property setter.
        This is not set here, because it's recursive and if it fails at a point, it would be set for nothing

        values: the new values to set
        name_el: "load"
        nb_els: self.n_load
        inner_vect: self.load_pos_topo_vect
        name_els: self.name_load
        outer_vect: self._set_topo_vect

        will modify outer_vect[inner_vect]
        """
        if isinstance(values, (bool, dt_bool)):
            raise IllegalAction(
                f"Impossible to set {name_el} values with a single boolean."
            )
        elif isinstance(values, (int, dt_int, np.int64)):
            raise IllegalAction(
                f"Impossible to set {name_el} values with a single integer."
            )
        elif isinstance(values, (float, dt_float, np.float64)):
            raise IllegalAction(
                f"Impossible to set {name_el} values with a single float."
            )
        elif isinstance(values, tuple):
            # i provide a tuple: load_id, new_vals
            if len(values) != 2:
                raise IllegalAction(
                    f"when set with tuple, this tuple should have size 2 and be: {name_el}_id, new_bus "
                    f"eg. (3, 0.0)"
                )
            el_id, new_val = values
            if isinstance(new_val, (bool, dt_bool)):
                raise IllegalAction(
                    f"new_val should be a float. A boolean was provided"
                )

            try:
                new_val = float(new_val)
            except Exception as exc_:
                raise IllegalAction(
                    f'new_val should be convertible to a float. Error was : "{exc_}"'
                )

            if isinstance(el_id, (float, dt_float, np.float64)):
                raise IllegalAction(
                    f"{name_el}_id should be integers you provided float!"
                )
            if isinstance(el_id, (bool, dt_bool)):
                raise IllegalAction(
                    f"{name_el}_id should be integers you provided bool!"
                )
            if isinstance(el_id, str):
                raise IllegalAction(
                    f"{name_el}_id should be integers you provided string "
                    f"(hint: you can use a dictionary to set the bus by name eg. "
                    f"act.{name_el}_set_bus = {{act.name_{name_el}[0] : 1, act.name_{name_el}[1] : "
                    f"0.0}} )!"
                )

            try:
                el_id = int(el_id)
            except Exception as exc_:
                raise IllegalAction(
                    f'{name_el}_id should be convertible to integer. Error was : "{exc_}"'
                )
            if el_id < 0:
                raise IllegalAction(
                    f"Impossible to set the bus of a {name_el} with negative id"
                )
            if el_id >= nb_els:
                raise IllegalAction(
                    f"Impossible to set a {name_el} id {el_id} because there are only "
                    f"{nb_els} on the grid (and in python id starts at 0)"
                )
            if np.isfinite(new_val):
                outer_vect[inner_vect[el_id]] = new_val
            return
        elif isinstance(values, np.ndarray):
            if (
                isinstance(values.dtype, int)
                or values.dtype == dt_int
                or values.dtype == np.int64
            ):
                # for this the user explicitly casted it as integer, this won't work.
                raise IllegalAction(f"{name_el}_id should be floats you provided int!")

            if isinstance(values.dtype, bool) or values.dtype == dt_bool:
                raise IllegalAction(
                    f"{name_el}_id should be floats you provided boolean!"
                )
            try:
                values = values.astype(dt_float)
            except Exception as exc_:
                raise IllegalAction(
                    f'{name_el}_id should be convertible to float. Error was : "{exc_}"'
                )
            indx_ok = np.isfinite(values)
            outer_vect[inner_vect[indx_ok]] = values[indx_ok]
            return
        elif isinstance(values, list):
            # 2 cases: list of tuple, or list (convertible to numpy array)
            if len(values) == nb_els:
                # 2 cases: either i set all loads in the form [(0,..), (1,..), (2,...)]
                # or i should have converted the list to np array
                if isinstance(values, (bool, dt_bool)):
                    raise IllegalAction(
                        f"Impossible to set {name_el} values with a single boolean."
                    )
                elif isinstance(values, (int, dt_int, np.int64)):
                    raise IllegalAction(
                        f"Impossible to set {name_el} values with a single integer."
                    )
                elif isinstance(values, (float, dt_float, np.float64)):
                    raise IllegalAction(
                        f"Impossible to set {name_el} values with a single float."
                    )
                elif isinstance(values[0], tuple):
                    # list of tuple, handled below
                    # TODO can be somewhat "hacked" if the type of the object on the list is not always the same
                    pass
                else:
                    # get back to case where it's a full vector
                    values = np.array(values)
                    self._aux_affect_object_float(
                        values,
                        name_el,
                        nb_els,
                        name_els,
                        inner_vect=inner_vect,
                        outer_vect=outer_vect,
                    )
                    return

            # expected list of tuple, each tuple is a pair with load_id, new_vals: example: [(0, -1.0), (2,2.7)]
            for el in values:
                if len(el) != 2:
                    raise IllegalAction(
                        f"If input is a list, it should be a  list of pair (el_id, new_val) "
                        f"eg. [(0, 1.0), (2, 2.7)]"
                    )
                el_id, new_val = el
                if isinstance(el_id, str):
                    tmp = (name_els == el_id).nonzero()[0]
                    if len(tmp) == 0:
                        raise IllegalAction(f"No known {name_el} with name {el_id}")
                    el_id = tmp[0]
                self._aux_affect_object_float(
                    (el_id, new_val),
                    name_el,
                    nb_els,
                    name_els,
                    inner_vect=inner_vect,
                    outer_vect=outer_vect,
                )
        elif isinstance(values, dict):
            # 2 cases: either key = load_id and value = new_bus or key = load_name and value = new bus
            for key, new_val in values.items():
                if isinstance(key, str):
                    tmp = (name_els == key).nonzero()[0]
                    if len(tmp) == 0:
                        raise IllegalAction(f"No known {name_el} with name {key}")
                    key = tmp[0]
                self._aux_affect_object_float(
                    (key, new_val),
                    name_el,
                    nb_els,
                    name_els,
                    inner_vect=inner_vect,
                    outer_vect=outer_vect,
                )
        else:
            raise IllegalAction(
                f"Impossible to modify the {name_el} with inputs {values}. "
                f"Please see the documentation."
            )

    @property
    def redispatch(self) -> np.ndarray:
        """
        Allows to retrieve (and affect) the redispatching setpoint of the generators.

        Returns
        -------
        res:
            A vector of integer, of size `act.n_gen` indicating what type of action is performed for
            each generator units. Note that these are the setpoint. The actual redispatching that will
            be available might be different. See :ref:`generator-mod-el` for more information.

        Examples
        --------

        To retrieve the impact of the action on the generator unit, you can do:

        .. code-block:: python

            redisp = act.redispatch

        For each generator it will give the amount of redispatch this action wants to perform.

        To change the setpoint of the redispatching, you can do:

        .. code-block:: python

            # create an environment where i can modify everything
            import numpy as np
            import grid2op
            from grid2op.Action import CompleteAction
            env = grid2op.make("educ_case14_storage", test=True, action_class=CompleteAction)

            # create an action
            act = env.action_space()

            # method 1 : provide the full vector
            act.redispatch = np.ones(act.n_gen, dtype=float)  # only floats are accepted !

            # method 2: provide the index of the unit you want to modify
            act.redispatch = (1, 2.5)

            # method 3: provide a list of the units you want to modify
            act.redispatch = [(1, 2.5), (0, -1.3)]

            # method 4: change the generators by their name with a dictionary
            act.redispatch = {"gen_1_0": 2.0}

        .. note:: The "rule of thumb" to perform redispatching is to provide always
            the ID of an object AND its value. The ID should be an integer (or a name in some cases)
            and the value a float representing what amount of redispatching you want to perform on the
            unit with the associated ID.

        Notes
        -----
        It is a "property", you don't have to use parenthesis to access it:

        .. code-block:: python

            # valid code
            redisp = act.redispatch

            # invalid code, it will crash, do not run
            redisp = act.redispatch()
            # end do not run

        And neither should you uses parenthesis to modify it:

        .. code-block:: python

            # valid code
            act.redispatch = [(1, 2.5), (0, -1.3)]

            # invalid code, it will crash, do not run
            act.redispatch([(1, 2.5), (0, -1.3)])
            # end do not run

        Property cannot be set "directly", you have to use the `act.XXX = ..` syntax. For example:

        .. code-block:: python

            # valid code
            act.redispatch = [(1, 2.5), (0, -1.3)]

            # invalid code, it will raise an error, and even if it did not it would have not effect
            # do not run
            act.redispatch[1] = 2.5
            # end do not run

        .. note:: Be careful not to mix action to set something on a bus bar (where the values are integer,
            like "set_bus" or "set_status")
            and continuous action (where the values are float, like "redispatch" or "storage_p")

        """
        res = 1.0 * self._redispatch
        res.flags.writeable = False
        return res

    @redispatch.setter
    def redispatch(self, values):
        if "redispatch" not in self.authorized_keys:
            raise IllegalAction(
                "Impossible to perform redispatching with this action type."
            )
        orig_ = self.redispatch
        try:
            self._aux_affect_object_float(
                values,
                "redispatching",
                self.n_gen,
                self.name_gen,
                np.arange(self.n_gen),
                self._redispatch,
            )
            self._modif_redispatch = True
        except Exception as exc_:
            self._redispatch[:] = orig_
            raise IllegalAction(
                f"Impossible to modify the redispatching with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def storage_p(self) -> np.ndarray:
        """
        Allows to modify the setpoint of the storage units.

        It behaves similarly as :attr:`BaseAction.redispatch`. See the help there for more information.

        Notes
        ------
        The "load convention" is used for storage units. This means that:

        - if you ask a positive value, the storage unit will charge, power will be "taken" from the
          grid to the unit. The unit in this case will behave like a *load*
        - if you ask a negative value, the storage unit will discharge, power will be injected from
          the unit to the grid. The unit, in this case, will behave like a *generator*.

        For more information, feel free to consult the documentation :ref:`storage-mod-el` where more
        details are given about the modeling ot these storage units.
        """
        res = 1.0 * self._storage_power
        res.flags.writeable = False
        return res

    @storage_p.setter
    def storage_p(self, values):
        if "set_storage" not in self.authorized_keys:
            raise IllegalAction(
                "Impossible to perform storage action with this action type."
            )
        if self.n_storage == 0:
            raise IllegalAction(
                "Impossible to perform storage action with this grid (no storage unit"
                "available)"
            )
        orig_ = self.storage_p
        try:
            self._aux_affect_object_float(
                values,
                "storage",
                self.n_storage,
                self.name_storage,
                np.arange(self.n_storage),
                self._storage_power,
            )
            self._modif_storage = True
        except Exception as exc_:
            self._storage_power[:] = orig_
            raise IllegalAction(
                f"Impossible to modify the storage active power with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    @property
    def set_storage(self) -> np.ndarray:
        """Another name for the property :func:`BaseAction.storage_p`"""
        return self.storage_p

    @set_storage.setter
    def set_storage(self, values):
        self.storage_p = values

    @property
    def curtail(self) -> np.ndarray:
        """
        Allows to perfom some curtailment on some generators

        It behaves similarly as :attr:`BaseAction.redispatch`. See the help there for more information.

        For more information, feel free to consult the documentation :ref:`generator-mod-el` where more
        details are given about the modeling ot these storage units.
        """
        res = 1.0 * self._curtail
        res.flags.writeable = False
        return res

    @curtail.setter
    def curtail(self, values):
        if "curtail" not in self.authorized_keys:
            raise IllegalAction(
                "Impossible to perform curtailment action with this action type."
            )
        if not self.redispatching_unit_commitment_availble:
            raise IllegalAction(
                "Impossible to perform curtailment as it is not possible to compute redispatching. "
                'Your backend do not support "redispatching_unit_commitment_availble"'
            )

        orig_ = self.curtail
        try:
            self._aux_affect_object_float(
                values,
                "curtailment",
                self.n_gen,
                self.name_gen,
                np.arange(self.n_gen),
                self._curtail,
            )
            self._modif_curtailment = True
        except Exception as exc_:
            self._curtail[:] = orig_
            raise IllegalAction(
                f"Impossible to perform curtailment with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    def _aux_aux_convert_and_check_np_array(self, array_):
        try:
            array_ = np.array(array_)
        except Exception as exc_:
            raise IllegalAction(
                f"When setting the topology by substation and by giving a tuple, the "
                f"second element of the tuple should be convertible to a numpy "
                f'array of type int. Error was: "{exc_}"'
            )
        if (
            isinstance(array_.dtype, (bool, dt_bool))
            or array_.dtype == dt_bool
            or array_.dtype == bool
        ):
            raise IllegalAction(
                "To set substation topology, you need a vector of integers, and not a vector "
                "of bool."
            )
        elif (
            isinstance(array_.dtype, (float, dt_float))
            or array_.dtype == dt_float
            or array_.dtype == float
        ):
            raise IllegalAction(
                "To set substation topology, you need a vector of integers, and not a vector "
                "of float."
            )
        array_ = array_.astype(dt_int)
        if (array_ < -1).any():
            raise IllegalAction(
                f"Impossible to set element to bus {np.min(array_)}. Buses must be "
                f"-1, 0, 1 or 2."
            )
        if (array_ > type(self).n_busbar_per_sub).any():
            raise IllegalAction(
                f"Impossible to set element to bus {np.max(array_)}. Buses must be "
                f"-1, 0, 1 or 2."
            )
        return array_

    def _aux_set_bus_sub(self, values):
        cls = type(self)
        if isinstance(values, (bool, dt_bool)):
            raise IllegalAction(
                "Impossible to modify bus by substation with a single bool."
            )
        elif isinstance(values, (int, dt_int, np.int64, np.int32)):
            raise IllegalAction(
                "Impossible to modify bus by substation with a single integer."
            )
        elif isinstance(values, (float, dt_float, np.float64, np.float32)):
            raise IllegalAction(
                "Impossible to modify bus by substation with a single float."
            )
        elif isinstance(values, np.ndarray):
            # full topo vect
            if values.shape[0] != cls.dim_topo:
                raise IllegalAction(
                    "Impossible to modify bus when providing a full topology vector "
                    "that has not the right "
                )
            if values.dtype == dt_bool or values.dtype == bool:
                raise IllegalAction(
                    "When using a full vector for setting the topology, it should be "
                    "of integer types"
                )
            values = self._aux_aux_convert_and_check_np_array(values)
            self._set_topo_vect[:] = values
        elif isinstance(values, tuple):
            # should be a tuple (sub_id, new_topo)
            sub_id, topo_repr, nb_el = self._check_for_right_vectors_sub(values)
            topo_repr = self._aux_aux_convert_and_check_np_array(topo_repr)
            start_ = cls.sub_info[:sub_id].sum()
            end_ = start_ + nb_el
            self._set_topo_vect[start_:end_] = topo_repr
        elif isinstance(values, list):
            if len(values) == cls.dim_topo:
                # if list is the size of the full topo vect, it's a list representing it
                values = self._aux_aux_convert_and_check_np_array(values)
                self._aux_set_bus_sub(values)
                return
            # otherwise it should be a list of tuples: [(sub_id, topo), (sub_id, topo)]
            for el in values:
                if not isinstance(el, tuple):
                    raise IllegalAction(
                        "When provided a list, it should be a list of tuples: "
                        "[(sub_id, topo), (sub_id, topo), ... ] "
                    )
                self._aux_set_bus_sub(el)
        elif isinstance(values, dict):
            for sub_id, topo_repr in values.items():
                sub_id = self._aux_sub_when_dict_get_id(sub_id)
                self._aux_set_bus_sub((sub_id, topo_repr))
        else:
            raise IllegalAction(
                "Impossible to set the topology by substation with your input."
                "Please consult the documentation."
            )

    @property
    def sub_set_bus(self) -> np.ndarray:
        # TODO doc
        res = 1 * self.set_bus
        res.flags.writeable = False
        return res

    @sub_set_bus.setter
    def sub_set_bus(self, values):
        if "set_bus" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the substation bus (with "set") with this action type.'
            )
        orig_ = self.sub_set_bus
        try:
            self._aux_set_bus_sub(values)
            self._modif_set_bus = True
        except Exception as exc_:
            self._set_topo_vect[:] = orig_
            raise IllegalAction(
                f"Impossible to modify the substation bus with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    def _aux_aux_convert_and_check_np_array_change(self, array_):
        try:
            array_ = np.array(array_)
        except Exception as exc_:
            raise IllegalAction(
                f"When setting the topology by substation and by giving a tuple, the "
                f"second element of the tuple should be convertible to a numpy "
                f'array of type int. Error was: "{exc_}"'
            )
        if (
            isinstance(array_.dtype, (int, dt_int))
            or array_.dtype == dt_int
            or array_.dtype == int
        ):
            raise IllegalAction(
                "To change substation topology, you need a vector of bools, and not a vector "
                "of int."
            )
        elif (
            isinstance(array_.dtype, (float, dt_float))
            or array_.dtype == dt_float
            or array_.dtype == float
        ):
            raise IllegalAction(
                "To change substation topology, you need a vector of bools, and not a vector "
                "of float."
            )
        array_ = array_.astype(dt_bool)
        return array_

    def _check_for_right_vectors_sub(self, values):
        if len(values) != 2:
            raise IllegalAction(
                "Impossible to set the topology of a substation with a tuple which "
                "has not a size of 2 (substation_id, topology_representation)"
            )
        sub_id, topo_repr = values
        if isinstance(sub_id, (bool, dt_bool)):
            raise IllegalAction("Substation id should be integer")
        if isinstance(sub_id, (float, dt_float, np.float64)):
            raise IllegalAction("Substation id should be integer")
        try:
            el_id = int(sub_id)
        except Exception as exc_:
            raise IllegalAction(
                f"Substation id should be convertible to integer. "
                f'Error was "{exc_}"'
            )
        try:
            size_ = len(topo_repr)
        except Exception as exc_:
            raise IllegalAction(
                f"Topology cannot be set with your input." f'Error was "{exc_}"'
            )
        nb_el = self.sub_info[el_id]
        if size_ != nb_el:
            raise IllegalAction(
                f"To set topology of a substation, you must provide the full list of the "
                f"elements you want to modify. You provided a vector with {size_} components "
                f"while there are {self.sub_info[el_id]} on the substation."
            )

        return sub_id, topo_repr, nb_el

    def _aux_change_bus_sub(self, values):
        if isinstance(values, (bool, dt_bool)):
            raise IllegalAction(
                "Impossible to modify bus by substation with a single bool."
            )
        elif isinstance(values, (int, dt_int, np.int64)):
            raise IllegalAction(
                "Impossible to modify bus by substation with a single integer."
            )
        elif isinstance(values, (float, dt_float, np.float64)):
            raise IllegalAction(
                "Impossible to modify bus by substation with a single float."
            )
        elif isinstance(values, np.ndarray):
            # full topo vect
            if values.shape[0] != self.dim_topo:
                raise IllegalAction(
                    "Impossible to modify bus when providing a full topology vector "
                    "that has not the right size."
                )
            if values.dtype == dt_int or values.dtype == int:
                raise IllegalAction(
                    "When using a full vector for setting the topology, it should be "
                    "of bool types"
                )
            values = self._aux_aux_convert_and_check_np_array_change(values)
            self._change_bus_vect[:] = values
        elif isinstance(values, tuple):
            # should be a tuple (sub_id, new_topo)
            sub_id, topo_repr, nb_el = self._check_for_right_vectors_sub(values)

            topo_repr = self._aux_aux_convert_and_check_np_array_change(topo_repr)
            start_ = self.sub_info[:sub_id].sum()
            end_ = start_ + nb_el
            self._change_bus_vect[start_:end_] = topo_repr
        elif isinstance(values, list):
            if len(values) == self.dim_topo:
                # if list is the size of the full topo vect, it's a list representing it
                values = self._aux_aux_convert_and_check_np_array_change(values)
                self._aux_change_bus_sub(values)
                return
            # otherwise it should be a list of tuples: [(sub_id, topo), (sub_id, topo)]
            for el in values:
                if not isinstance(el, tuple):
                    raise IllegalAction(
                        "When provided a list, it should be a list of tuples: "
                        "[(sub_id, topo), (sub_id, topo), ... ] "
                    )
                self._aux_change_bus_sub(el)
        elif isinstance(values, dict):
            for sub_id, topo_repr in values.items():
                sub_id = self._aux_sub_when_dict_get_id(sub_id)
                self._aux_change_bus_sub((sub_id, topo_repr))
        else:
            raise IllegalAction(
                "Impossible to set the topology by substation with your input."
                "Please consult the documentation."
            )

    def _aux_sub_when_dict_get_id(self, sub_id):
        if isinstance(sub_id, str):
            tmp = (self.name_sub == sub_id).nonzero()[0]
            if len(tmp) == 0:
                raise IllegalAction(f"No substation named {sub_id}")
            sub_id = tmp[0]
        elif not isinstance(sub_id, int):
            raise IllegalAction(
                f"When using a dictionary it should be either with key = name of the "
                f"substation or key = id of the substation. You provided neither string nor"
                f"int but {type(sub_id)}."
            )
        return sub_id

    @property
    def sub_change_bus(self) -> np.ndarray:
        res = copy.deepcopy(self.change_bus)
        res.flags.writeable = False
        return res

    @sub_change_bus.setter
    def sub_change_bus(self, values):
        if "change_bus" not in self.authorized_keys:
            raise IllegalAction(
                'Impossible to modify the substation bus (with "change") with this action type.'
            )
        orig_ = self.sub_change_bus
        try:
            self._aux_change_bus_sub(values)
            self._modif_change_bus = True
        except Exception as exc_:
            self._change_bus_vect[:] = orig_
            raise IllegalAction(
                f"Impossible to modify the substation bus with your input. "
                f"Please consult the documentation. "
                f'The error was:\n"{exc_}"'
            )

    def curtailment_mw_to_ratio(self, curtailment_mw) -> np.ndarray:
        """
        Transform a "curtailment" given as maximum MW to the grid2op formalism (in ratio of gen_pmax)

        Parameters
        ----------
        curtailment_mw:
            Same type of inputs you can use in `act.curtail = ...`

        Returns
        -------
        A proper input to `act.curtail` with the converted input expressed in ratio of gen_pmax

        Examples
        --------

        If you want to limit the production of generator 1 (suppose its renewable) at 1.5MW
        then you can do:

        .. code-block:: python

            gen_id = 1
            amount_max = 1.5
            act.curtail = act.curtailment_mw_to_ratio([(gen_id, amount_max)])

        """
        values = self._curtail * self.gen_pmax
        self._aux_affect_object_float(
            curtailment_mw,
            "curtailment",
            self.n_gen,
            self.name_gen,
            np.arange(self.n_gen),
            values,
        )
        values /= self.gen_pmax
        values[values >= 1.0] = 1.0
        values[values < 0.0] = -1.0
        return values

    @property
    def curtail_mw(self) -> np.ndarray:
        """
        Allows to perfom some curtailment on some generators in MW (by default in grid2Op it should be expressed
        in ratio of gen_pmax)

        It behaves similarly as :attr:`BaseAction.redispatch`. See the help there for more information.

        For more information, feel free to consult the documentation :ref:`generator-mod-el` where more
        details are given about the modeling ot these storage units.
        
        .. warnings:
            We remind that "curtailment" will limit the number of MW produce by renewable energy sources. The agent
            is asked to provide the limit it wants and not the amount of MW it wants the generator to be cut off.
            
            For example, if a generator with a Pmax of 100 produces 55MW and you ask to "curtail_mw = 15" for this generator,
            its production will be limited to 15 MW (then droping from 55MW to 15MW) so loosing 40MW (and not 15 !)
            
        """
        res = 1.0 * self._curtail * self.gen_pmax
        res[res < 0.0] = -1.0
        res.flags.writeable = False
        return res

    @curtail_mw.setter
    def curtail_mw(self, values_mw):
        self.curtail = self.curtailment_mw_to_ratio(values_mw)

    def limit_curtail_storage(self,
                              obs: "BaseObservation",
                              margin: float=10.,
                              do_copy: bool=False,
                              _tol_equal : float=0.01) -> Tuple["BaseAction", np.ndarray, np.ndarray]:
        """
        This function tries to limit the possibility to end up
        with a "game over" because actions on curtailment or storage units (see the "Notes" section
        for more information).
        
        It will modify the action (unless `do_copy` is `True`) from a given observation `obs`.
        It limits the curtailment / storage unit to ensure that the
        amount of MW curtailed / taken to-from the storage units
        are within `-sum(obs.gen_margin_down)` and `sum(obs.gen_margin_up)`
        
        The `margin` parameter is here to allows to "take into account" the uncertainties. Indeed, if you 
        limit only to `-sum(obs.gen_margin_down)` and `sum(obs.gen_margin_up)`, because you don't know 
        how much the production will vary (due to loads, or intrisinc variability of
        renewable energy sources). The higher `margin` the less likely you will end up with 
        a "game over" but the more your action will possibly be affected. The lower
        this parameter, the more likely you will end up with a game over but the less
        your action will be impacted. It represents a certain amount of `MW`.
        
        Notes
        -------
        
        At each time, the environment ensures that the following equations are met:

        1) for each controlable generators $p^{(c)}_{min} <= p^{(c)}_t <= p^{(c)}_{max}$
        2) for each controlable generators $-ramp_{min}^{(c)} <= p^{(c)}_t - p^{(c)}_{t-1} <= ramp_{max}^{(c)}$
        3) at each step the sum of MW curtailed and the total contribution of storage units 
           is absorbed by the controlable generators so that the total amount of power injected 
           at this step does not change: 
           $\sum_{\text{all generators } g} p^{(g, scenario)}_t = \sum_{\text{controlable generators } c}  p^{(c)}_t + \sum_{\text{storage unit } s} p^{s}_t + \sum_{\text{renewable generator} r} p^{(r)}_t$
           where $p^{(g)}_t$ denotes the productions of generator $g$ in the input data "scenario" 
           (*ie* "in the current episode", "before any modification", "decided by the market / central authority").

        In the above equations, `\sum_{\text{storage unit } s} p^{s}_t` are controled by the action (thanks to the storage units)
        and `\sum_{\text{renewable generator} r} p^{(r)}_t` are controlled by the curtailment.
        
        `\sum_{\text{all generators } g} p^{(g, scenario)}_t` are input data from the environment (that cannot be modify).
        
        The exact value of each `p^{(c)}_t` (for each controlable generator) is computed by an internal routine of the
        environment. 
        
        The constraint comes from the fact that `\sum_{\text{controlable generators } c}  p^{(c)}_t` is determined by the last equation
        above but at the same time the values of each `p^{(c)}_t` (for each controllable generator) is heavily constrained
        by equations 1) and 2).

        .. note::
            This argument and the :func:`grid2op.Parameters.Parameters.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION` have the same objective:
            prevent an agent to do some curtailment too strong for the grid.
            
            When using  :func:`grid2op.Parameters.Parameters.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION`, 
            the environment will do it knowing exactly what will happen next (its a bit "cheating") and limit 
            exactly the action to exactly right amount.
            
            Using :func:`grid2op.Aciton.BaseAction.limit_curtail_storage` is always feasible, but less precise
            and subject to uncertainties.
        
        .. warning::
            If the action has no effect (for example you give a limit of the curtailment above the
            actual production of renewable generators) then regardless of the "margin" parameter
            your action will be declared "legal" which may cause unfeasibility in the future.
        
        Parameters
        ----------
        obs : ``Observation``
            The current observation. The main attributes used for the observation are 
            `obs.gen_margin_down` and `obs.gen_margin_up`.
            
        margin : ``float``, optional
            The "margin" taken from the controlable generators "margin" to 
            "take into account" when limiting the action 
            (see description for more information), by default 10.
            
        do_copy : ``bool``, optional
            Whether to make a copy of the current action (if set to ``True``) or to modify the
            action "in-place" (default, when ``False``)

        Returns
        -------
        `Action`, np.ndarray, np.ndarray:
            
            - `act`: the action after the storage unit / curtailment are modified (by default it's also `self`)
            - `res_add_curtailed`: the modification made to the curtailment
            - `res_add_storage`: the modification made to the storage units
            
        """
        cls = type(self)
        if do_copy:
            res = copy.deepcopy(self)
        else:
            res = self
            
        res_add_storage = np.zeros(cls.n_storage, dtype=dt_float)
        res_add_curtailed = np.zeros(cls.n_gen, dtype=dt_float)
        
        max_down = obs.gen_margin_down.sum()
        max_up = obs.gen_margin_up.sum()
        
        # storage
        total_mw_storage = res._storage_power.sum()
        total_storage_consumed = res._storage_power.sum()
        
        # curtailment
        gen_curtailed = (np.abs(res._curtail + 1) >= 1e-7) & cls.gen_renewable
        gen_curtailed &= ( (obs.gen_p > res._curtail * cls.gen_pmax) | (obs.gen_p_before_curtail > obs.gen_p ))
        gen_p_after_max = (res._curtail * cls.gen_pmax)[gen_curtailed]
        
        # I might have a problem because curtailment decreases too rapidly (ie i set a limit too low)
        prod_after_down = np.minimum(gen_p_after_max, obs.gen_p[gen_curtailed])
        # I might have a problem because curtailment increase too rapidly (limit was low and I set it too high too
        # rapidly)
        prod_after_up = np.minimum(gen_p_after_max, obs.gen_p_before_curtail[gen_curtailed])
        gen_p_after = np.maximum(prod_after_down, prod_after_up)
        mw_curtailed = obs.gen_p[gen_curtailed] - gen_p_after
        mw_curtailed_down = 1.0 * mw_curtailed
        mw_curtailed_down[mw_curtailed_down < 0.] = 0.
        mw_curtailed_up = -1.0 * mw_curtailed
        mw_curtailed_up[mw_curtailed_up < 0.] = 0.
        total_mw_curtailed_down = mw_curtailed_down.sum()
        total_mw_curtailed_up = mw_curtailed_up.sum()
        total_mw_curtailed = total_mw_curtailed_down - total_mw_curtailed_up
        total_mw_act = total_mw_curtailed + total_mw_storage
        
        if (total_mw_act > 0) and (total_mw_act > max_up - margin):
            # controlable generators should be asked to increase their production too much, I need to limit
            # the storage unit (consume too much) or the curtailment (curtailment too strong)
            if max_up < margin + _tol_equal:
                # not enough ramp up anyway so I don't do anything
                res_add_storage[:] = -res._storage_power
                res_add_curtailed[gen_curtailed] = obs.gen_p[gen_curtailed] / obs.gen_pmax[gen_curtailed] - res._curtail[gen_curtailed]
                res._storage_power[:] = 0.  # don't act on storage
                res._curtail[gen_curtailed] = -1  # reset curtailment
                
            else:
                remove_mw = total_mw_act - (max_up - margin)
                # fix curtailment
                if total_mw_curtailed_down > 0.: 
                    remove_curtail_mw = remove_mw * total_mw_curtailed_down / (total_mw_curtailed_down + total_mw_storage)
                    tmp_ = mw_curtailed_down / total_mw_curtailed_down * remove_curtail_mw / cls.gen_pmax[gen_curtailed]
                    res_add_curtailed[gen_curtailed] = tmp_
                    res._curtail[gen_curtailed] += tmp_ 
                    
                # fix storage
                if total_storage_consumed > 0.:
                    # only consider storage units that consume something (do not attempt to modify the others)
                    do_storage_consum = res._storage_power > 0. 
                    remove_storage_mw =  remove_mw * total_mw_storage / (total_mw_curtailed_down + total_mw_storage)
                    tmp_ = -(res._storage_power[do_storage_consum] * 
                             remove_storage_mw / res._storage_power[do_storage_consum].sum())
                    res._storage_power[do_storage_consum] += tmp_
                    res_add_storage[do_storage_consum] = tmp_
                    
        elif (total_mw_act < 0) and (total_mw_act < -max_down + margin):
            # controlable generators should be asked to decrease their production too much, I need to limit
            # the storage unit (produce too much) or the curtailment (curtailment too little)
            if max_down < margin + _tol_equal:
                # not enough ramp down anyway so I don't do anything
                res_add_storage[:] = -res._storage_power
                res_add_curtailed[gen_curtailed] = obs.gen_p[gen_curtailed] / obs.gen_pmax[gen_curtailed] - res._curtail[gen_curtailed]
                res._storage_power[:] = 0.  # don't act on storage
                res._curtail[gen_curtailed] = -1  # reset curtailment
            else:
                add_mw = -(total_mw_act + (max_down - margin))
                # fix curtailment  => does not work at all !
                if total_mw_curtailed_up > 0.: 
                    add_curtail_mw = add_mw * total_mw_curtailed_up / (total_mw_curtailed_up + total_mw_storage)
                    tmp_ = (obs.gen_p_before_curtail[gen_curtailed] * res._curtail[gen_curtailed] - mw_curtailed_up / total_mw_curtailed_up * add_curtail_mw )/ cls.gen_pmax[gen_curtailed]
                    res_add_curtailed[gen_curtailed] = tmp_ - res._curtail[gen_curtailed]
                    res._curtail[gen_curtailed] = tmp_ 
                    
                # fix storage
                if total_storage_consumed < 0.:
                    # only consider storage units that consume something (do not attempt to modify the others)
                    do_storage_prod = res._storage_power < 0. 
                    remove_storage_mw = add_mw * total_mw_storage / (total_mw_curtailed_up + total_mw_storage)
                    tmp_ = (res._storage_power[do_storage_prod] * 
                             remove_storage_mw / res._storage_power[do_storage_prod].sum())
                    res._storage_power[do_storage_prod] += tmp_
                    res_add_storage[do_storage_prod] = tmp_
        return res, res_add_curtailed, res_add_storage

    def _aux_decompose_as_unary_actions_change(self, cls, group_topo, res):
        if group_topo:
            tmp = cls()
            tmp._modif_change_bus = True
            tmp._change_bus_vect = copy.deepcopy(self._change_bus_vect)
            res["change_bus"] = [tmp]
        else:
            subs_changed = cls.grid_objects_types[self._change_bus_vect, cls.SUB_COL]
            subs_changed = np.unique(subs_changed)
            res["change_bus"] = []
            for sub_id in subs_changed:
                tmp = cls()
                tmp._modif_change_bus = True
                mask_sub = cls.grid_objects_types[:, cls.SUB_COL] == sub_id
                tmp._change_bus_vect[mask_sub] = self._change_bus_vect[mask_sub]
                res["change_bus"].append(tmp)

    def _aux_decompose_as_unary_actions_change_ls(self, cls, group_line_status, res):
        if group_line_status:
            tmp = cls()
            tmp._modif_change_status = True
            tmp._switch_line_status = copy.deepcopy(self._switch_line_status)
            res["change_line_status"] = [tmp]
        else:
            lines_changed = (self._switch_line_status).nonzero()[0]
            res["change_line_status"] = []
            for l_id in lines_changed:
                tmp = cls()
                tmp._modif_change_status = True   
                tmp._switch_line_status[l_id] = True
                res["change_line_status"].append(tmp)

    def _aux_decompose_as_unary_actions_set(self, cls, group_topo, res):
        if group_topo:
            tmp = cls()
            tmp._modif_set_bus = True
            tmp._set_topo_vect = 1 * self._set_topo_vect
            res["set_bus"] = [tmp]
        else:
            subs_changed = cls.grid_objects_types[self._set_topo_vect != 0, cls.SUB_COL]
            subs_changed = np.unique(subs_changed)
            res["set_bus"] = []
            for sub_id in subs_changed:
                tmp = cls()
                tmp._modif_set_bus = True
                mask_sub = cls.grid_objects_types[:, cls.SUB_COL] == sub_id    
                tmp._set_topo_vect[mask_sub] = self._set_topo_vect[mask_sub]
                res["set_bus"].append(tmp)
             
    def _aux_decompose_as_unary_actions_set_ls(self, cls, group_line_status, res):
        if group_line_status:
            tmp = cls()
            tmp._modif_set_status = True
            tmp._set_line_status = 1 * self._set_line_status
            res["set_line_status"] = [tmp]
        else:
            lines_changed = (self._set_line_status != 0).nonzero()[0]
            res["set_line_status"] = []
            for l_id in lines_changed:
                tmp = cls()
                tmp._modif_set_status = True   
                tmp._set_line_status[l_id] = self._set_line_status[l_id]
                res["set_line_status"].append(tmp)
    
    def _aux_decompose_as_unary_actions_redisp(self, cls, group_redispatch, res):
        if group_redispatch:
            tmp = cls()
            tmp._modif_redispatch = True
            tmp._redispatch = 1. * self._redispatch
            res["redispatch"] = [tmp]
        else:
            gen_changed = (np.abs(self._redispatch) >= 1e-7).nonzero()[0]
            res["redispatch"] = []
            for g_id in gen_changed:
                tmp = cls()
                tmp._modif_redispatch = True   
                tmp._redispatch[g_id] = self._redispatch[g_id]
                res["redispatch"].append(tmp)
                
    def _aux_decompose_as_unary_actions_storage(self, cls, group_storage, res):
        if group_storage:
            tmp = cls()
            tmp._modif_storage = True
            tmp._storage_power = 1. * self._storage_power
            res["set_storage"] = [tmp]
        else:
            sto_changed = (np.abs(self._storage_power) >= 1e-7).nonzero()[0]
            res["set_storage"] = []
            for s_id in sto_changed:
                tmp = cls()
                tmp._modif_storage = True   
                tmp._storage_power[s_id] = self._storage_power[s_id]
                res["set_storage"].append(tmp)
                
    def _aux_decompose_as_unary_actions_curtail(self, cls, group_curtailment, res):
        if group_curtailment:
            tmp = cls()
            tmp._modif_curtailment = True
            tmp._curtail = 1. * self._curtail
            res["curtail"] = [tmp]
        else:
            gen_changed = (np.abs(self._curtail + 1.) >= 1e-7).nonzero()[0]  #self._curtail != -1
            res["curtail"] = []
            for g_id in gen_changed:
                tmp = cls()
                tmp._modif_curtailment = True   
                tmp._curtail[g_id] = self._curtail[g_id]
                res["curtail"].append(tmp)
            
    def decompose_as_unary_actions(self,
                                   group_topo=False,
                                   group_line_status=False,
                                   group_redispatch=True,
                                   group_storage=True,
                                   group_curtail=True) -> Dict[Literal["change_bus",
                                                                       "set_bus",
                                                                       "change_line_status",
                                                                       "set_line_status",
                                                                       "redispatch",
                                                                       "set_storage",
                                                                       "curtail"],
                                                               List["BaseAction"]]:
        """This function allows to split a possibly "complex" action into its
        "unary" counterpart.
        
        By "unary" action here we mean "action that acts on only 
        one type". For example an action that only `set_line_status` is 
        unary but an action that acts on `set_line_status` AND `set_bus` is
        not. Also, note that an action that acts on `set_line_status`
        and `change_line_status` is not considered as "unary" by this method.
        
        This functions output a dictionnary with up to 7 keys:
        
        -  "change_bus" if the action affects the grid with `change_bus`. 
           In this case the value associated with this key is a list containing
           only action that performs `change_bus`
        -  "set_bus" if the action affects the grid with`set_bus`. 
           In this case the value associated with this key is a list containing
           only action that performs `set_bus`
        -  "change_line_status" if the action affects the grid with `change_line_status`
           In this case the value associated with this key is a list containing
           only action that performs `change_line_status`
        -  "set_line_status" if the action affects the grid with `set_line_status`
           In this case the value associated with this key is a list containing
           only action that performs `set_line_status`
        -  "redispatch" if the action affects the grid with `redispatch`
           In this case the value associated with this key is a list containing
           only action that performs `redispatch`
        -  "set_storage" if the action affects the grid with `set_storage`
           In this case the value associated with this key is a list containing
           only action that performs `set_storage`
        -  "curtail" if the action affects the grid with `curtail`
           In this case the value associated with this key is a list containing
           only action that performs `curtail`

        **NB** if the action is a "do nothing" action type, then this function will
        return an empty dictionary.
        
        .. versionadded:: 1.9.1
        
        Notes
        -------
        If the action is not ambiguous (ie it is valid and can be correctly
        understood by grid2op) and if you sum all the actions in all 
        the lists of all the keys of the
        dictionnary returned by this function, you will retrieve exactly the
        current action.
        
        For example:
        
        .. code-block:: python
        
            import grid2op
            
            env_name = "l2rpn_case14_sandbox"
            env = grid2op.make(env_name, ...)
            
            act = env.action_space({"curtail": [(4, 0.8), (5, 0.7)],
                                    "set_storage": [(0, +1.), (1, -1.)],
                                    "redispatch": [(0, +1.), (1, -1.)],
                                    "change_line_status": [2, 3],
                                    "set_line_status": [(0, -1), (1, -1)],
                                    "set_bus": {"loads_id": [(0, 2), (1, 2)],
                                                "generators_id": [(0, 2)]},
                                    "change_bus": {"loads_id": [2, 3],
                                                "generators_id": [1]}
                                    })
            res = act.decompose_as_unary_actions()
            tmp = env.action_space()
            for k, v in res.items():
                for a in v:
                    tmp += a
            assert tmp == act
        
        
        Parameters
        ----------
        group_topo : bool, optional
            This flag allows you to control the size of the `change_bus` and 
            `set_bus` values. If it's ``True`` then the values 
            associated with this keys will be unique (made of one single element)
            that will affect all the elements affected by this action (grouping them 
            all together)
            Otherwise, it will counts as many elements as the number of 
            substations affected by a `change_bus` or a `set_bus`. Each action
            returned by this will then act on only one substation. By default False (meaning there will be as many element 
            in `change_bus` as the number of substations affected by a `change_bus` 
            action [same for `set_bus`])
        group_line_status : bool, optional
            Whether to group the line status in one single action (so the values associated
            with the keys `set_line_status` and `change_line_status` will count
            exactly one element - if present) or not. By default False (meaning there will be as many element 
            in `change_line_status` as the number of lines affected by a 
            `change_line_status` action [same for `set_line_status`] : if 
            the original action `set` the status of two powerlines, then the 
            value associated with `set_line_status` will count 2 elements: 
            the first action will `set` the status of the first line affected by 
            the action, the second will... `set` the status of the
            second line affected by the action)
        group_redispatch : bool, optional
            same behaviour as `group_line_status` but for "generators" and 
            "redispatching" instead of "powerline" and `set_line_status`, by default True (meaning the value associated with 
            the key `redispatch` will be a list of one element performing 
            a redispatching action on all generators modified by the current action)
        group_storage : bool, optional
            same behaviour as `group_line_status` but for "storage units" and 
            "set setpoint" instead of "powerline" and `set_line_status`, by default True (meaning the value associated with 
            the key `set_storage` will be a list of one element performing 
            a set point action on all storage units modified by the current action)
        group_curtail : bool, optional
            same behaviour as `group_line_status` but for "generators" and 
            "curtailment" instead of "powerline" and `set_line_status`, , by default True (meaning the value associated with 
            the key `curtail` will be a list of one element performing 
            a curtailment on all storage generators modified by the current action)

        Returns
        -------
        dict
            See description for further information.
        """
        res = {}
        cls = type(self)
        if self._modif_change_bus:
            self._aux_decompose_as_unary_actions_change(cls, group_topo, res)
        if self._modif_set_bus:
            self._aux_decompose_as_unary_actions_set(cls, group_topo, res)
        if self._modif_change_status:
            self._aux_decompose_as_unary_actions_change_ls(cls, group_line_status, res)
        if self._modif_set_status:
            self._aux_decompose_as_unary_actions_set_ls(cls, group_line_status, res)
        if self._modif_redispatch:
            self._aux_decompose_as_unary_actions_redisp(cls, group_redispatch, res)
        if self._modif_storage:
            self._aux_decompose_as_unary_actions_storage(cls, group_storage, res)
        if self._modif_curtailment:
            self._aux_decompose_as_unary_actions_curtail(cls, group_curtail, res)
        return res
