# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np
from typing import Tuple, Union
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from grid2op.Action.baseAction import BaseAction
from grid2op.dtypes import dt_int, dt_bool, dt_float
from grid2op.Space import GridObjects


# TODO see if it can be done in c++ easily
class ValueStore:
    """
    USE ONLY IF YOU WANT TO CODE A NEW BACKEND

    .. warning:: /!\\\\ Internal, do not modify, alter, change, override the implementation unless you know what you are doing /!\\\\

        If you override them you might even notice some extremely weird behaviour. It's not "on purpose", we are aware of
        it  but we won't change it (for now at least)
        
    .. warning::
        Objects from this class should never be created by anyone except by objects of the :class:`grid2op.Action._backendAction._BackendAction`
        when they are created or when instances of `_BackendAction` are process *eg* with :func:`_BackendAction.__call__` or 
        :func:`_BackendAction.get_loads_bus` etc.
    
    There are two correct uses for this class:
    
    #. by iterating manually with the `for xxx in value_stor_instance: `
    #. by checking which objects have been changed (with :attr:`ValueStore.changed`) and then check the 
       new value of the elements **changed** with :attr:`ValueStore.values` [el_id]

    .. danger::
    
        You should never trust the values in :attr:`ValueStore.values` [el_id] if :attr:`ValueStore.changed` [el_id] is `False`.
        
        Access data (values) only when the corresponding "mask" (:attr:`ValueStore.changed`) is `True`. 
        
        This is, of course, ensured by default if you use the practical way of iterating through them with:
        
        .. code-block:: python
        
            load_p: ValueStore  # a ValueStore object named "load_p"
            
            for load_id, new_p in load_p:
                # do something
                
        In this case only "new_p" will be given if corresponding `changed` mask is true.

    Attributes
    ----------
    
            
    Examples
    ---------
    
    Say you have a "ValueStore" `val_sto` (in :class:`grid2op.Action._backendAction._BackendAction` you will end up manipulating 
    pretty much all the time `ValueStore` if you use it correctly, with :func:`_BackendAction.__call__` but also is you call
    :func:`_BackendAction.get_loads_bus`, :func:`_BackendAction.get_loads_bus_global`, :func:`_BackendAction.get_gens_bus`, ...)
    
    Basically, the "variables" named `prod_p`, `prod_v`, `load_p`, `load_q`, `storage_p`,
    `topo__`, `shunt_p`, `shunt_q`, `shunt_bus`, `backendAction.get_lines_or_bus()`,
    `backendAction.get_lines_or_bus_global()`, etc in the doc of :class:`grid2op.Action._backendAction._BackendAction`
    are all :class:`ValueStore`.
    
    Recommended usage:
    
    .. code-block:: python
    
        val_sto: ValueStore  # a ValueStore object named "val_sto"
        
        for el_id, new_val in val_sto:
            # do something    
        
        # less abstractly, say `load_p` is a ValueStore:
        # for load_id, new_p in load_p:
            # do the real changes of load active value in self._grid        
            # load_id => id of loads for which the active consumption changed
            # new_p => new load active consumption for `load_id`
            # self._grid.change_load_active_value(load_id, new_p)  # fictive example of course...
        
    
    More advanced / vectorized  usage (only do that if you found out your backend was
    slow because of the iteration in python above, this is error-prone and in general
    might not be worth it...):
    
    .. code-block:: python
    
        val_sto: ValueStore  # a ValueStore object named "val_sto"
        
        # less abstractly, say `load_p` is a ValueStore:
        # self._grid.change_all_loads_active_value(where_changed=load_p.changed,
                                                   new_vals=load_p.values[load_p.changed])
        # fictive example of couse, I highly doubt the self._grid
        # implements a method named exactly `change_all_loads_active_value`
        
        WARNING, DANGER AHEAD:
        Never trust the data in load_p.values[~load_p.changed], they might even be un intialized...
        
    """

    def __init__(self, size, dtype):
        ## TODO at the init it's mandatory to have everything at "1" here
        # if topo is not "fully connected" it will not work
        
        #: :class:`np.ndarray` 
        #: The new target values to be set in `backend._grid` in `apply_action`
        #: never use the values if the corresponding mask is set to `False`
        #: (it might be non initialized).
        self.values = np.empty(size, dtype=dtype)
        
        #: :class:`np.ndarray` (bool)
        #: Mask representing which values (stored in :attr:`ValueStore.values` ) are
        #: meaningful. The other values (corresponding to `changed=False` ) are meaningless.
        self.changed = np.full(size, dtype=dt_bool, fill_value=False)
        
        #: used internally for iteration
        self.last_index = 0
        self.__size = size

        if issubclass(dtype, dt_int):
            self.set_val = self._set_val_int
            self.change_val = self._change_val_int
        elif issubclass(dtype, dt_float):
            self.set_val = self._set_val_float
            self.change_val = self._change_val_float

    def _set_val_float(self, newvals):
        changed_ = np.isfinite(newvals)
        self.changed[changed_] = True
        self.values[changed_] = newvals[changed_]

    def _set_val_int(self, newvals):
        changed_ = newvals != 0
        self.changed[changed_] = True
        self.values[changed_] = newvals[changed_]

    def _change_val_int(self, newvals):
        changed_ = newvals & (self.values > 0)
        self.changed[changed_] = True
        self.values[changed_] = (1 - self.values[changed_]) + 2

    def _change_val_float(self, newvals):
        changed_ = np.abs(newvals) >= 1e-7
        self.changed[changed_] = True
        self.values[changed_] += newvals[changed_]

    def reset(self):
        self.changed[:] = False
        self.last_index = 0

    def change_status(self, switch, lineor_id, lineex_id, old_vect):
        if not switch.any():
            # nothing is modified so i stop here
            return

        # changed
        changed_ = switch

        # make it to ids
        id_chg_or = lineor_id[changed_]
        id_chg_ex = lineex_id[changed_]

        self.changed[id_chg_or] = True
        self.changed[id_chg_ex] = True

        # disconnect the powerlines
        me_or_bus = self.values[id_chg_or]
        me_ex_bus = self.values[id_chg_ex]
        was_connected = (me_or_bus > 0) | (me_ex_bus > 0)
        was_disco = ~was_connected

        # it was connected, i disconnect it
        self.values[id_chg_or[was_connected]] = -1
        self.values[id_chg_ex[was_connected]] = -1

        # it was disconnected, i reconnect it
        reco_or = id_chg_or[was_disco]
        reco_ex = id_chg_ex[was_disco]
        self.values[reco_or] = old_vect[reco_or]
        self.values[reco_ex] = old_vect[reco_ex]

    def set_status(self, set_status, lineor_id, lineex_id, old_vect):
        id_or = lineor_id
        id_ex = lineex_id

        # disco
        disco_ = set_status == -1
        reco_ = set_status == 1

        # make it to ids
        id_reco_or = id_or[reco_]
        id_reco_ex = id_ex[reco_]
        id_disco_or = id_or[disco_]
        id_disco_ex = id_ex[disco_]

        self.changed[id_reco_or] = True
        self.changed[id_reco_ex] = True
        self.changed[id_disco_or] = True
        self.changed[id_disco_ex] = True

        # disconnect the powerlines
        self.values[id_disco_or] = -1
        self.values[id_disco_ex] = -1

        # reconnect the powerlines
        # don't consider powerlines that have been already changed with topology
        # ie reconnect to the old bus only powerline from which we don't know the status
        id_reco_or = id_reco_or[self.values[id_reco_or] < 0]
        id_reco_ex = id_reco_ex[self.values[id_reco_ex] < 0]

        self.values[id_reco_or] = old_vect[id_reco_or]
        self.values[id_reco_ex] = old_vect[id_reco_ex]

    def get_line_status(self, lineor_id, lineex_id):
        return self.values[lineor_id], self.values[lineex_id]

    def update_connected(self, current_values):
        indx_conn = current_values.values > 0
        self.values[indx_conn] = current_values.values[indx_conn]

    def all_changed(self):
        self.reset()
        self.changed[:] = True

    def __getitem__(self, item):
        return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value
        self.changed[key] = value

    def __iter__(self):
        return self

    def __next__(self):
        res = None
        while self.last_index < self.values.shape[0]:
            if self.changed[self.last_index]:
                res = (self.last_index, self.values[self.last_index])
            self.last_index += 1
            if res is not None:
                break
        if res is not None:
            return res
        else:
            raise StopIteration

    def __len__(self):
        return self.__size

    def reorder(self, new_order):
        """reorder the element modified, this is use when converting backends only and should not be use
        outside of this usecase"""
        self.changed[:] = self.changed[new_order]
        self.values[:] = self.values[new_order]

    def copy_from_index(self, ref, index):
        self.reset()
        self.changed[:] = ref.changed[index]
        self.values[:] = ref.values[index]

    def __copy__(self):
        res = type(self)(self.values.shape[0], self.values.dtype.type)
        res.values[:] = self.values
        res.changed[:] = self.changed
        res.last_index = self.last_index
        res.__size = self.__size
        return res

    def __deepcopy__(self, memodict={}):
        res = type(self)(self.values.shape[0], self.values.dtype.type)
        res.values[:] = self.values
        res.changed[:] = self.changed
        res.last_index = self.last_index
        res.__size = self.__size
        return res

    def copy(self, other):
        """deepcopy, shallow or deep, without having to initialize everything again"""
        self.values[:] = other.values
        self.changed[:] = other.changed
        self.last_index = other.last_index
        self.__size = other.__size
        
    def force_unchanged(self, mask, local_bus):
        to_unchanged = local_bus == -1
        to_unchanged[~mask] = False
        self.changed[to_unchanged] = False
    
    def register_new_topo(self, current_topo: "ValueStore"):
        mask_co = current_topo.values >= 1
        self.values[mask_co] = current_topo.values[mask_co]
        

class _BackendAction(GridObjects):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Internal class, use at your own risk.

    This class "digest" the players / environment / opponent / voltage controlers "actions",
    and transform it to one single "state" that can in turn be process by the backend
    in the function :func:`grid2op.Backend.Backend.apply_action`.
    
    .. note::
        In a :class:`_BackendAction` only the state of the element that have been modified
        by an "entity" (agent, environment, opponent, voltage controler etc.) is given.
        
        We expect the backend to "remember somehow" the state of all the rest. 
        
        This is to save a lot of computation time for larger grid.
    
    .. note::
        You probably don't need to import the `_BackendAction` class (this is why
        we "hide" it),
        but the `backendAction` you will receive in `apply_action` is indeed
        a :class:`_BackendAction`, hence this documentation.
    
    If you want to use grid2op to develop agents or new time series, 
    this class should behave transparently for you and you don't really 
    need to spend time reading its documentation. 
    
    If you want to develop in grid2op and code a new backend, you might be interested in:
    
    - :func:`_BackendAction.__call__`
    - :func:`_BackendAction.get_loads_bus`
    - :func:`_BackendAction.get_loads_bus_global`
    - :func:`_BackendAction.get_gens_bus`
    - :func:`_BackendAction.get_gens_bus_global`
    - :func:`_BackendAction.get_lines_or_bus`
    - :func:`_BackendAction.get_lines_or_bus_global`
    - :func:`_BackendAction.get_lines_ex_bus`
    - :func:`_BackendAction.get_lines_ex_bus_global`
    - :func:`_BackendAction.get_storages_bus`
    - :func:`_BackendAction.get_storages_bus_global`
    - :func:`_BackendAction.get_shunts_bus_global`
    
    And in this case, for usage examples, see the examples available in:
    
    - https://github.com/rte-france/Grid2Op/tree/master/examples/backend_integration: a step by step guide to 
      code a new backend
    - :class:`grid2op.Backend.educPandaPowerBackend.EducPandaPowerBackend` and especially the 
      :func:`grid2op.Backend.educPandaPowerBackend.EducPandaPowerBackend.apply_action`
    - :ref:`create-backend-module` page of the documentation, especially the 
      :ref:`backend-action-create-backend` section
    
    Otherwise, "TL;DR" (only relevant when you want to implement the :func:`grid2op.Backend.Backend.apply_action`
    function, rest is not shown):
    
    .. code-block:: python
    
        def apply_action(self, backendAction: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:
            if backendAction is None:
                return
        
            (
                active_bus,
                (prod_p, prod_v, load_p, load_q, storage_p),
                topo__,
                shunts__,
            ) = backendAction()
            
            # change the active values of the loads
            for load_id, new_p in load_p:
                # do the real changes in self._grid
            
            # change the reactive values of the loads
            for load_id, new_q in load_q:
                # do the real changes in self._grid
            
            # change the active value of generators
            for gen_id, new_p in prod_p:
                # do the real changes in self._grid
                
            # for the voltage magnitude, pandapower expects pu but grid2op provides kV,
            # so we need a bit of change
            for gen_id, new_v in prod_v:
                # do the real changes in self._grid
            
            # process the topology :
            
            # option 1: you can directly set the element of the grid in the "topo_vect" 
            # order, for example you can modify in your solver the busbar to which
            # element 17 of `topo_vect` is computed (this is necessarily a local view of
            # the buses )
            for el_topo_vect_id, new_el_bus in topo__:
                # connect this object to the `new_el_bus` (local) in self._grid
            
            # OR !!! (use either option 1 or option 2.a or option 2.b - exclusive OR)
            
            # option 2: use "per element type" view (this is usefull)
            # if your solver has organized its data by "type" and you can
            # easily access "all loads" and "all generators" etc.
            
            # option 2.a using "local view": 
            # new_bus is either -1, 1, 2, ..., backendAction.n_busbar_per_sub
            lines_or_bus = backendAction.get_lines_or_bus()
            for line_id, new_bus in lines_or_bus:
                # connect "or" side of "line_id" to (local) bus `new_bus` in self._grid
                
            # OR !!! (use either option 1 or option 2.a or option 2.b - exclusive OR)
            
            # option 2.b using "global view":
            # new_bus is either 0, 1, 2, ..., backendAction.n_busbar_per_sub * backendAction.n_sub
            # (this suppose internally that your solver and grid2op have the same 
            # "ways" of labelling the buses...)
            lines_or_bus = backendAction.get_lines_or_bus_global()
            for line_id, new_bus in lines_or_bus:
                # connect "or" side of "line_id" to (global) bus `new_bus` in self._grid
            
            # now repeat option a OR b calling the right methods 
            # for each element types (*eg* get_lines_ex_bus, get_loads_bus, get_gens_bus,
            # get_storages_bus for "option a-like") 
            
            ######## end processing of the topology  ###############
            
            # now implement the shunts:
            
            if shunts__ is not None:
                shunt_p, shunt_q, shunt_bus = shunts__

                if (shunt_p.changed).any():
                    # p has changed for at least a shunt
                    for shunt_id, new_shunt_p in shunt_p:
                        # do the real changes in self._grid
                        
                if (shunt_q.changed).any():
                    # q has changed for at least a shunt
                    for shunt_id, new_shunt_q in shunt_q:
                        # do the real changes in self._grid
                        
                if (shunt_bus.changed).any():
                    # at least one shunt has been disconnected
                    # or has changed the buses
                    
                    # do like for normal topology with:
                    # option a -like (using local bus): 
                    for shunt_id, new_shunt_bus in shunt_bus:
                         ...
                    # OR
                    # option b -like (using global bus):
                    shunt_global_bus = backendAction.get_shunts_bus_global()
                    for shunt_id, new_shunt_bus in shunt_global_bus:
                        # connect shunt_id to (global) bus `new_shunt_bus` in self._grid
        
    .. warning::
        The steps shown here are generic and might not be optimised for your backend. This
        is why you probably do not see any of them directly in :class:`grid2op.Backend.PandaPowerBackend`
        (where everything is vectorized to make things fast **with pandapower**).
        
        It is probably a good idea to first get this first implementation up and running, passing 
        all the tests, and then to worry about optimization:
        
          The real problem is that programmers have spent far too much 
          time worrying about efficiency in the wrong places and at the wrong times; 
          premature optimization is the root of all evil (or at least most of it) 
          in programming.
        
        Donald Knuth, "*The Art of Computer Programming*"
        
    """

    def __init__(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
            This is handled by the environment !

        """
        GridObjects.__init__(self)
        cls = type(self)
        # last connected registered
        self.last_topo_registered: ValueStore = ValueStore(cls.dim_topo, dtype=dt_int)

        # topo at time t
        self.current_topo: ValueStore = ValueStore(cls.dim_topo, dtype=dt_int)
        
        # by default everything is on busbar 1
        self.last_topo_registered.values[:] = 1
        self.current_topo.values[:] = 1  

        # injection at time t
        self.prod_p: ValueStore = ValueStore(cls.n_gen, dtype=dt_float)
        self.prod_v: ValueStore = ValueStore(cls.n_gen, dtype=dt_float)
        self.load_p: ValueStore = ValueStore(cls.n_load, dtype=dt_float)
        self.load_q: ValueStore = ValueStore(cls.n_load, dtype=dt_float)
        self.storage_power: ValueStore = ValueStore(cls.n_storage, dtype=dt_float)

        self.activated_bus = np.full((cls.n_sub, cls.n_busbar_per_sub), dtype=dt_bool, fill_value=False)
        self.big_topo_to_subid: np.ndarray = np.repeat(
            list(range(cls.n_sub)), repeats=cls.sub_info
        )

        # shunts
        if cls.shunts_data_available:
            self.shunt_p: ValueStore = ValueStore(cls.n_shunt, dtype=dt_float)
            self.shunt_q: ValueStore = ValueStore(cls.n_shunt, dtype=dt_float)
            self.shunt_bus: ValueStore = ValueStore(cls.n_shunt, dtype=dt_int)
            self.current_shunt_bus: ValueStore = ValueStore(cls.n_shunt, dtype=dt_int)
            self.current_shunt_bus.values[:] = 1

        self._status_or_before: np.ndarray = np.ones(cls.n_line, dtype=dt_int)
        self._status_ex_before: np.ndarray = np.ones(cls.n_line, dtype=dt_int)
        self._status_or: np.ndarray = np.ones(cls.n_line, dtype=dt_int)
        self._status_ex: np.ndarray = np.ones(cls.n_line, dtype=dt_int)

        self._loads_bus = None
        self._gens_bus = None
        self._lines_or_bus = None
        self._lines_ex_bus = None
        self._storage_bus = None

    def __deepcopy__(self, memodict={}) -> Self:
        
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        """
        res = type(self)()
        # last connected registered
        res.last_topo_registered.copy(self.last_topo_registered)
        res.current_topo.copy(self.current_topo)
        res.prod_p.copy(self.prod_p)
        res.prod_v.copy(self.prod_v)
        res.load_p.copy(self.load_p)
        res.load_q.copy(self.load_q)
        res.storage_power.copy(self.storage_power)
        res.activated_bus[:, :] = self.activated_bus
        # res.big_topo_to_subid[:] = self.big_topo_to_subid  # cste
        cls = type(self)
        if cls.shunts_data_available:
            res.shunt_p.copy(self.shunt_p)
            res.shunt_q.copy(self.shunt_q)
            res.shunt_bus.copy(self.shunt_bus)
            res.current_shunt_bus.copy(self.current_shunt_bus)

        res._status_or_before[:] = self._status_or_before
        res._status_ex_before[:] = self._status_ex_before
        res._status_or[:] = self._status_or
        res._status_ex[:] = self._status_ex
        
        res._loads_bus = copy.deepcopy(self._loads_bus)
        res._gens_bus = copy.deepcopy(self._gens_bus)
        res._lines_or_bus = copy.deepcopy(self._lines_or_bus)
        res._lines_ex_bus = copy.deepcopy(self._lines_ex_bus)
        res._storage_bus = copy.deepcopy(self._storage_bus)
        
        return res

    def __copy__(self) -> Self:
        
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        """
        res = self.__deepcopy__()  # nothing less to do
        return res

    def reorder(self, no_load, no_gen, no_topo, no_storage, no_shunt) -> None:
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
            This is handled by BackendConverter, do not alter

        Reorder the element modified, this is use when converting backends only and should not be use
        outside of this usecase

        no_* stands for "new order"
        """
        self.last_topo_registered.reorder(no_topo)
        self.current_topo.reorder(no_topo)
        self.prod_p.reorder(no_gen)
        self.prod_v.reorder(no_gen)
        self.load_p.reorder(no_load)
        self.load_q.reorder(no_load)

        self.storage_power.reorder(no_storage)

        cls = type(self)
        if cls.shunts_data_available:
            self.shunt_p.reorder(no_shunt)
            self.shunt_q.reorder(no_shunt)
            self.shunt_bus.reorder(no_shunt)
            self.current_shunt_bus.reorder(no_shunt)

    def reset(self) -> None:
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
            This is called by the environment, do not alter.

        """
        # last known topo
        self.last_topo_registered.reset()

        # topo at time t
        self.current_topo.reset()

        # injection at time t
        self.prod_p.reset()
        self.prod_v.reset()
        self.load_p.reset()
        self.load_q.reset()
        self.storage_power.reset()
        
        # storage unit have their power reset to 0. each step
        self.storage_power.changed[:] = True
        self.storage_power.values[:] = 0.0

        # shunts
        cls = type(self)
        if cls.shunts_data_available:
            self.shunt_p.reset()
            self.shunt_q.reset()
            self.shunt_bus.reset()
            self.current_shunt_bus.reset()
            
        self.last_topo_registered.register_new_topo(self.current_topo)

    def all_changed(self) -> None:
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
            This is called by the environment, do not alter.
        """
        # last topo
        self.last_topo_registered.all_changed()

        # topo at time t
        self.current_topo.all_changed()

        # injection at time t
        self.prod_p.all_changed()
        self.prod_v.all_changed()
        self.load_p.all_changed()
        self.load_q.all_changed()
        self.storage_power.all_changed()

        # TODO handle shunts
        # shunts
        # if self.shunts_data_available:
        #     self.shunt_p.all_changed()
        #     self.shunt_q.all_changed()
        #     self.shunt_bus.all_changed()

    def set_redispatch(self, new_redispatching):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
            This is called by the environment, do not alter.
        """
        self.prod_p.change_val(new_redispatching)

    def _aux_iadd_inj(self, dict_injection):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
            Internal implementation of +=
            
        """
        if "load_p" in dict_injection:
            tmp = dict_injection["load_p"]
            self.load_p.set_val(tmp)
        if "load_q" in dict_injection:
            tmp = dict_injection["load_q"]
            self.load_q.set_val(tmp)
        if "prod_p" in dict_injection:
            tmp = dict_injection["prod_p"]
            self.prod_p.set_val(tmp)
        if "prod_v" in dict_injection:
            tmp = dict_injection["prod_v"]
            self.prod_v.set_val(tmp)
    
    def _aux_iadd_shunt(self, other):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
            Internal implementation of +=
            
        """
        shunts = {}
        if type(other).shunts_data_available:
            shunts["shunt_p"] = other.shunt_p
            shunts["shunt_q"] = other.shunt_q
            shunts["shunt_bus"] = other.shunt_bus

            arr_ = shunts["shunt_p"]
            self.shunt_p.set_val(arr_)
            arr_ = shunts["shunt_q"]
            self.shunt_q.set_val(arr_)
            arr_ = shunts["shunt_bus"]
            self.shunt_bus.set_val(arr_)
        self.current_shunt_bus.values[self.shunt_bus.changed] = self.shunt_bus.values[self.shunt_bus.changed]

    def _aux_iadd_reconcile_disco_reco(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
            Internal implementation of +=
            
        """
        disco_or = (self._status_or_before == -1) | (self._status_or == -1)
        disco_ex = (self._status_ex_before == -1) | (self._status_ex == -1)
        disco_now = (
            disco_or | disco_ex
        )  # a powerline is disconnected if at least one of its extremity is
        # added
        reco_or = (self._status_or_before == -1) & (self._status_or >= 1)
        reco_ex = (self._status_or_before == -1) & (self._status_ex >= 1)
        reco_now = reco_or | reco_ex
        # Set nothing
        set_now = np.zeros_like(self._status_or)
        # Force some disconnections
        set_now[disco_now] = -1
        set_now[reco_now] = 1

        self.current_topo.set_status(
            set_now,
            self.line_or_pos_topo_vect,
            self.line_ex_pos_topo_vect,
            self.last_topo_registered,
        )
           
    def __iadd__(self, other : BaseAction) -> Self:
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
            This is called by the environment, do not alter.
            
        The goal of this function is to "fused" together all the different types
        of modifications handled by:
        
        - the Agent
        - the opponent
        - the time series (part of the environment)
        - the voltage controler
        
        It might be called multiple times per step.

        Parameters
        ----------
        other: :class:`grid2op.Action.BaseAction`

        Returns
        -------
        The updated state of `self` after the new action `other` has been added to it.
        
        """

        set_status = other._set_line_status
        switch_status = other._switch_line_status
        set_topo_vect = other._set_topo_vect
        switcth_topo_vect = other._change_bus_vect
        redispatching = other._redispatch
        storage_power = other._storage_power

        # I deal with injections
        # Ia set the injection
        if other._modif_inj:
            self._aux_iadd_inj(other._dict_inj)
            
        # Ib change the injection aka redispatching
        if other._modif_redispatch:
            self.prod_p.change_val(redispatching)

        # Ic storage unit
        if other._modif_storage:
            self.storage_power.set_val(storage_power)

        # II shunts
        if type(self).shunts_data_available:
            self._aux_iadd_shunt(other)
            
        # III line status
        # this need to be done BEFORE the topology, as a connected powerline will be connected to their old bus.
        # regardless if the status is changed in the action or not.
        if other._modif_change_status:
            self.current_topo.change_status(
                switch_status,
                self.line_or_pos_topo_vect,
                self.line_ex_pos_topo_vect,
                self.last_topo_registered,
            )
        if other._modif_set_status:
            self.current_topo.set_status(
                set_status,
                self.line_or_pos_topo_vect,
                self.line_ex_pos_topo_vect,
                self.last_topo_registered,
            )

        # if other._modif_change_status or other._modif_set_status:
        (
            self._status_or_before[:],
            self._status_ex_before[:],
        ) = self.current_topo.get_line_status(
            self.line_or_pos_topo_vect, self.line_ex_pos_topo_vect
        )

        # IV topo
        if other._modif_change_bus:
            self.current_topo.change_val(switcth_topo_vect)
        if other._modif_set_bus:
            self.current_topo.set_val(set_topo_vect)

        # V Force disconnected status
        # of disconnected powerlines extremities
        self._status_or[:], self._status_ex[:] = self.current_topo.get_line_status(
            self.line_or_pos_topo_vect, self.line_ex_pos_topo_vect
        )

        # At least one disconnected extremity
        if other._modif_change_bus or other._modif_set_bus:
            self._aux_iadd_reconcile_disco_reco()
        return self

    def _assign_0_to_disco_el(self) -> None:
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
            This is handled by the environment, do not alter.
        
        Do not consider disconnected elements are modified for there active / reactive / voltage values
        """
        cls = type(self)
        gen_changed = self.current_topo.changed[cls.gen_pos_topo_vect]
        gen_bus = self.current_topo.values[cls.gen_pos_topo_vect]
        self.prod_p.force_unchanged(gen_changed, gen_bus)
        self.prod_v.force_unchanged(gen_changed, gen_bus)
        
        load_changed = self.current_topo.changed[cls.load_pos_topo_vect]
        load_bus = self.current_topo.values[cls.load_pos_topo_vect]
        self.load_p.force_unchanged(load_changed, load_bus)
        self.load_q.force_unchanged(load_changed, load_bus)
        
        sto_changed = self.current_topo.changed[cls.storage_pos_topo_vect]
        sto_bus = self.current_topo.values[cls.storage_pos_topo_vect]
        self.storage_power.force_unchanged(sto_changed, sto_bus)
        
    def __call__(self) -> Tuple[np.ndarray,
                                Tuple[ValueStore, ValueStore, ValueStore, ValueStore, ValueStore],
                                ValueStore,
                                Union[Tuple[ValueStore, ValueStore, ValueStore], None]]:
        """
        This function should be called at the top of the :func:`grid2op.Backend.Backend.apply_action`
        implementation when you decide to code a new backend.
        
        It processes the state of the backend into a form "easy to use" in the `apply_action` method.
        
        .. danger::
            It is mandatory to call it, otherwise some features might not work.
            
        .. warning:: /!\\\\ Do not alter / modify / change / override this implementation /!\\\\
            
        Examples
        -----------
        
        A typical implementation of `apply_action` will start with:
        
        .. code-block:: python
        
            def apply_action(self, backendAction: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:
                if backendAction is None:
                    return
            
                (
                    active_bus,
                    (prod_p, prod_v, load_p, load_q, storage),
                    topo__,
                    shunts__,
                ) = backendAction()
                
                # process the backend action by updating `self._grid`
        
        Returns
        -------
        
        - `active_bus`: matrix with `type(self).n_sub` rows and `type(self).n_busbar_per_bus` columns. Each elements
          represents a busbars of the grid. ``False`` indicates that nothing is connected to this busbar and ``True``
          means that at least an element is connected to this busbar
        - (prod_p, prod_v, load_p, load_q, storage): 5-tuple of Iterable to set the new values of generators, loads and storage units.
        - topo: iterable representing the target topology (in local bus, elements are ordered with their 
          position in the `topo_vect` vector)
        
        """
        self._assign_0_to_disco_el()
        injections = (
            self.prod_p,
            self.prod_v,
            self.load_p,
            self.load_q,
            self.storage_power,
        )
        topo = self.current_topo
        shunts = None
        if type(self).shunts_data_available:
            shunts = self.shunt_p, self.shunt_q, self.shunt_bus
        self._get_active_bus()
        return self.activated_bus, injections, topo, shunts
    
    def get_loads_bus(self) -> ValueStore:
        """
        This function might be called in the implementation of :func:`grid2op.Backend.Backend.apply_action`.
        
        It is relevant when your solver expose API by "element types" for example
        you get the possibility to set and access all loads at once, all generators at
        once and your solver can easily move element from different busbar in a given
        substation.
        
        This corresponds to option 2a described (shortly) in :class:`_BackendAction`.
        
        In this setting, this function will give you the "local bus" id for each loads that 
        have been changed by the agent / time series / voltage controlers / opponent / etc.
            
        .. warning:: /!\\\\ Do not alter / modify / change / override this implementation /!\\\\
            
        .. seealso::
            The other related functions:
            
            - :func:`_BackendAction.get_loads_bus`
            - :func:`_BackendAction.get_gens_bus`
            - :func:`_BackendAction.get_lines_or_bus`
            - :func:`_BackendAction.get_lines_ex_bus`
            - :func:`_BackendAction.get_storages_bus`
            
        Examples
        -----------
        
        A typical use of `get_loads_bus` in `apply_action` is:
        
        .. code-block:: python
        
            def apply_action(self, backendAction: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:
                if backendAction is None:
                    return
            
                (
                    active_bus,
                    (prod_p, prod_v, load_p, load_q, storage),
                    _,
                    shunts__,
                ) = backendAction()
                
                # process the backend action by updating `self._grid`
                ...
                
                # now process the topology (called option 2.a in the doc):
                
                lines_or_bus = backendAction.get_lines_or_bus()
                for line_id, new_bus in lines_or_bus:
                    # connect "or" side of "line_id" to (local) bus `new_bus` in self._grid
                    self._grid.something(...)
                    # or
                    self._grid.something = ...
                    
                lines_ex_bus = backendAction.get_lines_ex_bus()
                for line_id, new_bus in lines_ex_bus:
                    # connect "ex" side of "line_id" to (local) bus `new_bus` in self._grid
                    self._grid.something(...)
                    # or
                    self._grid.something = ...
                    
                storages_bus = backendAction.get_storages_bus()
                for el_id, new_bus in storages_bus:
                    # connect storage id `el_id` to (local) bus `new_bus` in self._grid
                    self._grid.something(...)
                    # or
                    self._grid.something = ...
                    
                gens_bus = backendAction.get_gens_bus()
                for el_id, new_bus in gens_bus:
                    # connect generator id `el_id` to (local) bus `new_bus` in self._grid
                    self._grid.something(...)
                    # or
                    self._grid.something = ...
                    
                loads_bus = backendAction.get_loads_bus()
                for el_id, new_bus in loads_bus:
                    # connect generator id `el_id` to (local) bus `new_bus` in self._grid
                    self._grid.something(...)
                    # or
                    self._grid.something = ...
                    
                # continue implementation of `apply_action`
            
        """
        if self._loads_bus is None:
            self._loads_bus = ValueStore(type(self).n_load, dtype=dt_int)
        self._loads_bus.copy_from_index(self.current_topo, type(self).load_pos_topo_vect)
        return self._loads_bus

    def _aux_to_global(self, value_store, to_subid) -> ValueStore:
        value_store = copy.deepcopy(value_store)
        value_store.values = type(self).local_bus_to_global(value_store.values, to_subid)
        return value_store
        
    def get_loads_bus_global(self) -> ValueStore:
        """
        This function might be called in the implementation of :func:`grid2op.Backend.Backend.apply_action`.
        
        It is relevant when your solver expose API by "element types" for example
        you get the possibility to set and access all loads at once, all generators at
        once AND you can easily switch element from one "busbars" to another in 
        the whole grid handled by your solver.
        
        This corresponds to situation 2b described in :class:`_BackendAction`.
        
        In this setting, this function will give you the "local bus" id for each loads that 
        have been changed by the agent / time series / voltage controlers / opponent / etc.
            
        .. warning:: /!\\\\ Do not alter / modify / change / override this implementation /!\\\\
            
        .. seealso::
            The other related functions:
            
            - :func:`_BackendAction.get_loads_bus_global`
            - :func:`_BackendAction.get_gens_bus_global`
            - :func:`_BackendAction.get_lines_or_bus_global`
            - :func:`_BackendAction.get_lines_ex_bus_global`
            - :func:`_BackendAction.get_storages_bus_global`
            
        Examples
        -----------
        
        A typical use of `get_loads_bus_global` in `apply_action` is:
        
        .. code-block:: python
        
            def apply_action(self, backendAction: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:
                if backendAction is None:
                    return
            
                (
                    active_bus,
                    (prod_p, prod_v, load_p, load_q, storage),
                    _,
                    shunts__,
                ) = backendAction()
                
                # process the backend action by updating `self._grid`
                ...
                
                # now process the topology (called option 2.a in the doc):
                
                lines_or_bus = backendAction.get_lines_or_bus_global()
                for line_id, new_bus in lines_or_bus:
                    # connect "or" side of "line_id" to (global) bus `new_bus` in self._grid
                    self._grid.something(...)
                    # or
                    self._grid.something = ...
                    
                lines_ex_bus = backendAction.get_lines_ex_bus_global()
                for line_id, new_bus in lines_ex_bus:
                    # connect "ex" side of "line_id" to (global) bus `new_bus` in self._grid
                    self._grid.something(...)
                    # or
                    self._grid.something = ...
                    
                storages_bus = backendAction.get_storages_bus_global()
                for el_id, new_bus in storages_bus:
                    # connect storage id `el_id` to (global) bus `new_bus` in self._grid
                    self._grid.something(...)
                    # or
                    self._grid.something = ...
                    
                gens_bus = backendAction.get_gens_bus_global()
                for el_id, new_bus in gens_bus:
                    # connect generator id `el_id` to (global) bus `new_bus` in self._grid
                    self._grid.something(...)
                    # or
                    self._grid.something = ...
                    
                loads_bus = backendAction.get_loads_bus_global()
                for el_id, new_bus in loads_bus:
                    # connect generator id `el_id` to (global) bus `new_bus` in self._grid
                    self._grid.something(...)
                    # or
                    self._grid.something = ...
                    
                # continue implementation of `apply_action`
            
        """
        tmp_ = self.get_loads_bus()
        return self._aux_to_global(tmp_, type(self).load_to_subid)
    
    def get_gens_bus(self) -> ValueStore:
        """
        This function might be called in the implementation of :func:`grid2op.Backend.Backend.apply_action`.
        
        It is relevant when your solver expose API by "element types" for example
        you get the possibility to set and access all loads at once, all generators at
        once and your solver can easily move element from different busbar in a given
        substation.
        
        This corresponds to option 2a described (shortly) in :class:`_BackendAction`.
        
        In this setting, this function will give you the "local bus" id for each generators that 
        have been changed by the agent / time series / voltage controlers / opponent / etc.
            
        .. warning:: /!\\\\ Do not alter / modify / change / override this implementation /!\\\\
            
        .. seealso::
            The other related functions:
            
            - :func:`_BackendAction.get_loads_bus`
            - :func:`_BackendAction.get_gens_bus`
            - :func:`_BackendAction.get_lines_or_bus`
            - :func:`_BackendAction.get_lines_ex_bus`
            - :func:`_BackendAction.get_storages_bus`
        
        Examples
        ---------
        
        Some examples are given in the documentation of :func:`_BackendAction.get_loads_bus`
        
        """
        if self._gens_bus is None:
            self._gens_bus = ValueStore(type(self).n_gen, dtype=dt_int)
        self._gens_bus.copy_from_index(self.current_topo, type(self).gen_pos_topo_vect)
        return self._gens_bus

    def get_gens_bus_global(self) -> ValueStore:
        """
        This function might be called in the implementation of :func:`grid2op.Backend.Backend.apply_action`.
        
        It is relevant when your solver expose API by "element types" for example
        you get the possibility to set and access all loads at once, all generators at
        once AND you can easily switch element from one "busbars" to another in 
        the whole grid handled by your solver.
        
        This corresponds to situation 2b described in :class:`_BackendAction`.
        
        In this setting, this function will give you the "local bus" id for each loads that 
        have been changed by the agent / time series / voltage controlers / opponent / etc.
            
        .. warning:: /!\\\\ Do not alter / modify / change / override this implementation /!\\\\
            
        .. seealso::
            The other related functions:
            
            - :func:`_BackendAction.get_loads_bus_global`
            - :func:`_BackendAction.get_gens_bus_global`
            - :func:`_BackendAction.get_lines_or_bus_global`
            - :func:`_BackendAction.get_lines_ex_bus_global`
            - :func:`_BackendAction.get_storages_bus_global`
        
        Examples
        ---------
        
        Some examples are given in the documentation of :func:`_BackendAction.get_loads_bus_global`
        """            
        
        tmp_ = copy.deepcopy(self.get_gens_bus())
        return self._aux_to_global(tmp_, type(self).gen_to_subid)
    
    def get_lines_or_bus(self) -> ValueStore:
        """
        This function might be called in the implementation of :func:`grid2op.Backend.Backend.apply_action`.
        
        It is relevant when your solver expose API by "element types" for example
        you get the possibility to set and access all loads at once, all generators at
        once and your solver can easily move element from different busbar in a given
        substation.
        
        This corresponds to option 2a described (shortly) in :class:`_BackendAction`.
        
        In this setting, this function will give you the "local bus" id for each line (or side) that 
        have been changed by the agent / time series / voltage controlers / opponent / etc.
            
        .. warning:: /!\\\\ Do not alter / modify / change / override this implementation /!\\\\
            
        .. seealso::
            The other related functions:
            
            - :func:`_BackendAction.get_loads_bus`
            - :func:`_BackendAction.get_gens_bus`
            - :func:`_BackendAction.get_lines_or_bus`
            - :func:`_BackendAction.get_lines_ex_bus`
            - :func:`_BackendAction.get_storages_bus`
        
        Examples
        ---------
        
        Some examples are given in the documentation of :func:`_BackendAction.get_loads_bus`
        
        """
        if self._lines_or_bus is None:
            self._lines_or_bus = ValueStore(type(self).n_line, dtype=dt_int)
        self._lines_or_bus.copy_from_index(
            self.current_topo, type(self).line_or_pos_topo_vect
        )
        return self._lines_or_bus
    
    def get_lines_or_bus_global(self) -> ValueStore:
        """
        This function might be called in the implementation of :func:`grid2op.Backend.Backend.apply_action`.
        
        It is relevant when your solver expose API by "element types" for example
        you get the possibility to set and access all loads at once, all generators at
        once AND you can easily switch element from one "busbars" to another in 
        the whole grid handled by your solver.
        
        This corresponds to situation 2b described in :class:`_BackendAction`.
        
        In this setting, this function will give you the "local bus" id for each loads that 
        have been changed by the agent / time series / voltage controlers / opponent / etc.
            
        .. warning:: /!\\\\ Do not alter / modify / change / override this implementation /!\\\\
            
        .. seealso::
            The other related functions:
            
            - :func:`_BackendAction.get_loads_bus_global`
            - :func:`_BackendAction.get_gens_bus_global`
            - :func:`_BackendAction.get_lines_or_bus_global`
            - :func:`_BackendAction.get_lines_ex_bus_global`
            - :func:`_BackendAction.get_storages_bus_global`
        
        Examples
        ---------
        
        Some examples are given in the documentation of :func:`_BackendAction.get_loads_bus_global`
        """  
        tmp_ = self.get_lines_or_bus()
        return self._aux_to_global(tmp_, type(self).line_or_to_subid)

    def get_lines_ex_bus(self) -> ValueStore:
        """
        This function might be called in the implementation of :func:`grid2op.Backend.Backend.apply_action`.
        
        It is relevant when your solver expose API by "element types" for example
        you get the possibility to set and access all loads at once, all generators at
        once and your solver can easily move element from different busbar in a given
        substation.
        
        This corresponds to option 2a described (shortly) in :class:`_BackendAction`.
        
        In this setting, this function will give you the "local bus" id for each line (ex side) that 
        have been changed by the agent / time series / voltage controlers / opponent / etc.
            
        .. warning:: /!\\\\ Do not alter / modify / change / override this implementation /!\\\\
            
        .. seealso::
            The other related functions:
            
            - :func:`_BackendAction.get_loads_bus`
            - :func:`_BackendAction.get_gens_bus`
            - :func:`_BackendAction.get_lines_or_bus`
            - :func:`_BackendAction.get_lines_ex_bus`
            - :func:`_BackendAction.get_storages_bus`
        
        Examples
        ---------
        
        Some examples are given in the documentation of :func:`_BackendAction.get_loads_bus`
        
        """
        if self._lines_ex_bus is None:
            self._lines_ex_bus = ValueStore(type(self).n_line, dtype=dt_int)
        self._lines_ex_bus.copy_from_index(
            self.current_topo, type(self).line_ex_pos_topo_vect
        )
        return self._lines_ex_bus
    
    def get_lines_ex_bus_global(self) -> ValueStore:
        """
        This function might be called in the implementation of :func:`grid2op.Backend.Backend.apply_action`.
        
        It is relevant when your solver expose API by "element types" for example
        you get the possibility to set and access all loads at once, all generators at
        once AND you can easily switch element from one "busbars" to another in 
        the whole grid handled by your solver.
        
        This corresponds to situation 2b described in :class:`_BackendAction`.
        
        In this setting, this function will give you the "local bus" id for each loads that 
        have been changed by the agent / time series / voltage controlers / opponent / etc.
            
        .. warning:: /!\\\\ Do not alter / modify / change / override this implementation /!\\\\
            
        .. seealso::
            The other related functions:
            
            - :func:`_BackendAction.get_loads_bus_global`
            - :func:`_BackendAction.get_gens_bus_global`
            - :func:`_BackendAction.get_lines_or_bus_global`
            - :func:`_BackendAction.get_lines_ex_bus_global`
            - :func:`_BackendAction.get_storages_bus_global`
        
        Examples
        ---------
        
        Some examples are given in the documentation of :func:`_BackendAction.get_loads_bus_global`
        """  
        tmp_ = self.get_lines_ex_bus()
        return self._aux_to_global(tmp_, type(self).line_ex_to_subid)

    def get_storages_bus(self) -> ValueStore:
        """
        This function might be called in the implementation of :func:`grid2op.Backend.Backend.apply_action`.
        
        It is relevant when your solver expose API by "element types" for example
        you get the possibility to set and access all loads at once, all generators at
        once and your solver can easily move element from different busbar in a given
        substation.
        
        This corresponds to option 2a described (shortly) in :class:`_BackendAction`.
        
        In this setting, this function will give you the "local bus" id for each storage that 
        have been changed by the agent / time series / voltage controlers / opponent / etc.
            
        .. warning:: /!\\\\ Do not alter / modify / change / override this implementation /!\\\\
            
        .. seealso::
            The other related functions:
            
            - :func:`_BackendAction.get_loads_bus`
            - :func:`_BackendAction.get_gens_bus`
            - :func:`_BackendAction.get_lines_or_bus`
            - :func:`_BackendAction.get_lines_ex_bus`
            - :func:`_BackendAction.get_storages_bus`
        
        Examples
        ---------
        
        Some examples are given in the documentation of :func:`_BackendAction.get_loads_bus`
        
        """
        if self._storage_bus is None:
            self._storage_bus = ValueStore(type(self).n_storage, dtype=dt_int)
        self._storage_bus.copy_from_index(self.current_topo, type(self).storage_pos_topo_vect)
        return self._storage_bus
    
    def get_storages_bus_global(self) -> ValueStore:
        """
        This function might be called in the implementation of :func:`grid2op.Backend.Backend.apply_action`.
        
        It is relevant when your solver expose API by "element types" for example
        you get the possibility to set and access all loads at once, all generators at
        once AND you can easily switch element from one "busbars" to another in 
        the whole grid handled by your solver.
        
        This corresponds to situation 2b described in :class:`_BackendAction`.
        
        In this setting, this function will give you the "local bus" id for each loads that 
        have been changed by the agent / time series / voltage controlers / opponent / etc.
            
        .. warning:: /!\\\\ Do not alter / modify / change / override this implementation /!\\\\
            
        .. seealso::
            The other related functions:
            
            - :func:`_BackendAction.get_loads_bus_global`
            - :func:`_BackendAction.get_gens_bus_global`
            - :func:`_BackendAction.get_lines_or_bus_global`
            - :func:`_BackendAction.get_lines_ex_bus_global`
            - :func:`_BackendAction.get_storages_bus_global`
        
        Examples
        ---------
        
        Some examples are given in the documentation of :func:`_BackendAction.get_loads_bus_global`
        """  
        tmp_ = self.get_storages_bus()
        return self._aux_to_global(tmp_, type(self).storage_to_subid)
    
    def get_shunts_bus_global(self) -> ValueStore:
        """
        This function might be called in the implementation of :func:`grid2op.Backend.Backend.apply_action`.
        
        It is relevant when your solver expose API by "element types" for example
        you get the possibility to set and access all loads at once, all generators at
        once AND you can easily switch element from one "busbars" to another in 
        the whole grid handled by your solver.
        
        This corresponds to situation 2b described in :class:`_BackendAction`.
        
        In this setting, this function will give you the "local bus" id for each loads that 
        have been changed by the agent / time series / voltage controlers / opponent / etc.
            
        .. warning:: /!\\\\ Do not alter / modify / change / override this implementation /!\\\\
            
        .. seealso::
            The other related functions:
            
            - :func:`_BackendAction.get_loads_bus_global`
            - :func:`_BackendAction.get_gens_bus_global`
            - :func:`_BackendAction.get_lines_or_bus_global`
            - :func:`_BackendAction.get_lines_ex_bus_global`
            - :func:`_BackendAction.get_storages_bus_global`
        
        Examples
        ---------
        
        Some examples are given in the documentation of :func:`_BackendAction.get_loads_bus_global`
        """  
        tmp_ = self.shunt_bus
        return self._aux_to_global(tmp_, type(self).shunt_to_subid)

    def _get_active_bus(self) -> None:
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        """
        self.activated_bus[:, :] = False
        tmp = self.current_topo.values - 1
        is_el_conn = tmp >= 0
        self.activated_bus[self.big_topo_to_subid[is_el_conn], tmp[is_el_conn]] = True
        if type(self).shunts_data_available:
            is_el_conn = self.current_shunt_bus.values >= 0
            tmp = self.current_shunt_bus.values - 1
            self.activated_bus[type(self).shunt_to_subid[is_el_conn], tmp[is_el_conn]] = True

    def update_state(self, powerline_disconnected) -> None:
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
            This is handled by the environment !

        Update the internal state. Should be called after the cascading failures.

        """
        if (powerline_disconnected >= 0).any():
            arr_ = np.zeros(powerline_disconnected.shape, dtype=dt_int)
            arr_[powerline_disconnected >= 0] = -1
            self.current_topo.set_status(
                arr_,
                self.line_or_pos_topo_vect,
                self.line_ex_pos_topo_vect,
                self.last_topo_registered,
            )
        self.last_topo_registered.update_connected(self.current_topo)
        self.current_topo.reset()
