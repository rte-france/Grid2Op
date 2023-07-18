# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np

from grid2op.dtypes import dt_int, dt_bool, dt_float
from grid2op.Space import GridObjects


# TODO see if it can be done in c++ easily
class ValueStore:
    """
    INTERNAL USE ONLY

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    """

    def __init__(self, size, dtype):
        ## TODO at the init it's mandatory to have everything at "1" here
        # if topo is not "fully connected" it will not work
        self.values = np.empty(size, dtype=dtype)
        self.changed = np.full(size, dtype=dt_bool, fill_value=False)
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
        changed_ = newvals != 0.0
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
        

class _BackendAction(GridObjects):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Internal class, use at your own risk.

    This class "digest" the players / environment / opponent / voltage controlers "action",
    and transform it to setpoint for the backend.
    """

    def __init__(self):
        GridObjects.__init__(self)
        # last connected registered
        self.last_topo_registered = ValueStore(self.dim_topo, dtype=dt_int)

        # topo at time t
        self.current_topo = ValueStore(self.dim_topo, dtype=dt_int)
        
        # by default everything is on busbar 1
        self.last_topo_registered.values[:] = 1
        self.current_topo.values[:] = 1  

        # injection at time t
        self.prod_p = ValueStore(self.n_gen, dtype=dt_float)
        self.prod_v = ValueStore(self.n_gen, dtype=dt_float)
        self.load_p = ValueStore(self.n_load, dtype=dt_float)
        self.load_q = ValueStore(self.n_load, dtype=dt_float)
        self.storage_power = ValueStore(self.n_storage, dtype=dt_float)

        self.activated_bus = np.full((self.n_sub, 2), dtype=dt_bool, fill_value=False)
        self.big_topo_to_subid = np.repeat(
            list(range(self.n_sub)), repeats=self.sub_info
        )

        # shunts
        if self.shunts_data_available:
            self.shunt_p = ValueStore(self.n_shunt, dtype=dt_float)
            self.shunt_q = ValueStore(self.n_shunt, dtype=dt_float)
            self.shunt_bus = ValueStore(self.n_shunt, dtype=dt_int)

        self._status_or_before = np.ones(self.n_line, dtype=dt_int)
        self._status_ex_before = np.ones(self.n_line, dtype=dt_int)
        self._status_or = np.ones(self.n_line, dtype=dt_int)
        self._status_ex = np.ones(self.n_line, dtype=dt_int)

        self._loads_bus = None
        self._gens_bus = None
        self._lines_or_bus = None
        self._lines_ex_bus = None
        self._storage_bus = None

    def __deepcopy__(self, memodict={}):
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
        if self.shunts_data_available:
            res.shunt_p.copy(self.shunt_p)
            res.shunt_q.copy(self.shunt_q)
            res.shunt_bus.copy(self.shunt_bus)

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

    def __copy__(self):
        res = self.__deepcopy__()  # nothing less to do
        return res

    def reorder(self, no_load, no_gen, no_topo, no_storage, no_shunt):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        reorder the element modified, this is use when converting backends only and should not be use
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

        if self.shunts_data_available:
            self.shunt_p.reorder(no_shunt)
            self.shunt_q.reorder(no_shunt)
            self.shunt_bus.reorder(no_shunt)

    def reset(self):
        # last topo
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
        if self.shunts_data_available:
            self.shunt_p.reset()
            self.shunt_q.reset()
            self.shunt_bus.reset()

    def all_changed(self):
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
        self.prod_p.change_val(new_redispatching)

    def __iadd__(self, other):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        other: a grid2op action standard

        Parameters
        ----------
        other: :class:`grid2op.Action.BaseAction.BaseAction`

        Returns
        -------

        """

        dict_injection = other._dict_inj
        set_status = other._set_line_status
        switch_status = other._switch_line_status
        set_topo_vect = other._set_topo_vect
        switcth_topo_vect = other._change_bus_vect
        redispatching = other._redispatch
        storage_power = other._storage_power

        # I deal with injections
        # Ia set the injection
        if other._modif_inj:
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

        # Ib change the injection aka redispatching
        if other._modif_redispatch:
            self.prod_p.change_val(redispatching)

        # Ic storage unit
        if other._modif_storage:
            self.storage_power.set_val(storage_power)

        # II shunts
        if self.shunts_data_available:
            shunts = {}
            if other.shunts_data_available:
                shunts["shunt_p"] = other.shunt_p
                shunts["shunt_q"] = other.shunt_q
                shunts["shunt_bus"] = other.shunt_bus

            arr_ = shunts["shunt_p"]
            self.shunt_p.set_val(arr_)
            arr_ = shunts["shunt_q"]
            self.shunt_q.set_val(arr_)
            arr_ = shunts["shunt_bus"]
            self.shunt_bus.set_val(arr_)

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

        return self

    def _assign_0_to_disco_el(self):
        """do not consider disconnected elements are modified for there active / reactive / voltage values"""
        gen_changed = self.current_topo.changed[type(self).gen_pos_topo_vect]
        gen_bus = self.current_topo.values[type(self).gen_pos_topo_vect]
        self.prod_p.force_unchanged(gen_changed, gen_bus)
        self.prod_v.force_unchanged(gen_changed, gen_bus)
        
        load_changed = self.current_topo.changed[type(self).load_pos_topo_vect]
        load_bus = self.current_topo.values[type(self).load_pos_topo_vect]
        self.load_p.force_unchanged(load_changed, load_bus)
        self.load_q.force_unchanged(load_changed, load_bus)
        
        sto_changed = self.current_topo.changed[type(self).storage_pos_topo_vect]
        sto_bus = self.current_topo.values[type(self).storage_pos_topo_vect]
        self.storage_power.force_unchanged(sto_changed, sto_bus)
        
    def __call__(self):
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
        if self.shunts_data_available:
            shunts = self.shunt_p, self.shunt_q, self.shunt_bus
        self._get_active_bus()
        return self.activated_bus, injections, topo, shunts
    
    def get_loads_bus(self):
        if self._loads_bus is None:
            self._loads_bus = ValueStore(self.n_load, dtype=dt_int)
        self._loads_bus.copy_from_index(self.current_topo, self.load_pos_topo_vect)
        return self._loads_bus

    def _aux_to_global(self, value_store, to_subid):
        value_store = copy.deepcopy(value_store)
        value_store.values = type(self).local_bus_to_global(value_store.values, to_subid)
        return value_store
        
    def get_loads_bus_global(self):
        tmp_ = self.get_loads_bus()
        return self._aux_to_global(tmp_, self.load_to_subid)
    
    def get_gens_bus(self):
        if self._gens_bus is None:
            self._gens_bus = ValueStore(self.n_gen, dtype=dt_int)
        self._gens_bus.copy_from_index(self.current_topo, self.gen_pos_topo_vect)
        return self._gens_bus

    def get_gens_bus_global(self):
        tmp_ = copy.deepcopy(self.get_gens_bus())
        return self._aux_to_global(tmp_, self.gen_to_subid)
    
    def get_lines_or_bus(self):
        if self._lines_or_bus is None:
            self._lines_or_bus = ValueStore(self.n_line, dtype=dt_int)
        self._lines_or_bus.copy_from_index(
            self.current_topo, self.line_or_pos_topo_vect
        )
        return self._lines_or_bus
    
    def get_lines_or_bus_global(self):
        tmp_ = self.get_lines_or_bus()
        return self._aux_to_global(tmp_, self.line_or_to_subid)

    def get_lines_ex_bus(self):
        if self._lines_ex_bus is None:
            self._lines_ex_bus = ValueStore(self.n_line, dtype=dt_int)
        self._lines_ex_bus.copy_from_index(
            self.current_topo, self.line_ex_pos_topo_vect
        )
        return self._lines_ex_bus
    
    def get_lines_ex_bus_global(self):
        tmp_ = self.get_lines_ex_bus()
        return self._aux_to_global(tmp_, self.line_ex_to_subid)

    def get_storages_bus(self):
        if self._storage_bus is None:
            self._storage_bus = ValueStore(self.n_storage, dtype=dt_int)
        self._storage_bus.copy_from_index(self.current_topo, self.storage_pos_topo_vect)
        return self._storage_bus
    
    def get_storages_bus_global(self):
        tmp_ = self.get_storages_bus()
        return self._aux_to_global(tmp_, self.storage_to_subid)

    def _get_active_bus(self):
        self.activated_bus[:, :] = False
        tmp = self.current_topo.values - 1
        is_el_conn = tmp >= 0
        self.activated_bus[self.big_topo_to_subid[is_el_conn], tmp[is_el_conn]] = True

    def update_state(self, powerline_disconnected):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Update the internal state. Should be called after the cascading failures

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
