# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.dtypes import dt_int, dt_bool, dt_float
from grid2op.Exceptions import *
from grid2op.Space import GridObjects


# TODO see if it can be done in c++ easily
class ValueStore:
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    """
    def __init__(self, size, dtype):
        ## TODO at the init it's mandatory to have everything at "1" here
        # if topo is not "fully connected" it will not work
        self.values = np.ones(size, dtype=dtype)
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
        changed_ = newvals != 0.
        self.changed[changed_] = True
        self.values[changed_] += newvals[changed_]

    def reset(self):
        self.changed[:] = False
        self.last_index = 0

    def change_status(self, switch, lineor_id, lineex_id, old_vect):
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
        # self.changed[new_order] = self.changed[:]
        # self.values[new_order] = self.values[:]


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

        # injection at time t
        self.prod_p = ValueStore(self.n_gen, dtype=dt_float)
        self.prod_v = ValueStore(self.n_gen, dtype=dt_float)
        self.load_p = ValueStore(self.n_load, dtype=dt_float)
        self.load_q = ValueStore(self.n_load, dtype=dt_float)

        self.activated_bus = np.full((self.n_sub, 2), dtype=dt_bool, fill_value=False)
        self.big_topo_to_subid = np.repeat(list(range(self.n_sub)), repeats=self.sub_info)

        # shunts
        if self.shunts_data_available:
            self.shunt_p = ValueStore(self.n_shunt, dtype=dt_float)
            self.shunt_q = ValueStore(self.n_shunt, dtype=dt_float)
            self.shunt_bus = ValueStore(self.n_shunt, dtype=dt_int)

        self._status_or_before = np.ones(self.n_line, dtype=dt_int)
        self._status_ex_before = np.ones(self.n_line, dtype=dt_int)
        self._status_or = np.ones(self.n_line, dtype=dt_int)
        self._status_ex = np.ones(self.n_line, dtype=dt_int)

    def reorder(self, no_load, no_gen, no_topo, no_shunt):
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
        other: grid2op.Action.BaseAction.BaseAction

        Returns
        -------

        """

        dict_injection = other._dict_inj
        set_status = other._set_line_status
        switch_status = other._switch_line_status
        set_topo_vect = other._set_topo_vect
        switcth_topo_vect = other._change_bus_vect
        redispatching = other._redispatch

        # I deal with injections
        # Ia set the injection
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
        self.prod_p.change_val(redispatching)

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
        self.current_topo.change_status(switch_status,
                                        self.line_or_pos_topo_vect,
                                        self.line_ex_pos_topo_vect,
                                        self.last_topo_registered)
        self.current_topo.set_status(set_status,
                                     self.line_or_pos_topo_vect,
                                     self.line_ex_pos_topo_vect,
                                     self.last_topo_registered)

        topo_before = self.current_topo.values
        self._status_or_before[:], \
        self._status_ex_before[:] = self.current_topo.get_line_status(
            self.line_or_pos_topo_vect,
            self.line_ex_pos_topo_vect)

        # IV topo
        self.current_topo.change_val(switcth_topo_vect)
        self.current_topo.set_val(set_topo_vect)

        # V Force disconnected status
        # of disconnected powerlines extremities
        self._status_or[:], \
        self._status_ex[:] = self.current_topo.get_line_status(self.line_or_pos_topo_vect,
                                                               self.line_ex_pos_topo_vect)

        # At least one disconnected extremity
        disco_or = (self._status_or_before == -1) | (self._status_or == -1)
        disco_ex = (self._status_ex_before == -1) | (self._status_ex == -1)
        disco_now = disco_or | disco_ex  # a powerline is disconnected if at least one of its extremity is
        #
        # added
        reco_or = (self._status_or_before == -1) & (self._status_or >= 1)
        reco_ex = (self._status_or_before == -1) & (self._status_ex >= 1)
        reco_now = reco_or | reco_ex
        #
        # Set nothing
        set_now = np.zeros_like(self._status_or)
        # # Force some disconnections
        set_now[disco_now] = -1
        set_now[reco_now] = 1

        self.current_topo.set_status(set_now,
                                     self.line_or_pos_topo_vect,
                                     self.line_ex_pos_topo_vect,
                                     self.last_topo_registered)

        return self

    def __call__(self):
        injections = self.prod_p, self.prod_v, self.load_p, self.load_q
        topo = self.current_topo
        shunts = None
        if self.shunts_data_available:
            shunts = self.shunt_p, self.shunt_q, self.shunt_bus
        self._get_active_bus()
        return self.activated_bus, injections, topo, shunts

    def _get_active_bus(self):
        self.activated_bus[:] = False
        tmp = self.current_topo.values-1
        self.activated_bus[self.big_topo_to_subid, tmp] = True

    def update_state(self, powerline_disconnected):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Update the internal state. Should be called after the cascading failures

        """
        arr_ = np.zeros(powerline_disconnected.shape, dtype=dt_int)
        arr_[powerline_disconnected] = -1
        self.current_topo.set_status(arr_,
                                     self.line_or_pos_topo_vect,
                                     self.line_ex_pos_topo_vect,
                                     self.last_topo_registered)
        self.last_topo_registered.update_connected(self.current_topo)
        self.current_topo.reset()
