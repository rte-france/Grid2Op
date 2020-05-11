import numpy as np
import warnings

import pdb

from grid2op.dtypes import dt_int, dt_bool, dt_float
from grid2op.Exceptions import *
from grid2op.Space import GridObjects


# TODO see if it can be done in c++ easily
class ValueStore:
    def __init__(self, size, dtype):
        ## TODO at the init it's mandatory to have everything at "1" here
        # if topo is not "fully connected" it will not work
        self.values = np.ones(size, dtype=dtype)
        self.changed = np.full(size, dtype=dt_bool, fill_value=False)
        self.last_index = 0

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
        for i, el in enumerate(switch):
            if not el:
                continue
            id_or = lineor_id[i]
            id_ex = lineex_id[i]
            self.changed[id_or] = True
            self.changed[id_ex] = True
            me_or_bus = self.values[id_or]
            me_ex_bus = self.values[id_ex]
            if me_ex_bus > 0 or me_or_bus > 0:
                # powerline was connected, i disconnect it
                self.values[id_or] = -1
                self.values[id_ex] = -1
            else:
                # i need to reconnect it
                self.values[id_or] = old_vect[id_or]
                self.values[id_ex] = old_vect[id_ex]

    def set_status(self, set, lineor_id, lineex_id, old_vect):
        # TODO
        for i, el in enumerate(set):
            id_or = lineor_id[i]
            id_ex = lineex_id[i]
            if el == -1:
                if self.values[id_or] != -1:
                    self.values[id_or] = -1
                    self.changed[id_or] = True

                if self.values[id_ex] != -1:
                    self.values[id_ex] = -1
                    self.changed[id_ex] = True
            elif el == 1:
                if self.values[id_or] == -1:
                    self.values[id_or] = old_vect[id_or]
                    self.changed[id_or] = True

                if self.values[id_ex] == -1:
                    self.values[id_ex] = old_vect[id_ex]
                    self.changed[id_ex] = True

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


class _BackendAction(GridObjects):
    """
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

    def __iadd__(self, other):
        """
        other: a grid2op action standard

        Parameters
        ----------
        other

        Returns
        -------

        """
        dict_injection, set_status, switch_status, set_topo_vect, switcth_topo_vect, redispatching, shunts = other()
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
        if self.shunts_data_available and shunts:
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

        # IV topo
        self.current_topo.change_val(switcth_topo_vect)
        self.current_topo.set_val(set_topo_vect)

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
