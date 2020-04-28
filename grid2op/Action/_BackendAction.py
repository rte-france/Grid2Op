import numpy as np
import warnings

import pdb

from grid2op.dtypes import dt_int, dt_bool, dt_float
from grid2op.Exceptions import *
from grid2op.Space import GridObjects


# TODO see if it can be done in c++ easily
class ValueStore:
    def __init__(self, size, dtype):
        self.values = np.zeros(size, dtype=dtype)
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
        self.values[changed_] = newvals

    def _set_val_int(self, newvals):
        changed_ = newvals != 0
        self.changed[changed_] = True
        self.values[changed_] = newvals

    def _change_val_int(self, newvals):
        changed_ = newvals & self.values > 0
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
                self.values[id_or] = old_vect.values[id_or]
                self.values[id_ex] = old_vect.values[id_ex]

    def set_status(self, set, lineor_id, lineex_id):
        # TODO
    
    def __iter__(self):
        return self

    def __next__(self):
        while self.last_index < self.values.shape[0]:
            if self.changed[self.last_index]:
                return self.values[self.last_index]
            self.last_index += 1
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
        self.curent_topo = ValueStore(self.dim_topo, dtype=dt_int)

        # injection at time t
        self.prod_p = ValueStore(self.n_gen, dtype=dt_float)
        self.prod_v = ValueStore(self.n_gen, dtype=dt_float)
        self.load_p = ValueStore(self.n_load, dtype=dt_float)
        self.load_q = ValueStore(self.n_load, dtype=dt_float)

        # shunts
        if self.shunts_data_available:
            self.shunt_p = ValueStore(self.n_shunt, dtype=dt_float)
            self.shunt_q = ValueStore(self.n_shunt, dtype=dt_float)
            self.shunt_bus = ValueStore(self.n_shunt, dtype=dt_int)

    def reset(self):
        # last topo
        self.last_topo_registered.reset()

        # topo at time t
        self.curent_topo.reset()

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

        # deal with injections
        ## set the injection
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

        ## change the injection aka redispatching
        self.prod_p.set_val(redispatching)

        # shunts
        if self.shunts_data_available:
            arr_ = shunts["shunt_p"]
            self.shunt_p.set_val(arr_)
            arr_ = shunts["shunt_q"]
            self.shunt_q.set_val(arr_)
            arr_ = shunts["shunt_bus"]
            self.shunt_bus.set_val(arr_)

        # line status
        # TODO
        self.line_or_pos_topo_vect
        self.line_ex_pos_topo_vect

        # topo
        self.curent_topo.change_val(switcth_topo_vect)
        self.curent_topo.set_val(set_topo_vect)

    def __call__(self):
        injections = self.prod_p, self.prod_v, self.load_p, self.load_q
        topo = self.curent_topo
        shunts = None
        if self.shunts_data_available:
            shunts = self.shunt_p, self.shunt_q, self.shunt_bus

        return injections, topo, shunts

    def update_state(self, powerline_disconnected):
        # TODO update the last_topo_registered, and disconnect the proper powerlines