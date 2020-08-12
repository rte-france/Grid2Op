# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import copy
from grid2op.Backend import Backend
from grid2op.Exceptions import Grid2OpException

ERROR_NB_ELEMENTS = "Impossible to make a BackendConverter with backends having different number of {}."
ERROR_ELEMENT_CONNECTED = "No {0} connected at substation {1} for the target backend while a {0} is connected " \
                          "at substation {2} for the source backend"


class BackendConverter(Backend):
    """
    Convert two instance of backend to "align" them.

    This means that grid2op will behave exactly as is the "source backend" class is used everywhere, but
    the powerflow computation will be carried out by the "target backend".

    This means that from grid2op point of view, and from the agent point of view, line will be the order given
    by "source backend", load will be in the order of "source backend", topology will be given with the
    one from "source backend" etc. etc.

    Be careful, the BackendAction will also need to be transformed. Backend action is given with the order
    of the "source backend" and will have to be modified when transmitted to the "target backend".

    On the other end, no powerflow at all (except if some powerflows are performed at the initialization) will
    be computed using the source backend, only the target backend is relevant for the powerflow computations.
    
    Note that these backend need to access the grid description file from both "source backend" and "target backend" 
    class. The underlying grid must be the same.
    """
    def __init__(self,
                 source_backend_class,
                 target_backend_class,
                 target_backend_grid_path=None,
                 detailed_infos_for_cascading_failures=False):
        Backend.__init__(self, detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        difcf = detailed_infos_for_cascading_failures
        self.source_backend = source_backend_class(detailed_infos_for_cascading_failures=difcf)
        self.target_backend = target_backend_class(detailed_infos_for_cascading_failures=difcf)
        self.target_backend_grid_path = target_backend_grid_path

        # TODO
        self._sub_tg2sr = None
        self._sub_sr2tg = None
        self._line_tg2sr = None  # if tmp is from the target backend, then tmp[self._line_tg2sr] is ordered according to the source backend
        self._line_sr2tg = None
        self._gen_tg2sr = None
        self._gen_sr2tg = None
        self._load_tg2sr = None
        self._load_sr2tg = None
        self._shunt_tg2sr = None
        self._shunt_tg2sr = None
        self._topo_tg2sr = None
        self._topo_sr2tg = None

    def load_grid(self, path=None, filename=None):
        self.source_backend.load_grid(path, filename)
        # shortcut to set all information related to the class, except the name of the environment
        self.__class__ = self.init_grid(self.source_backend)

        # and now i load the target backend
        if self.target_backend_grid_path is not None:
            self.target_backend.load_grid(path=self.target_backend_grid_path)
        else:
            # both source and target backend understands the same format
            self.target_backend.load_grid(path, filename)

        if self.n_sub != self.target_backend.n_sub:
            raise Grid2OpException(ERROR_NB_ELEMENTS.format("substations"))
        if self.n_gen != self.target_backend.n_gen:
            raise Grid2OpException(ERROR_NB_ELEMENTS.format("generators"))
        if self.n_load != self.target_backend.n_load:
            raise Grid2OpException(ERROR_NB_ELEMENTS.format("loads"))
        if self.n_line != self.target_backend.n_line:
            raise Grid2OpException(ERROR_NB_ELEMENTS.format("lines"))

        # and now init all the converting vectors
        # a) for substation
        ## automatic mode
        if np.all(sorted(self.source_backend.name_sub) == sorted(self.target_backend.name_sub)):
            self._sub_tg2sr = np.full(self.n_sub, fill_value=-1, dtype=np.int)
            self._sub_sr2tg = np.full(self.n_sub, fill_value=-1, dtype=np.int)
            for id_source, nm_source in enumerate(self.source_backend.name_sub):
                id_target = np.where(self.target_backend.name_sub == nm_source)[0]
                self._sub_tg2sr[id_source] = id_target
                self._sub_sr2tg[id_target] = id_source
        else:
            raise RuntimeError("Non automatic mode requries a mapping dict for sub that is not provided")

        # b) for load
        ## automatic mode
        self._load_tg2sr = np.full(self.n_load, fill_value=-1, dtype=np.int)
        self._load_sr2tg = np.full(self.n_load, fill_value=-1, dtype=np.int)
        nb_load_per_sub = np.zeros(self.n_sub, dtype=np.int)
        for id_source in range(self.source_backend.n_load):
            id_sub_source = self.source_backend.load_to_subid[id_source]
            id_sub_target = self._sub_sr2tg[id_sub_source]
            id_target = np.where(self.target_backend.load_to_subid == id_sub_target)[0]
            if id_target.shape[0] == 0:
                raise RuntimeError(ERROR_ELEMENT_CONNECTED.format("load", id_sub_target, id_sub_source))
            id_target = id_target[nb_load_per_sub[id_sub_target]]
            # TODO no no no use the "to_sub_pos" to compute that, and even better raise an error in this case
            # this means automatic is failing here !
            nb_load_per_sub[id_sub_target] += 1
            self._load_tg2sr[id_source] = id_target
            self._load_sr2tg[id_target] = id_source

        self._line_tg2sr = np.arange(self.n_line)
        self._line_sr2tg = np.arange(self.n_line)
        self._gen_tg2sr = np.arange(self.n_gen)
        self._gen_sr2tg = np.arange(self.n_gen)
        self._topo_tg2sr = np.arange(self.dim_topo)
        self._topo_sr2tg = np.arange(self.dim_topo)

        if self.shunts_data_available:
            self._shunt_tg2sr = np.arange(self.n_shunt)
            self._shunt_sr2tg = np.arange(self.n_shunt)
        self.set_thermal_limit(self.target_backend.thermal_limit_a[self._line_tg2sr])

    def assert_grid_correct(self):
        # this is done before a call to this function, by the environment
        self.source_backend.set_env_name(self.env_name)
        self.target_backend.set_env_name(self.env_name)

        # now i assert that
        self.source_backend.assert_grid_correct()
        self.target_backend.assert_grid_correct()
        # and this should be called after all the rest
        super().assert_grid_correct()

        assert np.all(self.target_backend.name_sub[self._sub_tg2sr] == self.source_backend.name_sub)
        assert np.all(self.source_backend.name_sub[self._sub_sr2tg] == self.target_backend.name_sub)

    def assert_grid_correct_after_powerflow(self):
        # we don't assert that `self.source_backend.assert_grid_correct_after_powerflow()`
        # because obviously no powerflow are run using the source backend.
        self.target_backend.assert_grid_correct_after_powerflow()
        super().assert_grid_correct_after_powerflow()

    def reset(self, grid_path, grid_filename=None):
        """
        Reload the power grid.
        For backwards compatibility this method calls `Backend.load_grid`.
        But it is encouraged to overload it in the subclasses.
        """
        self.target_backend.reset(grid_path, grid_filename=None)

    def close(self):
        self.source_backend.close()
        self.target_backend.close()

    def apply_action(self, action):
        # action is from the source backend
        action_target = self._transform_action(action)
        self.target_backend.apply_action(action_target)

    def runpf(self, is_dc=False):
        return self.target_backend.runpf(is_dc=is_dc)

    def copy(self):
        res = self
        res.target_backend_grid_path = copy.deepcopy(self.target_backend_grid_path)
        res.source_backend = res.source_backend.copy()
        res.target_backend = res.target_backend.copy()
        return res

    def save_file(self, full_path):
        self.target_backend.save_file(full_path)
        self.source_backend.save_file(full_path)

    def get_line_status(self):
        tmp = self.target_backend.get_line_status()
        return tmp[self._line_tg2sr]

    def get_line_flow(self):
        tmp = self.target_backend.get_line_flow()
        return tmp[self._line_tg2sr]

    def set_thermal_limit(self, limits):
        super().set_thermal_limit(limits=limits)
        self.target_backend.set_thermal_limit(limits=limits)
        self.source_backend.set_thermal_limit(limits=limits[self._line_sr2tg])

    def update_thermal_limit(self, env):
        # TODO
        # env has the powerline stored in the order of the source backend, but i need
        # to have them stored in the order of the target backend for such function
        pass

    def get_thermal_limit(self):
        tmp = self.target_backend.get_thermal_limit()
        return tmp[self._line_tg2sr]

    def get_topo_vect(self):
        tmp = self.target_backend.get_topo_vect()
        return tmp[self._topo_tg2sr]

    def generators_info(self):
        prod_p, prod_q, prod_v = self.target_backend.generators_info()
        return prod_p[self._gen_tg2sr], prod_q[self._gen_tg2sr], prod_q[self._gen_tg2sr]

    def loads_info(self):
        load_p, load_q, load_v = self.target_backend.loads_info()
        return load_p[self._load_tg2sr], load_q[self._load_tg2sr], load_v[self._load_tg2sr]

    def lines_or_info(self):
        p_, q_, v_, a_ = self.target_backend.lines_or_info()
        return p_[self._line_tg2sr], q_[self._line_tg2sr], v_[self._line_tg2sr], a_[self._line_tg2sr]

    def lines_ex_info(self):
        p_, q_, v_, a_ = self.target_backend.lines_ex_info()
        return p_[self._line_tg2sr], q_[self._line_tg2sr], v_[self._line_tg2sr], a_[self._line_tg2sr]

    def shunt_info(self):
        if self._shunt_tg2sr is not None:
            # shunts are supported by both source and target backend
            sh_p, sh_q, sh_v, sh_bus = self.target_backend.shunt_info()
            return sh_p[self._shunt_tg2sr], sh_q[self._shunt_tg2sr], sh_v[self._shunt_tg2sr], sh_bus[self._shunt_tg2sr]
        # shunt are not supported by either source or target backend
        return [], [], [], []

    def sub_from_bus_id(self, bus_id):
        # not supported because the bus_id is given into the source backend,
        # and i need to convert to to the target backend, not sure how to do that atm
        raise Grid2OpException("This backend doesn't allow to get the substation from the bus id.")

    def _disconnect_line(self, id):
        id_target = self._line_sr2tg[id]
        self.target_backend._disconnect_line(id_target)

    def _transform_action(self, source_action):
        # transform the source action into the target backend action
        # TODO
        return source_action

