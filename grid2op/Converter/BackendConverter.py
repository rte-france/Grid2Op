# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import copy
import warnings

from grid2op.dtypes import dt_float, dt_int
from grid2op.Backend import Backend
from grid2op.Exceptions import Grid2OpException, BackendError

ERROR_NB_ELEMENTS = (
    "Impossible to make a BackendConverter with backends having different number of {}."
)
ERROR_ELEMENT_CONNECTED = (
    "No {0} connected at substation {1} for the target backend while a {0} is connected "
    "at substation {2} for the source backend"
)
ERROR_INVALID_VECTOR = "invalid vector: some element are not found in either source or target"


class BackendConverter(Backend):
    """
    Convert two instance of backend to "align" them. This can be usefull if you have time series
    from a given "type of backend" (with certain names) but want to use another type 
    of backend to run the computation.
    
    Another usecase is if you have agents trained with certain backend type (so having loads, 
    generators etc. in certain order) but want to use a different backend to run the powerflow. You train
    your agent with a "simple and fast" powerflow but want to test them with a "more realistic" one.

    This means that grid2op will behave exactly as is the "source backend" class is used everywhere, but
    the powerflow computation will be carried out by the "target backend".
    
    # TODO: rename: "source backend" as "user backend" and "target backend" as "powerflow backend" or something
    like that.

    This means that from grid2op point of view, and from the agent point of view, line will be the order given
    by "source backend", load will be in the order of "source backend", topology will be given with the
    one from "source backend" etc. etc.

    Be careful, the BackendAction will also need to be transformed. Backend action is given with the order
    of the "source backend" and will have to be modified when transmitted to the "target backend".

    On the other end, no powerflow at all (except if some powerflows are performed at the initialization) will
    be computed using the source backend, only the target backend is relevant for the powerflow computations.

    Note that these backend need to access the grid description file from both "source backend" and "target backend"
    class. The underlying grid must be the same.

    # TODO: have a "mapping" from source name to target name in the constructor. If not provided it's the current
    behaviour (everything is automatic) but if present it maps the things according to this mapping. This mapping
    could be like `names_chronics_to_backend`
    
    Examples
    ---------
    Here is a (dummy and useless) example of how to use this class.

    .. code-block:: python

            import grid2op
            from grid2op.Converter import BackendConverter
            from grid2op.Backend import PandaPowerBackend
            from lightsim2grid import LightSimBackend
            backend = BackendConverter(source_backend_class=PandaPowerBackend,
                                       target_backend_class=LightSimBackend,
                                       target_backend_grid_path=None)

            # and now your environment behaves as if PandaPowerBackend did the computation (same load order, same
            generator order etc.) but real computation are made with LightSimBackend.
            # NB: for this specific example it is useless to do it because LightSimBackend and PandaPowerBackend have
            # by default the same order etc. This is just an illustration here

            # NB as of now you cannot use a runner with this method (yet)

            env = grid2op.make(..., backend=backend)

            # do regular computations here

    """
    IS_BK_CONVERTER = True
    
    def __init__(
        self,
        source_backend_class,
        target_backend_class,
        target_backend_grid_path=None,
        sub_source_target=None,
        detailed_infos_for_cascading_failures=False,
        use_target_backend_name=False,
        kwargs_target_backend=None,
        kwargs_source_backend=None,
    ):
        Backend.__init__(
            self,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
            use_target_backend_name=use_target_backend_name,
            kwargs_target_backend=kwargs_target_backend,
            kwargs_source_backend=kwargs_source_backend,
        )
        difcf = detailed_infos_for_cascading_failures
        if kwargs_source_backend is None:
            kwargs_source_backend = {}
        self.source_backend = source_backend_class(
            detailed_infos_for_cascading_failures=difcf,
            **kwargs_source_backend
        )  # the one for the order of the elements
        if kwargs_target_backend is None:
            kwargs_target_backend = {}
        self.target_backend = target_backend_class(
            detailed_infos_for_cascading_failures=difcf,
            **kwargs_target_backend
        )  # the one to computes powerflow
        # if the target backend (the one performing the powerflows) needs a different file
        self.target_backend_grid_path = target_backend_grid_path

        # key: name in the source backend, value name in the target backend, for the substations
        self.sub_source_target = sub_source_target

        # if tmp is from the target backend, then tmp[self._line_tg2sr] is ordered according to the source backend
        self._sub_tg2sr = None
        self._sub_sr2tg = None
        self._line_tg2sr = None
        self._line_sr2tg = None
        self._gen_tg2sr = None
        self._gen_sr2tg = None
        self._load_tg2sr = None
        self._load_sr2tg = None
        self._shunt_tg2sr = None
        self._shunt_sr2tg = None
        self._topo_tg2sr = None
        self._topo_sr2tg = None
        self._storage_tg2sr = None
        self._storage_sr2tg = None

        # for redispatching data
        self.path_redisp = None
        self.name_redisp = None
        self.path_grid_layout = None
        self.name_grid_layout = None
        self.path_storage_data = None
        self.name_storage_data = None

        # for easier copy of np array
        self.cst1 = dt_float(1.0)

        # which name to use when loading the data (if names_chronics_to_backend has not been set)
        self.use_target_backend_name = use_target_backend_name 
        self.names_target_to_source = None
        
        # TODO storage check all this class ! + the doc of the backend

    def load_grid(self, path=None, filename=None):
        self.source_backend.load_grid(path, filename)
        # and now i load the target backend
        if self.target_backend_grid_path is not None:
            self.target_backend.load_grid(path=self.target_backend_grid_path)
        else:
            # both source and target backend understands the same format
            self.target_backend.load_grid(path, filename)

    def _assert_same_grid(self):
        """basic assertion that self and the target backend have the same grid
        but not necessarily the same object at the same place of course"""
        cls = type(self)
        tg_cls = type(self.target_backend)
        if cls.n_sub != tg_cls.n_sub:
            raise Grid2OpException(ERROR_NB_ELEMENTS.format("substations"))
        if cls.n_gen != tg_cls.n_gen:
            raise Grid2OpException(ERROR_NB_ELEMENTS.format("generators"))
        if cls.n_load != tg_cls.n_load:
            raise Grid2OpException(ERROR_NB_ELEMENTS.format("loads"))
        if cls.n_line != tg_cls.n_line:
            raise Grid2OpException(ERROR_NB_ELEMENTS.format("lines"))
        if cls.n_storage > 0:
            if cls.n_storage != tg_cls.n_storage:
                raise Grid2OpException(ERROR_NB_ELEMENTS.format("storages"))
        else:
            # a possible reason is that the "source backend" do not support storage
            # but the target one does. In this case I issue a warning,
            # and make sure I have not storage.
            if tg_cls.n_storage > 0:
                warnings.warn("BackendConverter: the source backend does not appear to support storage units, "
                              "but the target one does (and there are some storage units on the grid). "
                              "Be aware that the converted backend will NOT support storage units.")

    def _init_myself(self):
        # shortcut to set all information related to the class, except the name of the environment
        # this should been done when the source backend is fully initialized only
        self._assert_same_grid()

        # and now init all the converting vectors
        # a) for substation
        self._sub_tg2sr = np.full(self.n_sub, fill_value=-1, dtype=dt_int)
        self._sub_sr2tg = np.full(self.n_sub, fill_value=-1, dtype=dt_int)
        if self.sub_source_target is None:
            # automatic mode
            # I can only do it if the names matches
            if np.all(
                sorted(self.source_backend.name_sub)
                == sorted(self.target_backend.name_sub)
            ):
                for id_source, nm_source in enumerate(self.source_backend.name_sub):
                    id_target = np.nonzero(self.target_backend.name_sub == nm_source)[0]
                    self._sub_tg2sr[id_source] = id_target
                    self._sub_sr2tg[id_target] = id_source
        else:
            for id_source, nm_source in enumerate(self.source_backend.name_sub):
                nm_target = self.sub_source_target[nm_source]
                id_target = np.nonzero(self.target_backend.name_sub == nm_target)[0]
                self._sub_tg2sr[id_source] = id_target
                self._sub_sr2tg[id_target] = id_source

        # b) for load
        self._load_tg2sr = np.full(self.n_load, fill_value=-1, dtype=dt_int)
        self._load_sr2tg = np.full(self.n_load, fill_value=-1, dtype=dt_int)
        # automatic mode
        self._auto_fill_vect_load_gen_shunt(
            n_element=self.n_load,
            source_2_id_sub=self.source_backend.load_to_subid,
            target_2_id_sub=self.target_backend.load_to_subid,
            tg2sr=self._load_tg2sr,
            sr2tg=self._load_sr2tg,
            nm="load",
        )

        # c) for generator
        self._gen_tg2sr = np.full(self.n_gen, fill_value=-1, dtype=dt_int)
        self._gen_sr2tg = np.full(self.n_gen, fill_value=-1, dtype=dt_int)
        # automatic mode
        self._auto_fill_vect_load_gen_shunt(
            n_element=self.n_gen,
            source_2_id_sub=self.source_backend.gen_to_subid,
            target_2_id_sub=self.target_backend.gen_to_subid,
            tg2sr=self._gen_tg2sr,
            sr2tg=self._gen_sr2tg,
            nm="gen",
        )

        # d) for powerline
        self._line_tg2sr = np.full(self.n_line, fill_value=-1, dtype=dt_int)
        self._line_sr2tg = np.full(self.n_line, fill_value=-1, dtype=dt_int)
        # automatic
        self._auto_fill_vect_powerline()

        # e) and now the topology vectors.
        self._topo_tg2sr = np.full(self.source_backend.dim_topo, fill_value=-1, dtype=dt_int)
        self._topo_sr2tg = np.full(self.target_backend.dim_topo, fill_value=-1, dtype=dt_int)
        self._auto_fill_vect_topo()

        # f) for the storage units
        self._storage_tg2sr = np.full(self.n_storage, fill_value=-1, dtype=dt_int)
        self._storage_sr2tg = np.full(self.n_storage, fill_value=-1, dtype=dt_int)
        
        # automatic mode
        if self.n_storage:
            self._auto_fill_vect_load_gen_shunt(
                n_element=self.n_storage,
                source_2_id_sub=self.source_backend.storage_to_subid,
                target_2_id_sub=self.target_backend.storage_to_subid,
                tg2sr=self._storage_tg2sr,
                sr2tg=self._storage_sr2tg,
                nm="storage",
            )

        if type(self.source_backend).shunts_data_available and not type(self.target_backend).shunts_data_available:
            raise Grid2OpException("Source backend supports shunts and not target backend. This is not handled")
        elif type(self.target_backend).shunts_data_available and not type(self.source_backend).shunts_data_available:
            raise Grid2OpException("Target backend supports shunts and not source backend. This is not handled")
        
        # TODO shunts data available
        # shunt are available if both source and target provide it
        type(self).shunts_data_available = (
            type(self.source_backend).shunts_data_available
            and type(self.target_backend).shunts_data_available
        )
        if type(self).shunts_data_available:
            self._shunt_tg2sr = np.full(self.n_shunt, fill_value=-1, dtype=dt_int)
            self._shunt_sr2tg = np.full(self.n_shunt, fill_value=-1, dtype=dt_int)
            # automatic mode
            self._auto_fill_vect_load_gen_shunt(
                n_element=self.n_shunt,
                source_2_id_sub=self.source_backend.shunt_to_subid,
                target_2_id_sub=self.target_backend.shunt_to_subid,
                tg2sr=self._shunt_tg2sr,
                sr2tg=self._shunt_sr2tg,
                nm="shunt",
            )
        else:
            self.n_shunt = 0
            self.name_shunt = np.empty(0, dtype=str)
        self._thermal_limit_a = 1.0 * self.target_backend.thermal_limit_a
        self.set_thermal_limit(self.target_backend.thermal_limit_a[self._line_tg2sr])

    def _get_possible_target_ids(self, id_source, source_2_id_sub, target_2_id_sub, nm):
        id_sub_source = source_2_id_sub[id_source]
        id_sub_target = self._sub_tg2sr[id_sub_source]
        ids_target = np.nonzero(target_2_id_sub == id_sub_target)[0]
        if ids_target.shape[0] == 0:
            raise RuntimeError(
                ERROR_ELEMENT_CONNECTED.format(nm, id_sub_target, id_sub_source)
            )
        return id_sub_target, ids_target

    def _auto_fill_vect_load_gen_shunt(
        self, n_element, source_2_id_sub, target_2_id_sub, tg2sr, sr2tg, nm
    ):
        nb_load_per_sub = np.zeros(self.n_sub, dtype=dt_int)
        if source_2_id_sub.shape[0] != n_element:
            raise RuntimeError(
                "Impossible to convert backend that do not have the same number of objects"
            )
        if target_2_id_sub.shape[0] != n_element:
            raise RuntimeError(
                "Impossible to convert backend that do not have the same number of objects"
            )
        for id_source in range(n_element):
            id_sub_target, id_target = self._get_possible_target_ids(
                id_source, source_2_id_sub, target_2_id_sub, nm
            )
            id_target = id_target[nb_load_per_sub[id_sub_target]]
            # TODO no no no use the "to_sub_pos" to compute that, and even better raise an error in this case
            # this means automatic is failing here !
            nb_load_per_sub[id_sub_target] += 1
            tg2sr[id_source] = id_target
            sr2tg[id_target] = id_source

    def _auto_fill_vect_powerline(self):
        # automatic matching
        nb_load_per_sub = np.zeros((self.n_sub, self.n_sub), dtype=dt_int)
        n_element = self.n_line
        source_or_2_id_sub = self.source_backend.line_or_to_subid
        target_or_2_id_sub = self.target_backend.line_or_to_subid
        source_ex_2_id_sub = self.source_backend.line_ex_to_subid
        target_ex_2_id_sub = self.target_backend.line_ex_to_subid
        nm = "powerline"
        tg2sr = self._line_tg2sr
        sr2tg = self._line_sr2tg
        for id_source in range(n_element):
            idor_sub_source = source_or_2_id_sub[id_source]
            idor_sub_target = self._sub_tg2sr[idor_sub_source]
            idex_sub_source = source_ex_2_id_sub[id_source]
            idex_sub_target = self._sub_tg2sr[idex_sub_source]
            ids_target = np.nonzero(
                (target_or_2_id_sub == idor_sub_target)
                & (target_ex_2_id_sub == idex_sub_target)
            )[0]
            if ids_target.shape[0] == 0:
                raise RuntimeError(
                    ERROR_ELEMENT_CONNECTED.format(
                        nm,
                        "{}->{}".format(idor_sub_target, idex_sub_target),
                        "{}->{}".format(idor_sub_source, idex_sub_source),
                    )
                )
            id_target = ids_target[nb_load_per_sub[idor_sub_target, idex_sub_target]]
            # TODO no no no use the "to_sub_pos" to compute that, and even better raise an error in this case
            # this means automatic is failing here !
            nb_load_per_sub[idor_sub_target, idex_sub_target] += 1
            tg2sr[id_source] = id_target
            sr2tg[id_target] = id_source

    def _auto_fill_vect_topo(self):
        self._auto_fill_vect_topo_aux(
            self.n_load,
            self.source_backend.load_pos_topo_vect,
            self.target_backend.load_pos_topo_vect,
            self._load_sr2tg,
        )
        self._auto_fill_vect_topo_aux(
            self.n_gen,
            self.source_backend.gen_pos_topo_vect,
            self.target_backend.gen_pos_topo_vect,
            self._gen_sr2tg,
        )
        self._auto_fill_vect_topo_aux(
            self.n_line,
            self.source_backend.line_or_pos_topo_vect,
            self.target_backend.line_or_pos_topo_vect,
            self._line_sr2tg,
        )
        self._auto_fill_vect_topo_aux(
            self.n_line,
            self.source_backend.line_ex_pos_topo_vect,
            self.target_backend.line_ex_pos_topo_vect,
            self._line_sr2tg,
        )
        
        if self.n_storage:
            self._auto_fill_vect_topo_aux(
                self.n_storage,
                self.source_backend.storage_pos_topo_vect,
                self.target_backend.storage_pos_topo_vect,
                self._storage_sr2tg,
            )

    def _auto_fill_vect_topo_aux(self, n_elem, source_pos, target_pos, sr2tg):
        # TODO that might not be working as intented... it always says it's the identity...
        self._topo_tg2sr[source_pos[sr2tg]] = target_pos
        self._topo_sr2tg[target_pos] = source_pos[sr2tg]

    def assert_grid_correct(self):
        # this is done before a call to this function, by the environment
        tg_cls = type(self.target_backend)
        sr_cls = type(self.source_backend)
        
        env_name = type(self).env_name
        tg_cls.set_env_name(env_name)
        sr_cls.set_env_name(env_name)

        # handle specifc case of shunt data:
        if not type(self.target_backend).shunts_data_available:
            if type(self.source_backend).shunts_data_available:
                raise Grid2OpException("Impossible to use a converter when one of the backend "
                                       "supports shunt and the other not at the moment.")
            # TODO shunts_data_available: you need ideally to create a different class
            # for the backend that does not support it.
            
            # disable the shunt data in grid2op.
            # type(self.source_backend).shunts_data_available = False
            # type(self.source_backend).n_shunt = None
            # type(self.source_backend).name_shunt = np.empty(0, dtype=str)

        self._init_class_attr(obj=self.source_backend)
        if self.path_redisp is not None:
            # redispatching data were available
            try:
                super().load_redispacthing_data(self.path_redisp, name=self.name_redisp)
                self.source_backend.load_redispacthing_data(
                    self.path_redisp, name=self.name_redisp
                )
            except BackendError as exc_:
                self.redispatching_unit_commitment_availble = False
                warnings.warn(f"Impossible to load redispatching data. This is not an error but you will not be able "
                            f"to use all grid2op functionalities. "
                            f"The error was: \"{exc_}\"")
        if self.path_storage_data is not None:
            super().load_storage_data(self.path_storage_data, self.name_storage_data)
            self.source_backend.load_storage_data(
                self.path_storage_data, name=self.name_storage_data
            )
            self.target_backend.load_storage_data(
                self.path_storage_data, name=self.name_storage_data
            )
        if self.path_grid_layout is not None:
            # grid layout data were available
            super().load_grid_layout(self.path_grid_layout, self.name_grid_layout)
            self.source_backend.load_grid_layout(
                self.path_redisp, name=self.name_redisp
            )

        # init the target backend (the one that does the computation and that is initialized)
        self.target_backend.assert_grid_correct()

        # initialize the other one, because, well the grid should be seen from both backend
        self.source_backend._init_class_attr(obj=self)
        self.source_backend.assert_grid_correct()
        # and this should be called after all the rest
        super().assert_grid_correct()

        # everything went well, so i can properly terminate my initialization
        self._init_myself()
        
        # redefine this as the class changed after "assert grid correct"
        tg_cls = type(self.target_backend)
        sr_cls = type(self.source_backend)
        cls = type(self)

        if self.sub_source_target is None:
            # automatic mode for substations, names must match
            assert np.all(
                self.target_backend.name_sub[self._sub_tg2sr]
                == self.source_backend.name_sub
            )
            assert np.all(
                self.source_backend.name_sub[self._sub_sr2tg]
                == self.target_backend.name_sub
            )

        # check that all corresponding vectors are valid (and properly initialized, like every component above 0 etc.)
        self._check_both_consistent(self._line_tg2sr, self._line_sr2tg)
        self._check_both_consistent(self._load_tg2sr, self._load_sr2tg)
        self._check_both_consistent(self._gen_tg2sr, self._gen_sr2tg)
        self._check_both_consistent(self._sub_tg2sr, self._sub_sr2tg)
        
        if cls.n_storage == tg_cls.n_storage:
            # both source and target supports storage units
            self._check_both_consistent(self._topo_tg2sr, self._topo_sr2tg)
        elif self.n_storage == 0:
            # n_storage == 0 and there are storage units on the source backend
            # this means that the target_backend supports storage but not
            # the source one
            assert np.all(self._topo_sr2tg[self._topo_tg2sr] >= 0)
            assert np.all(sorted(self._topo_sr2tg[self._topo_tg2sr]) == np.arange(self.dim_topo))
            
            topo_sr2tg_without_storage = self._topo_sr2tg[self._topo_sr2tg >= 0]
            assert (self._topo_sr2tg == -1).sum() == tg_cls.n_storage
            assert np.all(self._topo_tg2sr[topo_sr2tg_without_storage] >= 0)
            target_without_storage = np.array([i for i in range(tg_cls.dim_topo) 
                                               if not i in tg_cls.storage_pos_topo_vect])
            assert np.all(sorted(self._topo_tg2sr[topo_sr2tg_without_storage]) == target_without_storage)
            self._topo_sr2tg = topo_sr2tg_without_storage

        if type(self).shunts_data_available:
            self._check_both_consistent(self._shunt_tg2sr, self._shunt_sr2tg)

        # finally check that powergrids are identical (up to the env name)
        tg_cls.same_grid_class(sr_cls)
        
        # once everything is done, make the converter for the names
        d_loads = {tg_cls.name_load[i]: sr_cls.name_load[self._load_sr2tg[i]]
                    for i in range(cls.n_load)}
        d_gens = {tg_cls.name_gen[i]: sr_cls.name_gen[self._gen_sr2tg[i]]
                    for i in range(cls.n_gen)}
        d_lines = {tg_cls.name_line[i]: sr_cls.name_line[self._line_sr2tg[i]]
                    for i in range(cls.n_line)}
        d_subs = {tg_cls.name_sub[i]: sr_cls.name_sub[self._sub_sr2tg[i]]
                    for i in range(cls.n_sub)}
        dict_ = {"loads": d_loads, "lines": d_lines, "prods": d_gens, "subs": d_subs}
        self.names_target_to_source = dict_

    def _check_vect_valid(self, vect):
        assert np.all(
            vect >= 0
        ), ERROR_INVALID_VECTOR
        assert sorted(np.unique(vect)) == sorted(
            vect
        ), ERROR_INVALID_VECTOR
        if vect.shape[0] > 0:
            assert (
                np.max(vect) == vect.shape[0] - 1
            ), ERROR_INVALID_VECTOR

    def _check_both_consistent(self, tg2sr, sr2tg):
        self._check_vect_valid(tg2sr)
        self._check_vect_valid(sr2tg)
        res = np.arange(tg2sr.shape[0])
        assert np.all(tg2sr[sr2tg] == res)
        assert np.all(sr2tg[tg2sr] == res)

    def assert_grid_correct_after_powerflow(self):
        # we don't assert that `self.source_backend.assert_grid_correct_after_powerflow()`
        # because obviously no powerflow are run using the source backend.
        self.target_backend.assert_grid_correct_after_powerflow()
        super().assert_grid_correct_after_powerflow()
        self._sh_vnkv = self.target_backend._sh_vnkv

    def reset(self, grid_path, grid_filename=None):
        """
        Reload the power grid.
        For backwards compatibility this method calls `Backend.load_grid`.
        But it is encouraged to overload it in the subclasses.
        """
        self.target_backend.reset(grid_path, grid_filename=grid_filename)

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
        source_backend_sv = self.source_backend
        target_backend_sv = self.target_backend
        self.source_backend = None
        self.target_backend = None
        res = copy.deepcopy(self)
        res.source_backend = source_backend_sv.copy()
        res.target_backend = target_backend_sv.copy()
        self.source_backend = source_backend_sv
        self.target_backend = target_backend_sv
        return res

    def save_file(self, full_path):
        self.target_backend.save_file(full_path)
        self.source_backend.save_file(full_path)

    def get_line_status(self):
        tmp = self.target_backend.get_line_status()
        return tmp[self._line_tg2sr]

    def get_line_flow(self):
        tmp = self.target_backend.get_line_flow()
        return self.cst1 * tmp[self._line_tg2sr]

    def set_thermal_limit(self, limits):
        super().set_thermal_limit(limits=limits)
        self.source_backend.set_thermal_limit(limits=limits)
        if limits is not None:
            self.target_backend.set_thermal_limit(limits=limits[self._line_sr2tg])

    def get_thermal_limit(self):
        tmp = self.target_backend.get_thermal_limit()
        return self.cst1 * tmp[self._line_tg2sr]

    def get_topo_vect(self):
        tmp = self.target_backend.get_topo_vect()
        return tmp[self._topo_tg2sr]

    def generators_info(self):
        prod_p, prod_q, prod_v = self.target_backend.generators_info()
        return (
            self.cst1 * prod_p[self._gen_tg2sr],
            self.cst1 * prod_q[self._gen_tg2sr],
            self.cst1 * prod_v[self._gen_tg2sr],
        )

    def loads_info(self):
        load_p, load_q, load_v = self.target_backend.loads_info()
        return (
            self.cst1 * load_p[self._load_tg2sr],
            self.cst1 * load_q[self._load_tg2sr],
            self.cst1 * load_v[self._load_tg2sr],
        )

    def lines_or_info(self):
        p_, q_, v_, a_ = self.target_backend.lines_or_info()
        return (
            self.cst1 * p_[self._line_tg2sr],
            self.cst1 * q_[self._line_tg2sr],
            self.cst1 * v_[self._line_tg2sr],
            self.cst1 * a_[self._line_tg2sr],
        )

    def lines_ex_info(self):
        p_, q_, v_, a_ = self.target_backend.lines_ex_info()
        return (
            self.cst1 * p_[self._line_tg2sr],
            self.cst1 * q_[self._line_tg2sr],
            self.cst1 * v_[self._line_tg2sr],
            self.cst1 * a_[self._line_tg2sr],
        )

    def storages_info(self):
        p_, q_, v_ = self.target_backend.storages_info()
        return (
            self.cst1 * p_[self._storage_tg2sr],
            self.cst1 * q_[self._storage_tg2sr],
            self.cst1 * v_[self._storage_tg2sr],
        )

    def shunt_info(self):
        if self._shunt_tg2sr is not None:
            # shunts are supported by both source and target backend
            sh_p, sh_q, sh_v, sh_bus = self.target_backend.shunt_info()
            return (
                sh_p[self._shunt_tg2sr],
                sh_q[self._shunt_tg2sr],
                sh_v[self._shunt_tg2sr],
                sh_bus[self._shunt_tg2sr],
            )
        # shunt are not supported by either source or target backend
        return [], [], [], []

    def sub_from_bus_id(self, bus_id):
        # not supported because the bus_id is given into the source backend,
        # and i need to convert to to the target backend, not sure how to do that atm
        raise Grid2OpException(
            "This backend doesn't allow to get the substation from the bus id."
        )

    def _disconnect_line(self, id_):
        id_target = int(
            self._line_tg2sr[id_]
        )  # not sure why, but it looks to work this way
        self.target_backend._disconnect_line(id_target)

    def _transform_action(self, source_action):
        # transform the source action into the target backend action
        # source_action: a backend action!
        target_action = copy.deepcopy(source_action)
        # consistent with TestLoadingBackendFunc, otherwise it's not correct
        target_action.reorder(
            no_load=self._load_sr2tg,
            no_gen=self._gen_sr2tg,
            no_topo=self._topo_sr2tg,
            no_shunt=self._shunt_sr2tg,
            no_storage=self._storage_sr2tg,
        )
        return target_action

    def load_redispacthing_data(self, path, name="prods_charac.csv"):
        # data are loaded with the name of the source backend, i need to map it to the target backend too
        self.path_redisp = path
        self.name_redisp = name

    def load_storage_data(self, path, name="storage_units_charac.csv"):
        # data are loaded with the name of the source backend, i need to map it to the target backend too
        self.path_storage_data = path
        self.name_storage_data = name

    def load_grid_layout(self, path, name="grid_layout.json"):
        self.path_grid_layout = path
        self.name_grid_layout = name

    def get_action_to_set(self):
        act = self.target_backend.get_action_to_set()
        line_vect = self._line_sr2tg
        gen_vect = self._gen_sr2tg
        load_vect = self._load_sr2tg
        topo_vect = self._topo_sr2tg
        dict_ = act._dict_inj
        if "prod_p" in dict_:
            dict_["dict_"] = dict_["prod_p"][gen_vect]
        if "prod_v" in dict_:
            dict_["dict_"] = dict_["prod_v"][gen_vect]
        if "load_p" in dict_:
            dict_["dict_"] = dict_["load_p"][load_vect]
        if "load_q" in dict_:
            dict_["dict_"] = dict_["load_q"][load_vect]

        act._set_topo_vect[:] = act._set_topo_vect[topo_vect]
        act._change_bus_vect[:] = act._change_bus_vect[topo_vect]
        act._hazards[:] = act._hazards[line_vect]
        act._maintenance[:] = act._hazards[line_vect]
        act._redispatch[:] = act._redispatch[gen_vect]
        act._set_line_status[:] = act._set_line_status[line_vect]
        act._switch_line_status[:] = act._switch_line_status[line_vect]

        if act.shunt_added and type(act).shunts_data_available:
            shunt_vect = self._shunt_sr2tg
            act.shunt_p[:] = act.shunt_p[shunt_vect]
            act.shunt_q[:] = act.shunt_q[shunt_vect]
            act.shunt_bus[:] = act.shunt_bus[shunt_vect]

        return act

    def update_thermal_limit(self, env):
        # TODO
        # env has the powerline stored in the order of the source backend, but i need
        # to have them stored in the order of the target backend for such function
        pass

    # TODO update_from_obs too, maybe ?
