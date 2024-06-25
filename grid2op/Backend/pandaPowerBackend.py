# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os  # load the python os default module
import sys  # laod the python sys default module
import copy
import warnings

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple

import pandapower as pp
import scipy

import grid2op
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Action import BaseAction
from grid2op.Exceptions import BackendError
from grid2op.Backend.backend import Backend

try:
    import numba
    NUMBA_ = True
except (ImportError, ModuleNotFoundError):
    NUMBA_ = False
    warnings.warn(
        "Numba cannot be loaded. You will gain possibly massive speed if installing it by "
        "\n\t{} -m pip install numba\n".format(sys.executable)
    )


class PandaPowerBackend(Backend):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        If you want to code a backend to use grid2op with another powerflow, you can get inspired
        from this class. Note However that implies knowing the behaviour
        of PandaPower.

    This module presents an example of an implementation of a `grid2op.Backend` when using the powerflow
    implementation "pandapower" available at `PandaPower <https://www.pandapower.org/>`_ for more details about
    this backend. This file is provided as an example of a proper :class:`grid2op.Backend.Backend` implementation.

    This backend currently does not work with 3 winding transformers and other exotic object.

    As explained in the `grid2op.Backend` module, every module must inherit the `grid2op.Backend` class.

    This class have more attributes that are used internally for faster information retrieval.

    Attributes
    ----------
    prod_pu_to_kv: :class:`numpy.array`, dtype:float
        The ratio that allow the conversion from pair-unit to kv for the generators

    load_pu_to_kv: :class:`numpy.array`, dtype:float
        The ratio that allow the conversion from pair-unit to kv for the loads

    lines_or_pu_to_kv: :class:`numpy.array`, dtype:float
        The ratio that allow the conversion from pair-unit to kv for the origin side of the powerlines

    lines_ex_pu_to_kv: :class:`numpy.array`, dtype:float
        The ratio that allow the conversion from pair-unit to kv for the extremity side of the powerlines

    p_or: :class:`numpy.array`, dtype:float
        The active power flowing at the origin side of each powerline

    q_or: :class:`numpy.array`, dtype:float
        The reactive power flowing at the origin side of each powerline

    v_or: :class:`numpy.array`, dtype:float
        The voltage magnitude at the origin bus of the powerline

    a_or: :class:`numpy.array`, dtype:float
        The current flowing at the origin side of each powerline

    p_ex: :class:`numpy.array`, dtype:float
        The active power flowing at the extremity side of each powerline

    q_ex: :class:`numpy.array`, dtype:float
        The reactive power flowing at the extremity side of each powerline

    a_ex: :class:`numpy.array`, dtype:float
        The current flowing at the extremity side of each powerline

    v_ex: :class:`numpy.array`, dtype:float
        The voltage magnitude at the extremity bus of the powerline

    Examples
    ---------
    The only recommended way to use this class is by passing an instance of a Backend into the "make"
    function of grid2op. Do not attempt to use a backend outside of this specific usage.

    .. code-block:: python

        import grid2op
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()

        env = grid2op.make(backend=backend)
        # and use "env" as "any open ai gym" environment.

    """
    shunts_data_available = True
    
    def __init__(
        self,
        detailed_infos_for_cascading_failures : bool=False,
        lightsim2grid : bool=False,  # use lightsim2grid as pandapower powerflow solver
        dist_slack : bool=False,
        max_iter : int=10,
        can_be_copied: bool=True,
        with_numba: bool=NUMBA_,
    ):
        from grid2op.MakeEnv.Make import _force_test_dataset
        if _force_test_dataset():
            if with_numba:
                warnings.warn(f"Forcing `test=True` will disable numba for {type(self)}")
            with_numba = False
            
        Backend.__init__(
            self,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
            can_be_copied=can_be_copied,
            lightsim2grid=lightsim2grid,
            dist_slack=dist_slack,
            max_iter=max_iter,
            with_numba=with_numba
        )
        self.with_numba : bool = with_numba
        self.prod_pu_to_kv : Optional[np.ndarray] = None
        self.load_pu_to_kv : Optional[np.ndarray]  = None
        self.lines_or_pu_to_kv : Optional[np.ndarray]  = None
        self.lines_ex_pu_to_kv : Optional[np.ndarray]  = None
        self.storage_pu_to_kv : Optional[np.ndarray]  = None

        self.p_or : Optional[np.ndarray]  = None
        self.q_or : Optional[np.ndarray]  = None
        self.v_or : Optional[np.ndarray]  = None
        self.a_or : Optional[np.ndarray]  = None
        self.p_ex : Optional[np.ndarray]  = None
        self.q_ex : Optional[np.ndarray]  = None
        self.v_ex : Optional[np.ndarray]  = None
        self.a_ex : Optional[np.ndarray]  = None

        self.load_p : Optional[np.ndarray]  = None
        self.load_q : Optional[np.ndarray]  = None
        self.load_v : Optional[np.ndarray]  = None

        self.storage_p : Optional[np.ndarray]  = None
        self.storage_q : Optional[np.ndarray]  = None
        self.storage_v : Optional[np.ndarray]  = None

        self.prod_p : Optional[np.ndarray]  = None
        self.prod_q : Optional[np.ndarray]  = None
        self.prod_v : Optional[np.ndarray]  = None
        self.line_status : Optional[np.ndarray]  = None

        self._pf_init : str = "flat"
        self._pf_init : str = "results"
        self._nb_bus_before : Optional[int] = None  # number of active bus at the preceeding step

        self.thermal_limit_a : Optional[np.ndarray]  = None

        self._iref_slack : Optional[int] = None
        self._id_bus_added : Optional[int] = None
        self._fact_mult_gen : int = -1
        self._what_object_where = None
        self._number_true_line = -1
        self._corresp_name_fun = {}
        self._get_vector_inj = {}
        self._vars_action = BaseAction.attr_list_vect
        self._vars_action_set = BaseAction.attr_list_vect
        self.cst_1 = dt_float(1.0)
        self._topo_vect = None
        self.slack_id = None

        # function to rstore some information
        self.__nb_bus_before = None  # number of substation in the powergrid
        self.__nb_powerline = (
            None  # number of powerline (real powerline, not transformer)
        )
        self._init_bus_load = None
        self._init_bus_gen = None
        self._init_bus_lor = None
        self._init_bus_lex = None
        self._get_vector_inj = None
        self._big_topo_to_obj = None
        self._big_topo_to_backend = None
        self.__pp_backend_initial_grid = None  # initial state to facilitate the "reset"

        # Mapping some fun to apply bus updates
        self._type_to_bus_set = [
            self._apply_load_bus,
            self._apply_gen_bus,
            self._apply_lor_bus,
            self._apply_trafo_hv,
            self._apply_lex_bus,
            self._apply_trafo_lv,
        ]

        self.tol = None  # this is NOT the pandapower tolerance !!!! this is used to check if a storage unit
        # produce / absorbs anything

        # TODO storage doc (in grid2op rst) of the backend
        self.can_output_theta = True  # I support the voltage angle
        self.theta_or  : Optional[np.ndarray] = None
        self.theta_ex  : Optional[np.ndarray] = None
        self.load_theta  : Optional[np.ndarray] = None
        self.gen_theta  : Optional[np.ndarray] = None
        self.storage_theta  : Optional[np.ndarray] = None

        self._lightsim2grid : bool = lightsim2grid
        self._dist_slack : bool = dist_slack
        self._max_iter : bool = max_iter
        self._in_service_line_col_id = None
        self._in_service_trafo_col_id = None
        self._in_service_storage_cold_id = None
        self.div_exception = None

    def _check_for_non_modeled_elements(self):
        """This function check for elements in the pandapower grid that will have no impact on grid2op.
        See the full list of grid2op modeled elements in :ref:`modeled-elements-module`
        """
        for el_nm in [
            "trafo3w",
            "sgen",
            "switch",
            "motor",
            "asymmetric_load",
            "asymmetric_sgen",
            "impedance",
            "ward",
            "xward",
            "dcline",
            "measurement",
        ]:
            if el_nm in self._grid:
                if self._grid[el_nm].shape[0]:
                    warnings.warn(
                        f'There are "{el_nm}" in the pandapower grid. These '
                        f"elements are not modeled on grid2op side (the environment will "
                        f"work, but you won't be able to modify them)."
                    )

    def get_theta(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        TODO doc

        Returns
        -------
        theta_or: ``numpy.ndarray``
            For each orgin side of powerline, gives the voltage angle (in degree)
        theta_ex: ``numpy.ndarray``
            For each extremity side of powerline, gives the voltage angle (in degree)
        load_theta: ``numpy.ndarray``
            Gives the voltage angle (in degree) to the bus at which each load is connected
        gen_theta: ``numpy.ndarray``
            Gives the voltage angle (in degree) to the bus at which each generator is connected
        storage_theta: ``numpy.ndarray``
            Gives the voltage angle (in degree) to the bus at which each storage unit is connected
        """
        return (
            self.cst_1 * self.theta_or,
            self.cst_1 * self.theta_ex,
            self.cst_1 * self.load_theta,
            self.cst_1 * self.gen_theta,
            self.cst_1 * self.storage_theta,
        )
    
    def get_nb_active_bus(self) -> int:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Compute the amount of buses "in service" eg with at least a powerline connected to it.

        Returns
        -------
        res: :class:`int`
            The total number of active buses.
        """
        return self._grid.bus["in_service"].sum()

    @staticmethod
    def _load_grid_load_p_mw(grid) -> pd.Series:
        return grid.load["p_mw"]

    @staticmethod
    def _load_grid_load_q_mvar(grid) -> pd.Series:
        return grid.load["q_mvar"]

    @staticmethod
    def _load_grid_gen_p_mw(grid) -> pd.Series:
        return grid.gen["p_mw"]

    @staticmethod
    def _load_grid_gen_vm_pu(grid) -> pd.Series:
        return grid.gen["vm_pu"]

    def reset(self,
              path : Union[os.PathLike, str],
              grid_filename : Optional[Union[os.PathLike, str]]=None) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Reload the grid.
        For pandapower, it is a bit faster to store of a copy of itself at the end of load_grid
        and deep_copy it to itself instead of calling load_grid again
        """
        # Assign the content of itself as saved at the end of load_grid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._grid = copy.deepcopy(self.__pp_backend_initial_grid)
        self._reset_all_nan()
        self._topo_vect[:] = self._get_topo_vect()
        self.comp_time = 0.0

    def load_grid(self,
                  path : Union[os.PathLike, str],
                  filename : Optional[Union[os.PathLike, str]]=None) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Load the _grid, and initialize all the member of the class. Note that in order to perform topological
        modification of the substation of the underlying powergrid, some buses are added to the test case loaded. They
        are set as "out of service" unless a topological action acts on these specific substations.

        """
        self.can_handle_more_than_2_busbar()
        full_path = self.make_complete_path(path, filename)

        with warnings.catch_warnings():
            # remove deprecationg warnings for old version of pandapower
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            self._grid = pp.from_json(full_path)
        self._check_for_non_modeled_elements()

        # add the slack bus that is often not modeled as a generator, but i need it for this backend to work
        bus_gen_added = None
        i_ref = None
        self._iref_slack = None
        self._id_bus_added = None
        
        self._aux_run_pf_init()  # run an intiail powerflow, just in case
        
        new_pp_version = False
        if not "slack_weight" in self._grid.gen:
            self._grid.gen["slack_weight"] = 1.0
        else:
            new_pp_version = True
            
        if np.all(~self._grid.gen["slack"]):
            # there are not defined slack bus on the data, i need to hack it up a little bit
            pd2ppc = self._grid._pd2ppc_lookups["bus"]  # pd2ppc[pd_id] = ppc_id
            ppc2pd = np.argsort(pd2ppc)  # ppc2pd[ppc_id] = pd_id
            for gen_id_pp, el in enumerate(self._grid._ppc["gen"][:, 0]):
                if (
                    int(el)
                    not in self._grid._pd2ppc_lookups["bus"][
                        self._grid.gen["bus"].values
                    ]
                ):
                    if bus_gen_added is not None:
                        # TODO handle better when distributed slack bus
                        # raise RuntimeError("Impossible to recognize the powergrid")
                        warnings.warn(
                            "Your grid has a distributed slack bus. Just so you know, it is not"
                            "fully supported at the moment. (it will be converted to a single slack bus)"
                        )

                    bus_gen_added = ppc2pd[int(el)]
                    # see https://matpower.org/docs/ref/matpower5.0/idx_gen.html for details on the comprehension of self._grid._ppc
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        # some warnings are issued depending on pp and pandas version
                        if new_pp_version:
                            id_added = pp.create_gen(
                                self._grid,
                                bus_gen_added,
                                p_mw=self._grid._ppc["gen"][gen_id_pp, 1],
                                vm_pu=self._grid._ppc["gen"][gen_id_pp, 5],
                                min_p_mw=self._grid._ppc["gen"][gen_id_pp, 9],
                                max_p_mw=self._grid._ppc["gen"][gen_id_pp, 8],
                                max_q_mvar=self._grid._ppc["gen"][gen_id_pp, 3],
                                min_q_mvar=self._grid._ppc["gen"][gen_id_pp, 4],
                                slack=i_ref is None,
                                slack_weight=1.0,
                                controllable=True,
                            )
                        else:
                            id_added = pp.create_gen(
                                self._grid,
                                bus_gen_added,
                                p_mw=self._grid._ppc["gen"][gen_id_pp, 1],
                                vm_pu=self._grid._ppc["gen"][gen_id_pp, 5],
                                min_p_mw=self._grid._ppc["gen"][gen_id_pp, 9],
                                max_p_mw=self._grid._ppc["gen"][gen_id_pp, 8],
                                max_q_mvar=self._grid._ppc["gen"][gen_id_pp, 3],
                                min_q_mvar=self._grid._ppc["gen"][gen_id_pp, 4],
                                slack=i_ref is None,
                                controllable=True,
                            )

                    if i_ref is None:
                        i_ref = gen_id_pp
                        self._iref_slack = i_ref
                        self._id_bus_added = id_added  # self._grid.gen.shape[0]
                        # TODO here i force the distributed slack bus too, by removing the other from the ext_grid...
                        self._grid.ext_grid = self._grid.ext_grid.iloc[:1]
        else:
            self.slack_id = (self._grid.gen["slack"].values).nonzero()[0]

        self._aux_run_pf_init()  # run another powerflow with the added generator

        self.__nb_bus_before = self._grid.bus.shape[0]
        self.__nb_powerline = self._grid.line.shape[0]
        self._init_bus_load = self.cst_1 * self._grid.load["bus"].values
        self._init_bus_gen = self.cst_1 * self._grid.gen["bus"].values
        self._init_bus_lor = self.cst_1 * self._grid.line["from_bus"].values
        self._init_bus_lex = self.cst_1 * self._grid.line["to_bus"].values

        t_for = self.cst_1 * self._grid.trafo["hv_bus"].values
        t_fex = self.cst_1 * self._grid.trafo["lv_bus"].values
        self._init_bus_lor = np.concatenate((self._init_bus_lor, t_for)).astype(dt_int)
        self._init_bus_lex = np.concatenate((self._init_bus_lex, t_fex)).astype(dt_int)

        self._grid["ext_grid"]["va_degree"] = 0.0

        # this has the effect to divide by 2 the active power in the added generator, if this generator and the "slack bus"
        # one are connected to the same bus.
        # if not, it must not be done. So basically, i create a vector for which p and q for generator must be multiply
        self._fact_mult_gen = np.ones(self._grid.gen.shape[0])
        # self._fact_mult_gen[-1] += 1

        # now extract the powergrid
        self.n_line = copy.deepcopy(self._grid.line.shape[0]) + copy.deepcopy(
            self._grid.trafo.shape[0]
        )
        if (
            "name" in self._grid.line.columns
            and not self._grid.line["name"].isnull().values.any()
        ):
            self.name_line = [name for name in self._grid.line["name"]]
        else:
            self.name_line = [
                "{from_bus}_{to_bus}_{id_powerline_me}".format(**row, id_powerline_me=i)
                for i, (_, row) in enumerate(self._grid.line.iterrows())
            ]
        if (
            "name" in self._grid.trafo.columns
            and not self._grid.trafo["name"].isnull().values.any()
        ):
            self.name_line += [name_traf for name_traf in self._grid.trafo["name"]]
        else:
            transfo = [
                ("{hv_bus}".format(**row), "{lv_bus}".format(**row))
                for i, (_, row) in enumerate(self._grid.trafo.iterrows())
            ]
            transfo = [sorted(el) for el in transfo]
            self.name_line += [
                "{}_{}_{}".format(*el, i + self._grid.line.shape[0])
                for i, el in enumerate(transfo)
            ]
        self.name_line = np.array(self.name_line)

        self.n_gen = copy.deepcopy(self._grid.gen.shape[0])
        if (
            "name" in self._grid.gen.columns
            and not self._grid.gen["name"].isnull().values.any()
        ):
            self.name_gen = [name_g for name_g in self._grid.gen["name"]]
        else:
            self.name_gen = [
                "gen_{bus}_{index_gen}".format(**row, index_gen=i)
                for i, (_, row) in enumerate(self._grid.gen.iterrows())
            ]
        self.name_gen = np.array(self.name_gen)
        self.n_load = copy.deepcopy(self._grid.load.shape[0])
        if (
            "name" in self._grid.load.columns
            and not self._grid.load["name"].isnull().values.any()
        ):
            self.name_load = [nl for nl in self._grid.load["name"]]
        else:
            self.name_load = [
                "load_{bus}_{index_load}".format(**row, index_load=i)
                for i, (_, row) in enumerate(self._grid.load.iterrows())
            ]
        self.name_load = np.array(self.name_load)

        self.n_storage = copy.deepcopy(self._grid.storage.shape[0])
        if self.n_storage == 0:
            self.set_no_storage()
        else:
            if (
                "name" in self._grid.storage.columns
                and not self._grid.storage["name"].isnull().values.any()
            ):
                self.name_storage = [nl for nl in self._grid.storage["name"]]
            else:
                self.name_storage = [
                    "storage_{bus}_{index_sto}".format(**row, index_sto=i)
                    for i, (_, row) in enumerate(self._grid.storage.iterrows())
                ]
            self.name_storage = np.array(self.name_storage)

        self.n_sub = copy.deepcopy(self._grid.bus.shape[0])
        self.name_sub = ["sub_{}".format(i) for i, row in self._grid.bus.iterrows()]
        self.name_sub = np.array(self.name_sub)

        if type(self).shunts_data_available:
            self.n_shunt = self._grid.shunt.shape[0]
        else:
            self.n_shunt = None
            
        # "hack" to handle topological changes, for now only 2 buses per substation
        add_topo = copy.deepcopy(self._grid.bus)
        # TODO n_busbar: what if non contiguous indexing ???
        for _ in range(self.n_busbar_per_sub - 1):   # self.n_busbar_per_sub and not type(self) here otherwise it erases can_handle_more_than_2_busbar / cannot_handle_more_than_2_busbar
            add_topo.index += add_topo.shape[0]
            add_topo["in_service"] = False
            for ind, el in add_topo.iterrows():
                pp.create_bus(self._grid, index=ind, **el)
        self._init_private_attrs()
        self._aux_run_pf_init()  # run yet another powerflow with the added buses
        
        # do this at the end
        self._in_service_line_col_id = int((self._grid.line.columns == "in_service").nonzero()[0][0])
        self._in_service_trafo_col_id = int((self._grid.trafo.columns == "in_service").nonzero()[0][0])
        self._in_service_storage_cold_id = int((self._grid.storage.columns == "in_service").nonzero()[0][0])
        self.comp_time = 0.
        
    def _aux_run_pf_init(self):
        """run a powerflow when the file is being loaded. This is called three times for each call to "load_grid" """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                self._aux_runpf_pp(False)
                if not self._grid.converged:
                    raise pp.powerflow.LoadflowNotConverged
            except pp.powerflow.LoadflowNotConverged:
                self._aux_runpf_pp(True)
        
    def _init_private_attrs(self) -> None:
        #  number of elements per substation
        self.sub_info = np.zeros(self.n_sub, dtype=dt_int)

        self.load_to_subid = np.zeros(self.n_load, dtype=dt_int)
        self.gen_to_subid = np.zeros(self.n_gen, dtype=dt_int)
        self.line_or_to_subid = np.zeros(self.n_line, dtype=dt_int)
        self.line_ex_to_subid = np.zeros(self.n_line, dtype=dt_int)

        self.load_to_sub_pos = np.zeros(self.n_load, dtype=dt_int)
        self.gen_to_sub_pos = np.zeros(self.n_gen, dtype=dt_int)
        self.line_or_to_sub_pos = np.zeros(self.n_line, dtype=dt_int)
        self.line_ex_to_sub_pos = np.zeros(self.n_line, dtype=dt_int)

        if self.n_storage > 0:
            self.storage_to_subid = np.zeros(self.n_storage, dtype=dt_int)
            self.storage_to_sub_pos = np.zeros(self.n_storage, dtype=dt_int)

        pos_already_used = np.zeros(self.n_sub, dtype=dt_int)
        self._what_object_where = [[] for _ in range(self.n_sub)]

        for i, (_, row) in enumerate(self._grid.line.iterrows()):
            sub_or_id = int(row["from_bus"])
            sub_ex_id = int(row["to_bus"])
            self.sub_info[sub_or_id] += 1
            self.sub_info[sub_ex_id] += 1
            self.line_or_to_subid[i] = sub_or_id
            self.line_ex_to_subid[i] = sub_ex_id

            self.line_or_to_sub_pos[i] = pos_already_used[sub_or_id]
            pos_already_used[sub_or_id] += 1
            self.line_ex_to_sub_pos[i] = pos_already_used[sub_ex_id]
            pos_already_used[sub_ex_id] += 1

            self._what_object_where[sub_or_id].append(("line", "from_bus", i))
            self._what_object_where[sub_ex_id].append(("line", "to_bus", i))

        lag_transfo = self._grid.line.shape[0]
        self._number_true_line = copy.deepcopy(self._grid.line.shape[0])
        for i, (_, row) in enumerate(self._grid.trafo.iterrows()):
            sub_or_id = int(row["hv_bus"])
            sub_ex_id = int(row["lv_bus"])
            self.sub_info[sub_or_id] += 1
            self.sub_info[sub_ex_id] += 1
            self.line_or_to_subid[i + lag_transfo] = sub_or_id
            self.line_ex_to_subid[i + lag_transfo] = sub_ex_id

            self.line_or_to_sub_pos[i + lag_transfo] = pos_already_used[sub_or_id]
            pos_already_used[sub_or_id] += 1
            self.line_ex_to_sub_pos[i + lag_transfo] = pos_already_used[sub_ex_id]
            pos_already_used[sub_ex_id] += 1

            self._what_object_where[sub_or_id].append(("trafo", "hv_bus", i))
            self._what_object_where[sub_ex_id].append(("trafo", "lv_bus", i))

        for i, (_, row) in enumerate(self._grid.gen.iterrows()):
            sub_id = int(row["bus"])
            self.sub_info[sub_id] += 1
            self.gen_to_subid[i] = sub_id
            self.gen_to_sub_pos[i] = pos_already_used[sub_id]
            pos_already_used[sub_id] += 1

            self._what_object_where[sub_id].append(("gen", "bus", i))

        for i, (_, row) in enumerate(self._grid.load.iterrows()):
            sub_id = int(row["bus"])
            self.sub_info[sub_id] += 1
            self.load_to_subid[i] = sub_id
            self.load_to_sub_pos[i] = pos_already_used[sub_id]
            pos_already_used[sub_id] += 1

            self._what_object_where[sub_id].append(("load", "bus", i))

        if self.n_storage > 0:
            for i, (_, row) in enumerate(self._grid.storage.iterrows()):
                sub_id = int(row["bus"])
                self.sub_info[sub_id] += 1
                self.storage_to_subid[i] = sub_id
                self.storage_to_sub_pos[i] = pos_already_used[sub_id]
                pos_already_used[sub_id] += 1

                self._what_object_where[sub_id].append(("storage", "bus", i))

        self.dim_topo = self.sub_info.sum()
        
        # shunts data
        if type(self).shunts_data_available:
            self.shunt_to_subid = np.zeros(self.n_shunt, dtype=dt_int) - 1
            name_shunt = []
            # TODO read name from the grid if provided
            for i, (_, row) in enumerate(self._grid.shunt.iterrows()):
                bus = int(row["bus"])
                name_shunt.append("shunt_{bus}_{index_shunt}".format(**row, index_shunt=i))
                self.shunt_to_subid[i] = bus
            self.name_shunt = np.array(name_shunt).astype(str)
            self._sh_vnkv = self._grid.bus["vn_kv"][self.shunt_to_subid].values.astype(
                dt_float
            )
        
        self._compute_pos_big_topo()

        # utilities for imeplementing apply_action
        self._corresp_name_fun = {}

        self._get_vector_inj = {}
        self._get_vector_inj[
            "load_p"
        ] = self._load_grid_load_p_mw  # lambda grid: grid.load["p_mw"]
        self._get_vector_inj[
            "load_q"
        ] = self._load_grid_load_q_mvar  # lambda grid: grid.load["q_mvar"]
        self._get_vector_inj[
            "prod_p"
        ] = self._load_grid_gen_p_mw  # lambda grid: grid.gen["p_mw"]
        self._get_vector_inj[
            "prod_v"
        ] = self._load_grid_gen_vm_pu  # lambda grid: grid.gen["vm_pu"]

        self.load_pu_to_kv = 1. * self._grid.bus["vn_kv"][self.load_to_subid].values.astype(
            dt_float
        )
        self.prod_pu_to_kv = 1. * self._grid.bus["vn_kv"][self.gen_to_subid].values.astype(
            dt_float
        )
        self.lines_or_pu_to_kv = 1. * self._grid.bus["vn_kv"][
            self.line_or_to_subid
        ].values.astype(dt_float)
        self.lines_ex_pu_to_kv = 1. * self._grid.bus["vn_kv"][
            self.line_ex_to_subid
        ].values.astype(dt_float)
        self.storage_pu_to_kv = 1. * self._grid.bus["vn_kv"][
            self.storage_to_subid
        ].values.astype(dt_float)

        self.thermal_limit_a = 1000. * np.concatenate(
            (
                self._grid.line["max_i_ka"].values,
                self._grid.trafo["sn_mva"].values
                / (np.sqrt(3) * self._grid.trafo["vn_hv_kv"].values),
            )
        )
        self.thermal_limit_a = self.thermal_limit_a.astype(dt_float)

        self.p_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.q_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.v_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.a_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.p_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.q_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.v_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.a_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.line_status = np.full(self.n_line, dtype=dt_bool, fill_value=np.NaN)
        self.load_p = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self.load_q = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self.load_v = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self.prod_p = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self.prod_v = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self.prod_q = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self.storage_p = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)
        self.storage_q = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)
        self.storage_v = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)
        self._nb_bus_before = None

        # store the topoid -> objid
        self._big_topo_to_obj = [(None, None) for _ in range(self.dim_topo)]
        nm_ = "load"
        for load_id, pos_big_topo in enumerate(self.load_pos_topo_vect):
            self._big_topo_to_obj[pos_big_topo] = (load_id, nm_)
        nm_ = "gen"
        for gen_id, pos_big_topo in enumerate(self.gen_pos_topo_vect):
            self._big_topo_to_obj[pos_big_topo] = (gen_id, nm_)
        nm_ = "lineor"
        for l_id, pos_big_topo in enumerate(self.line_or_pos_topo_vect):
            self._big_topo_to_obj[pos_big_topo] = (l_id, nm_)
        nm_ = "lineex"
        for l_id, pos_big_topo in enumerate(self.line_ex_pos_topo_vect):
            self._big_topo_to_obj[pos_big_topo] = (l_id, nm_)

        # store the topoid -> objid
        self._big_topo_to_backend = [(None, None, None) for _ in range(self.dim_topo)]
        for load_id, pos_big_topo in enumerate(self.load_pos_topo_vect):
            self._big_topo_to_backend[pos_big_topo] = (load_id, load_id, 0)
        for gen_id, pos_big_topo in enumerate(self.gen_pos_topo_vect):
            self._big_topo_to_backend[pos_big_topo] = (gen_id, gen_id, 1)
        for l_id, pos_big_topo in enumerate(self.line_or_pos_topo_vect):
            if l_id < self.__nb_powerline:
                self._big_topo_to_backend[pos_big_topo] = (l_id, l_id, 2)
            else:
                self._big_topo_to_backend[pos_big_topo] = (
                    l_id,
                    l_id - self.__nb_powerline,
                    3,
                )
        for l_id, pos_big_topo in enumerate(self.line_ex_pos_topo_vect):
            if l_id < self.__nb_powerline:
                self._big_topo_to_backend[pos_big_topo] = (l_id, l_id, 4)
            else:
                self._big_topo_to_backend[pos_big_topo] = (
                    l_id,
                    l_id - self.__nb_powerline,
                    5,
                )

        self.theta_or = np.full(self.n_line, fill_value=np.NaN, dtype=dt_float)
        self.theta_ex = np.full(self.n_line, fill_value=np.NaN, dtype=dt_float)
        self.load_theta = np.full(self.n_load, fill_value=np.NaN, dtype=dt_float)
        self.gen_theta = np.full(self.n_gen, fill_value=np.NaN, dtype=dt_float)
        self.storage_theta = np.full(self.n_storage, fill_value=np.NaN, dtype=dt_float)

        self._topo_vect = self._get_topo_vect()
        self.tol = 1e-5  # this is NOT the pandapower tolerance !!!! this is used to check if a storage unit
        # produce / absorbs anything

        # Create a deep copy of itself in the initial state
        # Store it under super private attribute
        with warnings.catch_warnings():
            # raised on some versions of pandapower / pandas
            warnings.simplefilter("ignore", FutureWarning)
            self.__pp_backend_initial_grid = copy.deepcopy(
                self._grid
            )  # will be initialized in the "assert_grid_correct"

    def storage_deact_for_backward_comaptibility(self) -> None:
        cls = type(self)
        self.storage_theta = np.full(cls.n_storage, fill_value=np.NaN, dtype=dt_float)
        self.storage_p = np.full(cls.n_storage, dtype=dt_float, fill_value=np.NaN)
        self.storage_q = np.full(cls.n_storage, dtype=dt_float, fill_value=np.NaN)
        self.storage_v = np.full(cls.n_storage, dtype=dt_float, fill_value=np.NaN)
        self._topo_vect = self._get_topo_vect()

    def _convert_id_topo(self, id_big_topo):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        convert an id of the big topo vector into:

        - the id of the object in its "only object" (eg if id_big_topo represents load 2, then it will be 2)
        - the type of object among: "load", "gen", "lineor" and "lineex"

        """
        return self._big_topo_to_obj[id_big_topo]

    def apply_action(self, backendAction: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Specific implementation of the method to apply an action modifying a powergrid in the pandapower format.
        """
        if backendAction is None:
            return
    
        cls = type(self)
        
        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,
            shunts__,
        ) = backendAction()

        # handle bus status
        self._grid.bus["in_service"] = pd.Series(data=active_bus.T.reshape(-1),
                                                 index=np.arange(cls.n_sub * cls.n_busbar_per_sub),
                                                 dtype=bool)
        # TODO n_busbar what if index is not continuous
        
        # handle generators
        tmp_prod_p = self._get_vector_inj["prod_p"](self._grid)
        if (prod_p.changed).any():
            tmp_prod_p.iloc[prod_p.changed] = prod_p.values[prod_p.changed]

        tmp_prod_v = self._get_vector_inj["prod_v"](self._grid)
        if (prod_v.changed).any():
            tmp_prod_v.iloc[prod_v.changed] = (
                prod_v.values[prod_v.changed] / self.prod_pu_to_kv[prod_v.changed]
            )

        if self._id_bus_added is not None and prod_v.changed[self._id_bus_added]:
            # handling of the slack bus, where "2" generators are present.
            self._grid["ext_grid"]["vm_pu"] = 1.0 * tmp_prod_v[self._id_bus_added]

        tmp_load_p = self._get_vector_inj["load_p"](self._grid)
        if (load_p.changed).any():
            tmp_load_p.iloc[load_p.changed] = load_p.values[load_p.changed]

        tmp_load_q = self._get_vector_inj["load_q"](self._grid)
        if (load_q.changed).any():
            tmp_load_q.iloc[load_q.changed] = load_q.values[load_q.changed]

        if cls.n_storage > 0:
            # active setpoint
            tmp_stor_p = self._grid.storage["p_mw"]
            if (storage.changed).any():
                tmp_stor_p.iloc[storage.changed] = storage.values[storage.changed]

            # topology of the storage
            stor_bus = backendAction.get_storages_bus()
            new_bus_num = dt_int(1) * self._grid.storage["bus"].values
            new_bus_id = stor_bus.values[stor_bus.changed]
            new_bus_num[stor_bus.changed] = cls.local_bus_to_global(new_bus_id, cls.storage_to_subid[stor_bus.changed])
            deactivated = new_bus_num <= -1
            deact_and_changed = deactivated & stor_bus.changed
            new_bus_num[deact_and_changed] = cls.storage_to_subid[deact_and_changed]
            self._grid.storage.loc[stor_bus.changed & deactivated, "in_service"] = False
            self._grid.storage.loc[stor_bus.changed & ~deactivated, "in_service"] = True
            self._grid.storage["bus"] = new_bus_num
            self._topo_vect[cls.storage_pos_topo_vect[stor_bus.changed]] = new_bus_id
            self._topo_vect[
                cls.storage_pos_topo_vect[deact_and_changed]
            ] = -1

        if type(backendAction).shunts_data_available:
            shunt_p, shunt_q, shunt_bus = shunts__

            if (shunt_p.changed).any():
                self._grid.shunt["p_mw"].iloc[shunt_p.changed] = shunt_p.values[
                    shunt_p.changed
                ]
            if (shunt_q.changed).any():
                self._grid.shunt["q_mvar"].iloc[shunt_q.changed] = shunt_q.values[
                    shunt_q.changed
                ]
            if (shunt_bus.changed).any():
                sh_service = shunt_bus.values[shunt_bus.changed] != -1
                self._grid.shunt["in_service"].iloc[shunt_bus.changed] = sh_service           
                chg_and_in_service = sh_service & shunt_bus.changed
                self._grid.shunt["bus"].loc[chg_and_in_service] = cls.local_bus_to_global(shunt_bus.values[chg_and_in_service],
                                                                                          cls.shunt_to_subid[chg_and_in_service])

        # i made at least a real change, so i implement it in the backend
        for id_el, new_bus in topo__:
            id_el_backend, id_topo, type_obj = self._big_topo_to_backend[id_el]

            if type_obj is not None:
                # storage unit are handled elsewhere
                self._type_to_bus_set[type_obj](new_bus, id_el_backend, id_topo)

    def _apply_load_bus(self, new_bus, id_el_backend, id_topo):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self._init_bus_load[id_el_backend]
        )
        if new_bus_backend >= 0:
            self._grid.load["bus"].iat[id_el_backend] = new_bus_backend
            self._grid.load["in_service"].iat[id_el_backend] = True
        else:
            self._grid.load["in_service"].iat[id_el_backend] = False
            # self._grid.load["bus"].iat[id_el_backend] = -1  # not needed and cause bugs with certain pandas version

    def _apply_gen_bus(self, new_bus, id_el_backend, id_topo):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self._init_bus_gen[id_el_backend]
        )
        if new_bus_backend >= 0:
            self._grid.gen["bus"].iat[id_el_backend] = new_bus_backend
            self._grid.gen["in_service"].iat[id_el_backend] = True
            # remember in this case slack bus is actually 2 generators for pandapower !
            if (
                id_el_backend == (self._grid.gen.shape[0] - 1)
                and self._iref_slack is not None
            ):
                self._grid.ext_grid["bus"].iat[0] = new_bus_backend
        else:
            self._grid.gen["in_service"].iat[id_el_backend] = False
            # self._grid.gen["bus"].iat[id_el_backend] = -1  # not needed and cause bugs with certain pandas version
            # in this case the slack bus cannot be disconnected

    def _apply_lor_bus(self, new_bus, id_el_backend, id_topo):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self._init_bus_lor[id_el_backend]
        )
        self.change_bus_powerline_or(id_el_backend, new_bus_backend)

    def change_bus_powerline_or(self, id_powerline_backend, new_bus_backend):
        if new_bus_backend >= 0:
            self._grid.line["in_service"].iat[id_powerline_backend] = True
            self._grid.line["from_bus"].iat[id_powerline_backend] = new_bus_backend
        else:
            self._grid.line["in_service"].iat[id_powerline_backend] = False

    def _apply_lex_bus(self, new_bus, id_el_backend, id_topo):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self._init_bus_lex[id_el_backend]
        )
        self.change_bus_powerline_ex(id_el_backend, new_bus_backend)

    def change_bus_powerline_ex(self, id_powerline_backend, new_bus_backend):
        if new_bus_backend >= 0:
            self._grid.line["in_service"].iat[id_powerline_backend] = True
            self._grid.line["to_bus"].iat[id_powerline_backend] = new_bus_backend
        else:
            self._grid.line["in_service"].iat[id_powerline_backend] = False

    def _apply_trafo_hv(self, new_bus, id_el_backend, id_topo):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self._init_bus_lor[id_el_backend]
        )
        self.change_bus_trafo_hv(id_topo, new_bus_backend)

    def change_bus_trafo_hv(self, id_powerline_backend, new_bus_backend):
        if new_bus_backend >= 0:
            self._grid.trafo["in_service"].iat[id_powerline_backend] = True
            self._grid.trafo["hv_bus"].iat[id_powerline_backend] = new_bus_backend
        else:
            self._grid.trafo["in_service"].iat[id_powerline_backend] = False

    def _apply_trafo_lv(self, new_bus, id_el_backend, id_topo):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self._init_bus_lex[id_el_backend]
        )
        self.change_bus_trafo_lv(id_topo, new_bus_backend)

    def change_bus_trafo_lv(self, id_powerline_backend, new_bus_backend):
        if new_bus_backend >= 0:
            self._grid.trafo["in_service"].iat[id_powerline_backend] = True
            self._grid.trafo["lv_bus"].iat[id_powerline_backend] = new_bus_backend
        else:
            self._grid.trafo["in_service"].iat[id_powerline_backend] = False

    def _aux_get_line_info(self, colname1, colname2):
        res = np.concatenate(
            (
                self._grid.res_line[colname1].values,
                self._grid.res_trafo[colname2].values,
            )
        )
        return res

    def _aux_runpf_pp(self, is_dc: bool):
        with warnings.catch_warnings():
            # remove the warning if _grid non connex. And it that case load flow as not converged
            warnings.filterwarnings(
                "ignore", category=scipy.sparse.linalg.MatrixRankWarning
            )
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self._pf_init = "dc"
            # nb_bus = self.get_nb_active_bus()
            # if self._nb_bus_before is None:
            #     self._pf_init = "dc"
            # elif nb_bus == self._nb_bus_before:
            #     self._pf_init = "results"
            # else:
            #     self._pf_init = "auto"

            if (~self._grid.load["in_service"]).any():
                # TODO see if there is a better way here -> do not handle this here, but rather in Backend._next_grid_state
                raise pp.powerflow.LoadflowNotConverged("Disconnected load: for now grid2op cannot handle properly"
                                                        " disconnected load. If you want to disconnect one, say it"
                                                        " consumes 0. instead. Please check loads: "
                                                        f"{(~self._grid.load['in_service'].values).nonzero()[0]}"
                                                        )
            if (~self._grid.gen["in_service"]).any():
                # TODO see if there is a better way here -> do not handle this here, but rather in Backend._next_grid_state
                raise pp.powerflow.LoadflowNotConverged("Disconnected gen: for now grid2op cannot handle properly"
                                                        " disconnected generators. If you want to disconnect one, say it"
                                                        " produces 0. instead. Please check generators: "
                                                        f"{(~self._grid.gen['in_service'].values).nonzero()[0]}"
                                                        )
            try:
                if is_dc:
                    pp.rundcpp(self._grid, check_connectivity=True, init="flat")
                    # if I put check_connectivity=False then the test AAATestBackendAPI.test_22_islanded_grid_make_divergence
                    # does not pass
                    
                    # if dc i start normally next time i call an ac powerflow
                    self._nb_bus_before = None
                else:
                    pp.runpp(
                        self._grid,
                        check_connectivity=False,
                        init=self._pf_init,
                        numba=self.with_numba,
                        lightsim2grid=self._lightsim2grid,
                        max_iteration=self._max_iter,
                        distributed_slack=self._dist_slack,
                    )
            except IndexError as exc_:
                raise pp.powerflow.LoadflowNotConverged(f"Surprising behaviour of pandapower when a bus is not connected to "
                                                        f"anything but present on the bus (with check_connectivity=False). "
                                                        f"Error was {exc_}"
                                                        )
                    
            # stores the computation time
            if "_ppc" in self._grid:
                if "et" in self._grid["_ppc"]:
                    self.comp_time += self._grid["_ppc"]["et"]
            if self._grid.res_gen.isnull().values.any():
                # TODO see if there is a better way here -> do not handle this here, but rather in Backend._next_grid_state
                # sometimes pandapower does not detect divergence and put Nan.
                raise pp.powerflow.LoadflowNotConverged("Divergence due to Nan values in res_gen table (most likely due to "
                                                        "a non connected grid).")
                            
    def runpf(self, is_dc : bool=False) -> Tuple[bool, Union[Exception, None]]:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Run a power flow on the underlying _grid. This implements an optimization of the powerflow
        computation: if the number of
        buses has not changed between two calls, the previous results are re used. This speeds up the computation
        in case of "do nothing" action applied.
        """
        try:
            self._aux_runpf_pp(is_dc)
            cls = type(self)     
            # if a connected bus has a no voltage, it's a divergence (grid was not connected)
            if self._grid.res_bus.loc[self._grid.bus["in_service"]]["va_degree"].isnull().any():
                buses_ko = self._grid.res_bus.loc[self._grid.bus["in_service"]]["va_degree"].isnull()
                buses_ko = buses_ko.values.nonzero()[0]
                raise pp.powerflow.LoadflowNotConverged(f"Isolated bus, check buses {buses_ko} with `env.backend._grid.res_bus.iloc[{buses_ko}, :]`")
                                           
            (
                self.prod_p[:],
                self.prod_q[:],
                self.prod_v[:],
                self.gen_theta[:],
            ) = self._gens_info()
            (
                self.load_p[:],
                self.load_q[:],
                self.load_v[:],
                self.load_theta[:],
            ) = self._loads_info()
            
            if not is_dc:
                if not np.isfinite(self.load_v).all():
                    # TODO see if there is a better way here
                    # some loads are disconnected: it's a game over case!
                    raise pp.powerflow.LoadflowNotConverged(f"Isolated load: check loads {np.isfinite(self.load_v).nonzero()[0]}")
            else:
                # fix voltages magnitude that are always "nan" for dc case
                # self._grid.res_bus["vm_pu"] is always nan when computed in DC
                self.load_v[:] = self.load_pu_to_kv  # TODO
                # need to assign the correct value when a generator is present at the same bus
                # TODO optimize this ugly loop
                # see https://github.com/e2nIEE/pandapower/issues/1996 for a fix
                for l_id in range(cls.n_load):
                    if cls.load_to_subid[l_id] in cls.gen_to_subid:
                        ind_gens = (
                            cls.gen_to_subid == cls.load_to_subid[l_id]
                        ).nonzero()[0]
                        for g_id in ind_gens:
                            if (
                                self._topo_vect[cls.load_pos_topo_vect[l_id]]
                                == self._topo_vect[cls.gen_pos_topo_vect[g_id]]
                            ):
                                self.load_v[l_id] = self.prod_v[g_id]
                                break
                            
            self.line_status[:] = self._get_line_status()
            # I retrieve the data once for the flows, so has to not re read multiple dataFrame
            self.p_or[:] = self._aux_get_line_info("p_from_mw", "p_hv_mw")
            self.q_or[:] = self._aux_get_line_info("q_from_mvar", "q_hv_mvar")
            self.v_or[:] = self._aux_get_line_info("vm_from_pu", "vm_hv_pu")
            self.a_or[:] = self._aux_get_line_info("i_from_ka", "i_hv_ka") * 1000.
            self.theta_or[:] = self._aux_get_line_info(
                "va_from_degree", "va_hv_degree"
            )
            self.a_or[~np.isfinite(self.a_or)] = 0.0
            self.v_or[~np.isfinite(self.v_or)] = 0.0

            self.p_ex[:] = self._aux_get_line_info("p_to_mw", "p_lv_mw")
            self.q_ex[:] = self._aux_get_line_info("q_to_mvar", "q_lv_mvar")
            self.v_ex[:] = self._aux_get_line_info("vm_to_pu", "vm_lv_pu")
            self.a_ex[:] = self._aux_get_line_info("i_to_ka", "i_lv_ka") * 1000.
            self.theta_ex[:] = self._aux_get_line_info(
                "va_to_degree", "va_lv_degree"
            )
            self.a_ex[~np.isfinite(self.a_ex)] = 0.0
            self.v_ex[~np.isfinite(self.v_ex)] = 0.0

            # it seems that pandapower does not take into account disconencted powerline for their voltage
            self.v_or[~self.line_status] = 0.0
            self.v_ex[~self.line_status] = 0.0
            self.v_or[:] *= self.lines_or_pu_to_kv
            self.v_ex[:] *= self.lines_ex_pu_to_kv
            
            # see issue https://github.com/rte-france/Grid2Op/issues/389
            self.theta_or[~np.isfinite(self.theta_or)] = 0.0
            self.theta_ex[~np.isfinite(self.theta_ex)] = 0.0

            self._nb_bus_before = None
            if self._iref_slack is not None:
                # a gen has been added to represent the slack, modeled as an "ext_grid"
                self._grid._ppc["gen"][self._iref_slack, 1] = 0.0

            # handle storage units
            # note that we have to look ourselves for disconnected storage
            (
                self.storage_p[:],
                self.storage_q[:],
                self.storage_v[:],
                self.storage_theta[:],
            ) = self._storages_info()
            deact_storage = ~np.isfinite(self.storage_v)
            if (np.abs(self.storage_p[deact_storage]) > self.tol).any():
                raise pp.powerflow.LoadflowNotConverged(
                    "Isolated storage set to absorb / produce something"
                )
            self.storage_p[deact_storage] = 0.0
            self.storage_q[deact_storage] = 0.0
            self.storage_v[deact_storage] = 0.0
            self._grid.storage["in_service"].values[deact_storage] = False

            self._topo_vect[:] = self._get_topo_vect()
            if not self._grid.converged:
                raise pp.powerflow.LoadflowNotConverged("Divergence without specific reason (self._grid.converged is False)")
            self.div_exception = None
            return True, None

        except pp.powerflow.LoadflowNotConverged as exc_:
            # of the powerflow has not converged, results are Nan
            self.div_exception = exc_
            self._reset_all_nan()
            msg = exc_.__str__()
            return False, BackendError(f'powerflow diverged with error :"{msg}", you can check `env.backend.div_exception` for more information')

    def _reset_all_nan(self) -> None:
        self.p_or[:] = np.NaN
        self.q_or[:] = np.NaN
        self.v_or[:] = np.NaN
        self.a_or[:] = np.NaN
        self.p_ex[:] = np.NaN
        self.q_ex[:] = np.NaN
        self.v_ex[:] = np.NaN
        self.a_ex[:] = np.NaN
        self.prod_p[:] = np.NaN
        self.prod_q[:] = np.NaN
        self.prod_v[:] = np.NaN
        self.load_p[:] = np.NaN
        self.load_q[:] = np.NaN
        self.load_v[:] = np.NaN
        self.storage_p[:] = np.NaN
        self.storage_q[:] = np.NaN
        self.storage_v[:] = np.NaN
        self._nb_bus_before = None

        self.theta_or[:] = np.NaN
        self.theta_ex[:] = np.NaN
        self.load_theta[:] = np.NaN
        self.gen_theta[:] = np.NaN
        self.storage_theta[:] = np.NaN

    def copy(self) -> "PandaPowerBackend":
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This should return a deep copy of the Backend itself and not just the `self._grid`
        """
        res = type(self)(**self._my_kwargs)

        # copy from base class (backend)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # warnings depending on pandas version and pp version
            res._grid = copy.deepcopy(self._grid)
        res.thermal_limit_a = copy.deepcopy(self.thermal_limit_a)
        res._sh_vnkv = copy.deepcopy(self._sh_vnkv)
        res.comp_time = self.comp_time
        res.can_output_theta = self.can_output_theta
        res._is_loaded = self._is_loaded

        # copy all attributes from myself
        res.prod_pu_to_kv = copy.deepcopy(self.prod_pu_to_kv)
        res.load_pu_to_kv = copy.deepcopy(self.load_pu_to_kv)
        res.lines_or_pu_to_kv = copy.deepcopy(self.lines_or_pu_to_kv)
        res.lines_ex_pu_to_kv = copy.deepcopy(self.lines_ex_pu_to_kv)
        res.storage_pu_to_kv = copy.deepcopy(self.storage_pu_to_kv)

        res.p_or = copy.deepcopy(self.p_or)
        res.q_or = copy.deepcopy(self.q_or)
        res.v_or = copy.deepcopy(self.v_or)
        res.a_or = copy.deepcopy(self.a_or)
        res.p_ex = copy.deepcopy(self.p_ex)
        res.q_ex = copy.deepcopy(self.q_ex)
        res.v_ex = copy.deepcopy(self.v_ex)
        res.a_ex = copy.deepcopy(self.a_ex)

        res.load_p = copy.deepcopy(self.load_p)
        res.load_q = copy.deepcopy(self.load_q)
        res.load_v = copy.deepcopy(self.load_v)

        res.storage_p = copy.deepcopy(self.storage_p)
        res.storage_q = copy.deepcopy(self.storage_q)
        res.storage_v = copy.deepcopy(self.storage_v)

        res.prod_p = copy.deepcopy(self.prod_p)
        res.prod_q = copy.deepcopy(self.prod_q)
        res.prod_v = copy.deepcopy(self.prod_v)
        res.line_status = copy.deepcopy(self.line_status)

        res._pf_init = self._pf_init
        res._nb_bus_before = self._nb_bus_before

        res.thermal_limit_a = copy.deepcopy(self.thermal_limit_a)

        res._iref_slack = self._iref_slack
        res._id_bus_added = self._id_bus_added
        res._fact_mult_gen = copy.deepcopy(self._fact_mult_gen)
        res._what_object_where = copy.deepcopy(self._fact_mult_gen)
        res._number_true_line = self._number_true_line
        res._corresp_name_fun = copy.deepcopy(self._corresp_name_fun)
        res.dim_topo = self.dim_topo
        res.cst_1 = self.cst_1
        res._topo_vect = copy.deepcopy(self._topo_vect)
        res.slack_id = self.slack_id

        # function to rstore some information
        res.__nb_bus_before = (
            self.__nb_bus_before
        )  # number of substation in the powergrid
        res.__nb_powerline = (
            self.__nb_powerline
        )  # number of powerline (real powerline, not transformer)
        res._init_bus_load = copy.deepcopy(self._init_bus_load)
        res._init_bus_gen = copy.deepcopy(self._init_bus_gen)
        res._init_bus_lor = copy.deepcopy(self._init_bus_lor)
        res._init_bus_lex = copy.deepcopy(self._init_bus_lex)
        res._get_vector_inj = copy.deepcopy(self._get_vector_inj)
        res._big_topo_to_obj = copy.deepcopy(self._big_topo_to_obj)
        res._big_topo_to_backend = copy.deepcopy(self._big_topo_to_backend)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  
            res.__pp_backend_initial_grid = copy.deepcopy(self.__pp_backend_initial_grid)
        
        # this is NOT the pandapower tolerance !!!! this is used to check if a storage unit
        # produce / absorbs anything
        res.tol = self.tol

        # TODO storage doc (in grid2op rst) of the backend
        res.can_output_theta = self.can_output_theta  # I support the voltage angle
        res.theta_or = copy.deepcopy(self.theta_or)
        res.theta_ex = copy.deepcopy(self.theta_ex)
        res.load_theta = copy.deepcopy(self.load_theta)
        res.gen_theta = copy.deepcopy(self.gen_theta)
        res.storage_theta = copy.deepcopy(self.storage_theta)
        
        res._in_service_line_col_id = self._in_service_line_col_id
        res._in_service_trafo_col_id = self._in_service_trafo_col_id
        
        res._missing_two_busbars_support_info = self._missing_two_busbars_support_info
        res.div_exception = self.div_exception
        return res

    def close(self) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Called when the :class:`grid2op;Environment` has terminated, this function only reset the grid to a state
        where it has not been loaded.
        """
        del self._grid
        self._grid = None
        del self.__pp_backend_initial_grid
        self.__pp_backend_initial_grid = None

    def save_file(self, full_path: Union[os.PathLike, str]) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        You might want to use it for debugging purpose only, and only if you develop yourself a backend.

        Save the file to json.
        :param full_path:
        :return:
        """
        pp.to_json(self._grid, full_path)

    def get_line_status(self) -> np.ndarray:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        As all the functions related to powerline, pandapower split them into multiple dataframe (some for transformers,
        some for 3 winding transformers etc.). We make sure to get them all here.
        """
        return self.line_status

    def _get_line_status(self):
        return np.concatenate(
            (
                self._grid.line["in_service"].values,
                self._grid.trafo["in_service"].values,
            )
        ).astype(dt_bool)

    def get_line_flow(self) -> np.ndarray:
        return self.a_or

    def _disconnect_line(self, id_):
        if id_ < self._number_true_line:
            self._grid.line.iloc[id_, self._in_service_line_col_id] = False
        else:
            self._grid.trafo.iloc[id_ - self._number_true_line, self._in_service_trafo_col_id]  = False
        self._topo_vect[self.line_or_pos_topo_vect[id_]] = -1
        self._topo_vect[self.line_ex_pos_topo_vect[id_]] = -1
        self.line_status[id_] = False

    def _reconnect_line(self, id_):
        if id_ < self._number_true_line:
            self._grid.line.iloc[id_, self._in_service_line_col_id] = True
        else:
            self._grid.trafo.iloc[id_ - self._number_true_line, self._in_service_trafo_col_id] = True
        self.line_status[id_] = True

    def get_topo_vect(self) -> np.ndarray:
        return self._topo_vect

    def _get_topo_vect(self):
        cls = type(self)
        res = np.full(cls.dim_topo, fill_value=np.iinfo(dt_int).max, dtype=dt_int)

        # lines / trafo
        line_status = self.get_line_status()
        glob_bus_or = np.concatenate((self._grid.line["from_bus"].values, self._grid.trafo["hv_bus"].values))
        res[cls.line_or_pos_topo_vect] = cls.global_bus_to_local(glob_bus_or, cls.line_or_to_subid)
        res[cls.line_or_pos_topo_vect[~line_status]] = -1
        glob_bus_ex = np.concatenate((self._grid.line["to_bus"].values, self._grid.trafo["lv_bus"].values))
        res[cls.line_ex_pos_topo_vect] = cls.global_bus_to_local(glob_bus_ex, cls.line_ex_to_subid)
        res[cls.line_ex_pos_topo_vect[~line_status]] = -1
        # load, gen
        load_status = self._grid.load["in_service"].values
        res[cls.load_pos_topo_vect] = cls.global_bus_to_local(self._grid.load["bus"].values, cls.load_to_subid)
        res[cls.load_pos_topo_vect[~load_status]] = -1
        gen_status = self._grid.gen["in_service"].values
        res[cls.gen_pos_topo_vect] = cls.global_bus_to_local(self._grid.gen["bus"].values, cls.gen_to_subid)
        res[cls.gen_pos_topo_vect[~gen_status]] = -1
        # storage
        if cls.n_storage:
            storage_status = self._grid.storage["in_service"].values
            res[cls.storage_pos_topo_vect] = cls.global_bus_to_local(self._grid.storage["bus"].values, cls.storage_to_subid)
            res[cls.storage_pos_topo_vect[~storage_status]] = -1
        return res

    def _gens_info(self):
        prod_p = self.cst_1 * self._grid.res_gen["p_mw"].values.astype(dt_float)
        prod_q = self.cst_1 * self._grid.res_gen["q_mvar"].values.astype(dt_float)
        prod_v = (
            self.cst_1
            * self._grid.res_gen["vm_pu"].values.astype(dt_float)
            * self.prod_pu_to_kv
        )
        prod_theta = self.cst_1 * self._grid.res_gen["va_degree"].values.astype(
            dt_float
        )
        if self._iref_slack is not None:
            # slack bus and added generator are on same bus. I need to add power of slack bus to this one.

            # if self._grid.gen["bus"].iloc[self._id_bus_added] == self.gen_to_subid[self._id_bus_added]:
            if "gen" in self._grid._ppc["internal"]:
                prod_p[self._id_bus_added] += self._grid._ppc["internal"]["gen"][
                    self._iref_slack, 1
                ]
                prod_q[self._id_bus_added] += self._grid._ppc["internal"]["gen"][
                    self._iref_slack, 2
                ]
        return prod_p, prod_q, prod_v, prod_theta

    def _loads_info(self):
        load_p = self.cst_1 * self._grid.res_load["p_mw"].values.astype(dt_float)
        load_q = self.cst_1 * self._grid.res_load["q_mvar"].values.astype(dt_float)
        load_v = (
            self._grid.res_bus.loc[self._grid.load["bus"].values][
                "vm_pu"
            ].values.astype(dt_float)
            * self.load_pu_to_kv
        )
        load_theta = self._grid.res_bus.loc[self._grid.load["bus"].values][
            "va_degree"
        ].values.astype(dt_float)
        return load_p, load_q, load_v, load_theta

    def generators_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.cst_1 * self.prod_p,
            self.cst_1 * self.prod_q,
            self.cst_1 * self.prod_v,
        )

    def loads_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.cst_1 * self.load_p,
            self.cst_1 * self.load_q,
            self.cst_1 * self.load_v,
        )

    def lines_or_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.cst_1 * self.p_or,
            self.cst_1 * self.q_or,
            self.cst_1 * self.v_or,
            self.cst_1 * self.a_or,
        )

    def lines_ex_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.cst_1 * self.p_ex,
            self.cst_1 * self.q_ex,
            self.cst_1 * self.v_ex,
            self.cst_1 * self.a_ex,
        )

    def shunt_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shunt_p = self.cst_1 * self._grid.res_shunt["p_mw"].values.astype(dt_float)
        shunt_q = self.cst_1 * self._grid.res_shunt["q_mvar"].values.astype(dt_float)
        shunt_v = (
            self._grid.res_bus["vm_pu"]
            .loc[self._grid.shunt["bus"].values]
            .values.astype(dt_float)
        )
        shunt_v *= (
            self._grid.bus["vn_kv"]
            .loc[self._grid.shunt["bus"].values]
            .values.astype(dt_float)
        )
        shunt_bus = type(self).global_bus_to_local(self._grid.shunt["bus"].values, self.shunt_to_subid)
        shunt_v[~self._grid.shunt["in_service"].values] = 0.
        shunt_bus[~self._grid.shunt["in_service"].values] = -1
        return shunt_p, shunt_q, shunt_v, shunt_bus

    def storages_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.cst_1 * self.storage_p,
            self.cst_1 * self.storage_q,
            self.cst_1 * self.storage_v,
        )

    def _storages_info(self):
        if self.n_storage:
            # this is because we support "backward comaptibility" feature. So the storage can be
            # deactivated from the Environment...
            p_storage = self._grid.res_storage["p_mw"].values.astype(dt_float)
            q_storage = self._grid.res_storage["q_mvar"].values.astype(dt_float)
            v_storage = (
                self._grid.res_bus.loc[self._grid.storage["bus"].values][
                    "vm_pu"
                ].values.astype(dt_float)
                * self.storage_pu_to_kv
            )
            theta_storage = (
                self._grid.res_bus.loc[self._grid.storage["bus"].values][
                    "vm_pu"
                ].values.astype(dt_float)
                * self.storage_pu_to_kv
            )
            v_storage[~self._grid.storage["in_service"].values] = 0.
        else:
            p_storage = np.zeros(shape=0, dtype=dt_float)
            q_storage = np.zeros(shape=0, dtype=dt_float)
            v_storage = np.zeros(shape=0, dtype=dt_float)
            theta_storage = np.zeros(shape=0, dtype=dt_float)
        return p_storage, q_storage, v_storage, theta_storage

    def sub_from_bus_id(self, bus_id : int) -> int:
        if bus_id >= self._number_true_line:
            return bus_id - self._number_true_line
        return bus_id
