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
from typing import Optional, Tuple, Union

import pandapower as pp
import scipy

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Backend.backend import Backend
from grid2op.Exceptions import *


class EducPandaPowerBackend(Backend):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This class does not work.

        It is mainly presented for educational purpose as example on coding your own backend.

        It is derived from PandaPowerBackend, but without all the "optimization" that could make the
        resulting backend harder to read.

    This module presents an example of an implementation of a `grid2op.Backend` when using the powerflow
    implementation "pandapower" available at `PandaPower <https://www.pandapower.org/>`_ for more details about
    this backend. This file is provided as an example of a proper :class:`grid2op.Backend.Backend` implementation.

    This backend currently does not work with 3 winding transformers and other exotic object.

    As explained in the `grid2op.Backend` module, every module must inherit the `grid2op.Backend` class.

    We illustrate here how to set up a backend with what we think is "rather standard" in the powersystem
    eco system.

    Please consult the documentation at :ref:`create-backend-module` for more information.

    You have at your disposal:

    - a tool that is able to compute power flows from a given grid in a given format (in this case call
      `pandapower.runpf(pandapower_grid)`)
    - a tool that is able to load a powergrid from a file store on the hard drive.

    We try to find a good compromise between the size of the code (for clarity) and the "closeness to a working
    code".

    For a complete working example, relatively optimized (but much less readable) please have a look at the
    real :class:`grid2op.Backend.PandaPowerBackend` class.

    """

    def __init__(self,
                 detailed_infos_for_cascading_failures : Optional[bool]=False,
                 can_be_copied : Optional[bool]=True):
        """
        Nothing much to do here except initializing what you would need (a tensorflow session, link to some
        external dependencies etc.)

        Nothing much for this example.

        Parameters
        ----------
        detailed_infos_for_cascading_failures: ``bool``
            See the documentation of :class:`grid2op.Backend.Backend.__init__` for more information
        """
        Backend.__init__(
            self,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
            can_be_copied=can_be_copied,
            # extra arguments that might be needed for building such a backend 
            # these extra kwargs will be stored (without copy) in the
            # base class and used when another backend will be created
            # for example in the Runner class.
        )
        warnings.warn(
            "This backend is used for demonstration purpose only, you should not use it under any "
            "circumstances. Please use grid2op.Backend.PandaPowerBackend instead"
        )
        self._nb_real_line_pandapower : int = None

        # NB: this instance of backend is here for academic purpose only. For clarity, it does not handle
        # neither shunt nor storage unit.

    ####### load the grid
    def load_grid(self,
                  path : Union[os.PathLike, str],
                  filename : Optional[Union[os.PathLike, str]]=None) -> None:
        """
        Demonstration on how you can load a powergrid and then initialize the proper grid2op attributes.

        The only thing we have to do here is to "order" the objects in each substation. Note that we don't even do it
        implicitly here (relying on default grid2op ordering).

        The only decision we had to make was concerning "grid2op powerlines" which represents both
        "pandapower transformers"
        and "pandapower powerlines".

        We decided that:

        - powerline are "before" trafo (so in the grid2op line id I will put first all powerlines, then all trafo)
        - grid2op "origin" side will be "from" side for pandapower powerline and "hv" side for pandapower trafo
        - grid2op "extremity" side will be "to" side for pandapower powerline and "lv" side for pandapower trafo


        .. note:: We use one "trick" here. Pandapower grid (as it will be the case for most format) will have one "bus"
            per substation. For grid2op, we want at least 2 busbar per substation. So we simply copy and paste the grid.

            And we will deactivate the busbar that are not connected (no element connected to it).

            This "coding" allows for easily mapping the bus id (each bus is represented with an id in pandapower)
            and whether its busbar 1 or busbar 2 (grid2op side). More precisely: busbar 1 of substation with
            id `sub_id` will have id `sub_id` and busbar 2 of the same substation will have id `sub_id + n_sub`
            (recall that n_sub is the number of substation on the grid).

            This "coding" is not optimal in the ram it takes. But we recommend you to adopt a similar one. It's
            pretty easy to change the topology using this trick, much easier than if you rely on "switches" for
            example. (But of course  you can still use switches if you really want to)

        """

        # first, handles different kind of path:
        if path is None and filename is None:
            raise RuntimeError(
                "You must provide at least one of path or file to load a powergrid."
            )
        if path is None:
            full_path = filename
        elif filename is None:
            full_path = path
        else:
            full_path = os.path.join(path, filename)
        if not os.path.exists(full_path):
            raise RuntimeError('There is no powergrid at "{}"'.format(full_path))

        # then load the grid located at the full path and store it in `self._grid`
        # raise an exception if it can't be loaded
        try:
            with warnings.catch_warnings():
                # remove deprecationg warnings for old version of pandapower
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                self._grid = pp.from_json(full_path)
        except Exception as exc_:
            raise BackendError(
                f'Impossible to load the powergrid located at "{full_path}". Please '
                f"check the file exist and that the file represent a valid pandapower "
                f"grid. For your information, the error is:\n{exc_}"
            )

        ######################################################################
        # this part is due to the "modeling" of the topology FOR THIS EXAMPLE
        # remember (see docstring of this function) that we "duplicate the buses" to code more easily the
        # topology modification (instead of relying on the `switches`)

        # first we remember the number of substation
        self.n_sub = self._grid.bus.shape[0]
        # and then we duplicate the bus
        add_topo = copy.deepcopy(self._grid.bus)
        add_topo.index += add_topo.shape[0]
        add_topo["in_service"] = False
        self._grid.bus = pd.concat((self._grid.bus, add_topo))
        self._nb_real_line_pandapower = self._grid.line.shape[0]
        # i do a powerflow to initialize the "results" dataframes
        # this last step is internal to pandapower
        pp.runpp(self._grid, check_connectivity=False)
        ######################################################################

        # and now we initialize the number of each of the elements
        self.n_line = (
            self._grid.line.shape[0] + self._grid.trafo.shape[0]
        )  # trafo are powerline for grid2op !
        self.n_gen = self._grid.gen.shape[0]
        self.n_load = self._grid.load.shape[0]
        # self.n_sub  # already initialize above

        # initialize the number of elements per substation
        # now export to grid2op the substation to which objects are connected
        self.load_to_subid = copy.deepcopy(self._grid.load["bus"])
        self.gen_to_subid = copy.deepcopy(self._grid.gen["bus"])
        # here we just decide (but that is a convention we could have done it differently)
        # that "origin side" (grid2op) corresponds to "from_bus" from pandapower line and "hv_bus" for
        # pandapower trafo.
        self.line_or_to_subid = np.concatenate(
            (
                copy.deepcopy(self._grid.line["from_bus"]),
                copy.deepcopy(self._grid.trafo["hv_bus"]),
            )
        )
        self.line_ex_to_subid = np.concatenate(
            (
                copy.deepcopy(self._grid.line["to_bus"]),
                copy.deepcopy(self._grid.trafo["lv_bus"]),
            )
        )

        # and now we don't forget to initialize the rest
        self._compute_pos_big_topo()  # we highly recommend you to call this !

        # and now the thermal limit
        self.thermal_limit_a = 1000. * np.concatenate(
            (
                self._grid.line["max_i_ka"].values,
                self._grid.trafo["sn_mva"].values
                / (np.sqrt(3) * self._grid.trafo["vn_hv_kv"].values),
            )
        )
        self.thermal_limit_a = self.thermal_limit_a.astype(dt_float)

        # NB: this instance of backend is here for academic purpose only. For clarity, it does not handle
        # neither shunt nor storage unit.
        type(self).shunts_data_available = False
        type(self).set_no_storage()

    ###### modify the grid
    def apply_action(self, backendAction: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:
        """
        Here the implementation of the "modify the grid" function.

        From the documentation, it's pretty straightforward, even though the implementation takes ~70 lines of code.
        Most of them being direct copy paste from the examples in the documentation.
        """
        if backendAction is None:
            return

        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            _,
            shunts__,
        ) = backendAction()

        for gen_id, new_p in prod_p:
            self._grid.gen["p_mw"].iloc[gen_id] = new_p
        for gen_id, new_v in prod_v:
            self._grid.gen["vm_pu"].iloc[gen_id] = new_v  # but new_v is not pu !
            self._grid.gen["vm_pu"].iloc[gen_id] /= self._grid.bus["vn_kv"][
                self.gen_to_subid[gen_id]
            ]  # now it is :-)

        for load_id, new_p in load_p:
            self._grid.load["p_mw"].iloc[load_id] = new_p
        for load_id, new_q in load_q:
            self._grid.load["q_mvar"].iloc[load_id] = new_q

        # now i deal with the topology
        loads_bus = backendAction.get_loads_bus()
        for load_id, new_bus in loads_bus:
            if new_bus == -1:
                self._grid.load["in_service"][load_id] = False
            else:
                self._grid.load["in_service"][load_id] = True
                # this formula is really convenient because we decided to duplicated buses in each substation.
                # and decided that: bus 1 of a substation with id `sub_id` will have id `sub_id` and
                # bus 2 of the same substation will have id `sub_id + n_substation`
                self._grid.load["bus"][load_id] = (
                    self.load_to_subid[load_id] + (new_bus - 1) * self.n_sub
                )

        gens_bus = backendAction.get_gens_bus()
        for gen_id, new_bus in gens_bus:
            if new_bus == -1:
                self._grid.gen["in_service"][gen_id] = False
            else:
                self._grid.gen["in_service"][gen_id] = True
                self._grid.gen["bus"][gen_id] = (
                    self.gen_to_subid[gen_id] + (new_bus - 1) * self.n_sub
                )

        lines_or_bus = backendAction.get_lines_or_bus()
        for line_id, new_bus in lines_or_bus:
            if line_id < self._nb_real_line_pandapower:
                dt = self._grid.line
                key = "from_bus"
                line_id_db = line_id
            else:
                dt = self._grid.trafo
                key = "hv_bus"
                line_id_db = line_id - self._nb_real_line_pandapower

            if new_bus == -1:
                dt["in_service"][line_id_db] = False
            else:
                dt["in_service"][line_id_db] = True
                dt[key][line_id_db] = (
                    self.line_or_to_subid[line_id] + (new_bus - 1) * self.n_sub
                )

        lines_ex_bus = backendAction.get_lines_ex_bus()
        for line_id, new_bus in lines_ex_bus:
            if line_id < self._nb_real_line_pandapower:
                dt = self._grid.line
                key = "to_bus"
                line_id_db = line_id
            else:
                dt = self._grid.trafo
                key = "lv_bus"
                line_id_db = line_id - self._nb_real_line_pandapower

            if new_bus == -1:
                dt["in_service"][line_id_db] = False
            else:
                dt["in_service"][line_id_db] = True
                dt[key][line_id_db] = (
                    self.line_ex_to_subid[line_id] + (new_bus - 1) * self.n_sub
                )

        # spec
        bus_is = self._grid.bus["in_service"]
        for i, (bus1_status, bus2_status) in enumerate(active_bus):
            bus_is[i] = bus1_status  # no iloc for bus, don't ask me why please :-/
            bus_is[i + self.n_sub] = bus2_status

    ###### computes powerflow
    def runpf(self, is_dc : bool=False) -> Tuple[bool, Union[Exception, None]]:
        """
        Now we just perform a powerflow with pandapower which can be done with either `rundcpp` for dc powerflow
        or with `runpp` for AC powerflow.

        This is really a 2 lines code. It is a bit more verbose here because we took care of silencing some
        warnings and try / catch some possible exceptions.

        """
        try:
            with warnings.catch_warnings():
                # remove the warning if _grid non connex. And it that case load flow as not converged
                warnings.filterwarnings(
                    "ignore", category=scipy.sparse.linalg.MatrixRankWarning
                )
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                if is_dc:
                    pp.rundcpp(self._grid, check_connectivity=False)
                else:
                    pp.runpp(self._grid, check_connectivity=False)
                return self._grid.converged, None
        except pp.powerflow.LoadflowNotConverged as exc_:
            # of the powerflow has not converged, results are Nan
            return False, exc_

    ###### getters
    def get_topo_vect(self) -> np.ndarray:
        """
        Retrieve the bus to which the objects are connected based on the information stored on the grid.

        This is fairly simple, again, because we choose to explicitly represents 2 buses per substation.

        Function is verbose (~40 lines of code), but pretty straightforward.
        """
        res = np.full(self.dim_topo, fill_value=np.NaN, dtype=dt_int)

        line_status = self.get_line_status()

        i = 0
        for row in self._grid.line[["from_bus", "to_bus"]].values:
            bus_or_id = row[0]
            bus_ex_id = row[1]
            if line_status[i]:
                res[self.line_or_pos_topo_vect[i]] = (
                    1 if bus_or_id == self.line_or_to_subid[i] else 2
                )
                res[self.line_ex_pos_topo_vect[i]] = (
                    1 if bus_ex_id == self.line_ex_to_subid[i] else 2
                )
            else:
                res[self.line_or_pos_topo_vect[i]] = -1
                res[self.line_ex_pos_topo_vect[i]] = -1
            i += 1

        nb = self._nb_real_line_pandapower
        i = 0
        for row in self._grid.trafo[["hv_bus", "lv_bus"]].values:
            bus_or_id = row[0]
            bus_ex_id = row[1]

            j = i + nb
            if line_status[j]:
                res[self.line_or_pos_topo_vect[j]] = (
                    1 if bus_or_id == self.line_or_to_subid[j] else 2
                )
                res[self.line_ex_pos_topo_vect[j]] = (
                    1 if bus_ex_id == self.line_ex_to_subid[j] else 2
                )
            else:
                res[self.line_or_pos_topo_vect[j]] = -1
                res[self.line_ex_pos_topo_vect[j]] = -1
            i += 1

        i = 0
        for bus_id in self._grid.gen["bus"].values:
            res[self.gen_pos_topo_vect[i]] = 1 if bus_id == self.gen_to_subid[i] else 2
            i += 1

        i = 0
        for bus_id in self._grid.load["bus"].values:
            res[self.load_pos_topo_vect[i]] = (
                1 if bus_id == self.load_to_subid[i] else 2
            )
            i += 1
        return res

    def generators_info(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        We chose to keep the same order in grid2op and in pandapower. So we just return the correct values.
        """
        # carefull with copy / deep copy
        prod_p = self._grid.res_gen["p_mw"].values.astype(dt_float)
        prod_q = self._grid.res_gen["q_mvar"].values.astype(dt_float)
        prod_v = self._grid.res_gen["vm_pu"].values.astype(dt_float)  # in pu
        prod_v *= (
            self._grid.bus["vn_kv"].iloc[self.gen_to_subid].values.astype(dt_float)
        )  # in kV
        return prod_p, prod_q, prod_v

    def loads_info(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        We chose to keep the same order in grid2op and in pandapower. So we just return the correct values.
        """
        # carefull with copy / deep copy
        load_p = self._grid.res_load["p_mw"].values.astype(dt_float)
        load_q = self._grid.res_load["q_mvar"].values.astype(dt_float)
        load_v = self._grid.res_bus.loc[self._grid.load["bus"].values][
            "vm_pu"
        ].values.astype(
            dt_float
        )  # in pu
        load_v *= self._grid.bus.loc[self._grid.load["bus"].values][
            "vn_kv"
        ].values.astype(
            dt_float
        )  # in kV
        return load_p, load_q, load_v

    def _aux_get_line_info(self, colname_powerline, colname_trafo) -> np.ndarray:
        """
        concatenate the information of powerlines and trafo using the convention that "powerlines go first"
        """
        res = np.concatenate(
            (
                self._grid.res_line[colname_powerline].values,
                self._grid.res_trafo[colname_trafo].values,
            )
        )
        return res

    def lines_or_info(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Main method to retrieve the information at the "origin" side of the powerlines and transformers.

        We simply need to follow the convention we adopted:

        - origin side (grid2op) will be "from" side for pandapower powerline
        - origin side (grid2op) will be "hv" side for pandapower trafo
        - we chose to first have powerlines, then transformers

        (convention chosen in :func:`EducPandaPowerBackend.load_grid`)

        """
        p_or = self._aux_get_line_info("p_from_mw", "p_hv_mw")
        q_or = self._aux_get_line_info("q_from_mvar", "q_hv_mvar")
        v_or = self._aux_get_line_info("vm_from_pu", "vm_hv_pu")
        a_or = self._aux_get_line_info("i_from_ka", "i_hv_ka") * 1000
        return p_or, q_or, v_or, a_or

    def lines_ex_info(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Main method to retrieve the information at the "extremity" side of the powerlines and transformers.

        We simply need to follow the convention we adopted:

        - extremity side (grid2op) will be "to" side for pandapower powerline
        - extremity side (grid2op) will be "lv" side for pandapower trafo
        - we chose to first have powerlines, then transformers

        (convention chosen in function :func:`EducPandaPowerBackend.load_grid`)

        """
        p_ex = self._aux_get_line_info("p_to_mw", "p_lv_mw")
        q_ex = self._aux_get_line_info("q_to_mvar", "q_lv_mvar")
        v_ex = self._aux_get_line_info("vm_to_pu", "vm_lv_pu")
        a_ex = self._aux_get_line_info("i_to_ka", "i_lv_ka") * 1000
        return p_ex, q_ex, v_ex, a_ex

    # other less important method that you will need to implement
    def get_line_status(self) -> np.ndarray:
        """
        you might consider implementing it

        .. warning::  /!\\\\ This is a not a "main method" but you might want to implement
            it for a new backend (default implementation most likely not efficient at all). /!\\\\
        """
        return np.concatenate(
            (
                self._grid.line["in_service"].values,
                self._grid.trafo["in_service"].values,
            )
        ).astype(dt_bool)

    def _disconnect_line(self, id_ : int) -> None:
        """
        you might consider implementing it

        .. warning:: /!\\\\ This is a not a "main method" but you might want to implement
            it for a new backend (default implementation most likely not efficient at all). /!\\\\
        """
        if id_ < self._nb_real_line_pandapower:
            self._grid.line["in_service"].iloc[id_] = False
        else:
            self._grid.trafo["in_service"].iloc[
                id_ - self._nb_real_line_pandapower
            ] = False

    def copy(self) -> "EducPandaPowerBackend":
        """
        you might consider implementing it

        .. warning:: /!\\\\ This is a not a "main method" but you might want to implement
            it for a new backend (default implementation most likely not efficient at all). /!\\\\

        Nothing really crazy here

        Performs a deep copy of the power :attr:`_grid`.
        As pandapower is pure python, the deep copy operator is perfectly suited for the task.
        """
        res = copy.deepcopy(self)
        return res

    def reset(self,
              path : Union[os.PathLike, str],
              grid_filename : Optional[Union[os.PathLike, str]]=None) -> None:
        """
        you might consider implementing it

        .. warning:: /!\\\\ This is a not a "main method" but you might want to implement
            it for a new backend (default implementation most likely not efficient at all). /!\\\\

        Reset the grid to the original state
        """

        # set everything to its proper bus (this is because we used a specific way to represent
        # the topology for this example by chosing not to use swtiches, but to double the number of
        # buses per "substation"
        self._grid.line["from_bus"].iloc[:] = self.line_or_to_subid[
            : self._nb_real_line_pandapower
        ]
        self._grid.trafo["hv_bus"].iloc[:] = self.line_or_to_subid[
            self._nb_real_line_pandapower :
        ]
        self._grid.line["to_bus"].iloc[:] = self.line_ex_to_subid[
            : self._nb_real_line_pandapower
        ]
        self._grid.trafo["lv_bus"].iloc[:] = self.line_ex_to_subid[
            self._nb_real_line_pandapower :
        ]
        self._grid.load["bus"].iloc[:] = self.load_to_subid
        self._grid.gen["bus"].iloc[:] = self.gen_to_subid

        # originally everything is in service
        self._grid.line["in_service"].iloc[:] = True
        self._grid.trafo["in_service"].iloc[:] = True
        self._grid.load["in_service"].iloc[:] = True
        self._grid.gen["in_service"].iloc[:] = True

        self._grid.bus["in_service"].iloc[: self.n_sub] = True
        self._grid.bus["in_service"].iloc[self.n_sub :] = False

    def close(self) -> None:
        """
        you might consider implementing it

        .. warning:: /!\\\\ This is a not a "main method" but you might want to implement
            it for a new backend (default implementation most likely not efficient at all). /!\\\\

        Called when the :class:`grid2op;Environment` has terminated, this function only reset the grid to a state
        where it has not been loaded.
        """
        del self._grid
        self._grid = None
