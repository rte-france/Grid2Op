# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import copy
import datetime
import numpy as np
from abc import abstractmethod

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import *
from grid2op.Space import GridObjects

# TODO make an action with the difference between the observation that would be an action.
# TODO have a method that could do "forecast" by giving the _injection by the agent, if he wants to make custom forecasts

# TODO fix "bug" when action not initalized, return nan in to_vect


class BaseObservation(GridObjects):
    """
    Basic class representing an observation.

    All observation must derive from this class and implement all its abstract methods.

    Attributes
    ----------
    action_helper: :class:`grid2op.Action.ActionSpace`
        A representation of the possible action space.

    year: ``int``
        The current year

    month: ``int``
        The current month (0 = january, 11 = december)

    day: ``int``
        The current day of the month

    hour_of_day: ``int``
        The current hour of the day

    minute_of_hour: ``int``
        The current minute of the current hour

    day_of_week: ``int``
        The current day of the week. Monday = 0, Sunday = 6

    prod_p: :class:`numpy.ndarray`, dtype:float
        The active production value of each generator (expressed in MW).

    prod_q: :class:`numpy.ndarray`, dtype:float
        The reactive production value of each generator (expressed in MVar).

    prod_v: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude of the bus to which each generator is connected (expressed in kV).

    load_p: :class:`numpy.ndarray`, dtype:float
        The active load value of each consumption (expressed in MW).

    load_q: :class:`numpy.ndarray`, dtype:float
        The reactive load value of each consumption (expressed in MVar).

    load_v: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude of the bus to which each consumption is connected (expressed in kV).

    p_or: :class:`numpy.ndarray`, dtype:float
        The active power flow at the origin end of each powerline (expressed in MW).

    q_or: :class:`numpy.ndarray`, dtype:float
        The reactive power flow at the origin end of each powerline (expressed in MVar).

    v_or: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude at the bus to which the origin end of each powerline is connected (expressed in kV).

    a_or: :class:`numpy.ndarray`, dtype:float
        The current flow at the origin end of each powerline (expressed in A).

    p_ex: :class:`numpy.ndarray`, dtype:float
        The active power flow at the extremity end of each powerline (expressed in MW).

    q_ex: :class:`numpy.ndarray`, dtype:float
        The reactive power flow at the extremity end of each powerline (expressed in MVar).

    v_ex: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude at the bus to which the extremity end of each powerline is connected (expressed in kV).

    a_ex: :class:`numpy.ndarray`, dtype:float
        The current flow at the extremity end of each powerline (expressed in A).

    rho: :class:`numpy.ndarray`, dtype:float
        The capacity of each powerline. It is defined at the observed current flow divided by the thermal limit of each
        powerline (no unit)

    connectivity_matrix_: :class:`numpy.ndarray`, dtype:float
        The connectivityt matrix (if computed, or None) see definition of :func:`connectivity_matrix` for
        more information

    bus_connectivity_matrix_: :class:`numpy.ndarray`, dtype:float
        The `bus_connectivity_matrix_` matrix (if computed, or None) see definition of
          :func:`BaseObservation.bus_connectivity_matrix` for more information

    vectorized: :class:`numpy.ndarray`, dtype:float
        The vector representation of this BaseObservation (if computed, or None) see definition of
        :func:`to_vect` for more information.

    topo_vect:  :class:`numpy.ndarray`, dtype:int
        For each object (load, generator, ends of a powerline) it gives on which bus this object is connected
        in its substation. See :func:`grid2op.Backend.Backend.get_topo_vect` for more information.

    line_status: :class:`numpy.ndarray`, dtype:bool
        Gives the status (connected / disconnected) for every powerline (``True`` at position `i` means the powerline
        `i` is connected)

    timestep_overflow: :class:`numpy.ndarray`, dtype:int
        Gives the number of time steps since a powerline is in overflow.

    time_before_cooldown_line:  :class:`numpy.ndarray`, dtype:int
        For each powerline, it gives the number of time step the powerline is unavailable due to "cooldown"
        (see :attr:`grid2op.Parameters.NB_TIMESTEP_COOLDOWN_LINE` for more information). 0 means the
        an action will be able to act on this same powerline, a number > 0 (eg 1) means that an action at this time step
        cannot act on this powerline (in the example the agent have to wait 1 time step)

    time_before_cooldown_sub: :class:`numpy.ndarray`, dtype:int
        Same as :attr:`BaseObservation.time_before_cooldown_line` but for substations. For each substation, it gives the
        number of timesteps to wait before acting on this substation (see
        see :attr:`grid2op.Parameters.NB_TIMESTEP_COOLDOWN_SUB` for more information).

    time_next_maintenance: :class:`numpy.ndarray`, dtype:int
        For each powerline, it gives the time of the next planned maintenance. For example if there is:

            - `1` at position `i` it means that the powerline `i` will be disconnected for maintenance operation at
              the next time step.
            - `0` at position `i` means that powerline `i` is disconnected from the powergrid for maintenance operation
              at the current time step.
            - `-1` at position `i` means that powerline `i` will not be disconnected for maintenance reason for this
              episode.
            - `k` > 1 at position `i` it means that the powerline `i` will be disconnected for maintenance operation at
              in `k` time steps

    duration_next_maintenance: :class:`numpy.ndarray`, dtype:int
        For each powerline, it gives the number of time step that the maintenance will last (if any). This means that,
        if at position `i` of this vector:

            - there is a `0`: the powerline is not disconnected from the grid for maintenance
            - there is a `1`, `2`, ... the powerline will be disconnected for at least `1`, `2`, ... timestep (**NB**
              in all case, the powerline will stay disconnected until a :class:`grid2op.BaseAgent.BaseAgent` performs the
              proper :class:`grid2op.BaseAction.BaseAction` to reconnect it).

    target_dispatch: :class:`numpy.ndarray`, dtype:float
        For **each** generators, it gives the target redispatching, asked by the agent. This is the sum of all
        redispatching asked by the agent for during all the episode. It for each generator it is a number between:
        - pmax and pmax. Note that there is information about all generators there, even the one that are not
        dispatchable.

    actual_dispatch: :class:`numpy.ndarray`, dtype:float
        For **each** generators, it gives the redispatching currently implemented by the environment.
        Indeed, the environment tries to implement at best the :attr:`BaseObservation.target_dispatch`, but sometimes,
        due to physical limitation (pmin, pmax, ramp min and ramp max) it cannot. In this case, only the best possible
        redispatching is implemented at the current time step, and this is what this vector stores. Note that there is
        information about all generators there, even the one that are not
        dispatchable.

    """
    def __init__(self,
                 obs_env=None,
                 action_helper=None,
                 seed=None):
        GridObjects.__init__(self)

        self.action_helper = action_helper

        # time stamp information
        self.year = 1970
        self.month = 0
        self.day = 0
        self.hour_of_day = 0
        self.minute_of_hour = 0
        self.day_of_week = 0

        # for non deterministic observation that would not use default np.random module
        self.seed = None

        # handles the forecasts here
        self._forecasted_grid_act = {}
        self._forecasted_inj = []
        self._obs_env = obs_env

        self.timestep_overflow = np.zeros(shape=(self.n_line,), dtype=dt_int)

        # 0. (line is disconnected) / 1. (line is connected)
        self.line_status = np.ones(shape=self.n_line, dtype=dt_bool)

        # topological vector
        self.topo_vect = np.full(shape=self.dim_topo, dtype=dt_int, fill_value=0)

        # generators information
        self.prod_p = np.full(shape=self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self.prod_q = np.full(shape=self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self.prod_v = np.full(shape=self.n_gen, dtype=dt_float, fill_value=np.NaN)
        # loads information
        self.load_p = np.full(shape=self.n_load, dtype=dt_float, fill_value=np.NaN)
        self.load_q = np.full(shape=self.n_load, dtype=dt_float, fill_value=np.NaN)
        self.load_v = np.full(shape=self.n_load, dtype=dt_float, fill_value=np.NaN)
        # lines origin information
        self.p_or = np.full(shape=self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.q_or = np.full(shape=self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.v_or = np.full(shape=self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.a_or = np.full(shape=self.n_line, dtype=dt_float, fill_value=np.NaN)
        # lines extremity information
        self.p_ex = np.full(shape=self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.q_ex = np.full(shape=self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.v_ex = np.full(shape=self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.a_ex = np.full(shape=self.n_line, dtype=dt_float, fill_value=np.NaN)
        # lines relative flows
        self.rho = np.full(shape=self.n_line, dtype=dt_float, fill_value=np.NaN)

        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line = np.full(shape=self.n_line, dtype=dt_int, fill_value=-1)
        self.time_before_cooldown_sub = np.full(shape=self.n_sub, dtype=dt_int, fill_value=-1)
        self.time_next_maintenance = np.full(shape=self.n_line, dtype=dt_int, fill_value=-1)
        self.duration_next_maintenance = np.full(shape=self.n_line, dtype=dt_int, fill_value=-1)

        # calendar data
        self.year = dt_int(1970)
        self.month = dt_int(0)
        self.day = dt_int(0)
        self.hour_of_day = dt_int(0)
        self.minute_of_hour = dt_int(0)
        self.day_of_week = dt_int(0)

        # redispatching
        self.target_dispatch = np.full(shape=self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self.actual_dispatch = np.full(shape=self.n_gen, dtype=dt_float, fill_value=np.NaN)

        # value to assess if two observations are equal
        self._tol_equal = 5e-1
        self.attr_list_vect = None

        # to save some computation time
        self._connectivity_matrix_ = None
        self._bus_connectivity_matrix_ = None
        self._dictionnarized = None

    def state_of(self, _sentinel=None, load_id=None, gen_id=None, line_id=None, substation_id=None):
        """
        Return the state of this action on a give unique load, generator unit, powerline of substation.
        Only one of load, gen, line or substation should be filled.

        The querry of these objects can only be done by id here (ie by giving the integer of the object in the backed).
        The :class:`ActionSpace` has some utilities to access them by name too.

        Parameters
        ----------
        _sentinel: ``None``
            Used to prevent positional parameters. Internal, do not use.

        load_id: ``int``
            ID of the load we want to inspect

        gen_id: ``int``
            ID of the generator we want to inspect

        line_id: ``int``
            ID of the powerline we want to inspect

        substation_id: ``int``
            ID of the substation we want to inspect

        Returns
        -------
        res: :class:`dict`
            A dictionnary with keys and value depending on which object needs to be inspected:

            - if a load is inspected, then the keys are:

                - "p" the active value consumed by the load
                - "q" the reactive value consumed by the load
                - "v" the voltage magnitude of the bus to which the load is connected
                - "bus" on which bus the load is connected in the substation
                - "sub_id" the id of the substation to which the load is connected

            - if a generator is inspected, then the keys are:

                - "p" the active value produced by the generator
                - "q" the reactive value consumed by the generator
                - "v" the voltage magnitude of the bus to which the generator is connected
                - "bus" on which bus the generator is connected in the substation
                - "sub_id" the id of the substation to which the generator is connected
                - "actual_dispatch" the actual dispatch implemented for this generator
                - "target_dispatch" the target dispatch (cumulation of all previously asked dispatch by the agent)
                  for this generator

            - if a powerline is inspected then the keys are "origin" and "extremity" each being dictionnary with keys:

                - "p" the active flow on line end (extremity or origin)
                - "q" the reactive flow on line end (extremity or origin)
                - "v" the voltage magnitude of the bus to which the line end (extremity or origin) is connected
                - "bus" on which bus the line end (extremity or origin) is connected in the substation
                - "sub_id" the id of the substation to which the generator is connected
                - "a" the current flow on the line end (extremity or origin)

                In the case of a powerline, additional information are:

                - "maintenance": information about the maintenance operation (time of the next maintenance and duration
                  of this next maintenance.
                - "cooldown_time": for how many timestep i am not supposed to act on the powerline due to cooldown
                  (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_COOLDOWN_LINE` for more information)

            - if a substation is inspected, it returns the topology to this substation in a dictionary with keys:

                - "topo_vect": the representation of which object is connected where
                - "nb_bus": number of active buses in this substations
                - "cooldown_time": for how many timestep i am not supposed to act on the substation due to cooldown
                  (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_COOLDOWN_SUB` for more information)

        Raises
        ------
        Grid2OpException
            If _sentinel is modified, or if None of the arguments are set or alternatively if 2 or more of the
            parameters are being set.

        """
        if _sentinel is not None:
            raise Grid2OpException("action.effect_on should only be called with named argument.")

        if load_id is None and gen_id is None and line_id is None and substation_id is None:
            raise Grid2OpException("You ask the state of an object in a observation without specifying the object id. "
                                   "Please provide \"load_id\", \"gen_id\", \"line_id\" or \"substation_id\"")

        if load_id is not None:
            if gen_id is not None or line_id is not None or substation_id is not None:
                raise Grid2OpException("You can only the inspect the effect of an action on one single element")
            if load_id >= len(self.load_p):
                raise Grid2OpException("There are no load of id \"load_id={}\" in this grid.".format(load_id))

            res = {"p": self.load_p[load_id],
                   "q": self.load_q[load_id],
                   "v": self.load_v[load_id],
                   "bus": self.topo_vect[self.load_pos_topo_vect[load_id]],
                   "sub_id": self.load_to_subid[load_id]
                   }
        elif gen_id is not None:
            if line_id is not None or substation_id is not None:
                raise Grid2OpException("You can only the inspect the effect of an action on one single element")
            if gen_id >= len(self.prod_p):
                raise Grid2OpException("There are no generator of id \"gen_id={}\" in this grid.".format(gen_id))

            res = {"p": self.prod_p[gen_id],
                   "q": self.prod_q[gen_id],
                   "v": self.prod_v[gen_id],
                   "bus": self.topo_vect[self.gen_pos_topo_vect[gen_id]],
                   "sub_id": self.gen_to_subid[gen_id],
                   "target_dispatch": self.target_dispatch[gen_id],
                   "actual_dispatch": self.target_dispatch[gen_id]
                   }
        elif line_id is not None:
            if substation_id is not None:
                raise Grid2OpException("You can only the inspect the effect of an action on one single element")
            if line_id >= len(self.p_or):
                raise Grid2OpException("There are no powerline of id \"line_id={}\" in this grid.".format(line_id))

            res = {}
            # origin information
            res["origin"] = {
                "p": self.p_or[line_id],
                "q": self.q_or[line_id],
                "v": self.v_or[line_id],
                "a": self.a_or[line_id],
                "bus": self.topo_vect[self.line_or_pos_topo_vect[line_id]],
                "sub_id": self.line_or_to_subid[line_id]
            }
            # extremity information
            res["extremity"] = {
                "p": self.p_ex[line_id],
                "q": self.q_ex[line_id],
                "v": self.v_ex[line_id],
                "a": self.a_ex[line_id],
                "bus": self.topo_vect[self.line_ex_pos_topo_vect[line_id]],
                "sub_id": self.line_ex_to_subid[line_id]
            }

            # maintenance information
            res["maintenance"] = {"next": self.time_next_maintenance[line_id],
                                  "duration_next": self.duration_next_maintenance[line_id]}

            # cooldown
            res["cooldown_time"] = self.time_before_cooldown_line[line_id]

        else:
            if substation_id >= len(self.sub_info):
                raise Grid2OpException("There are no substation of id \"substation_id={}\" in this grid.".format(substation_id))

            beg_ = int(np.sum(self.sub_info[:substation_id]))
            end_ = int(beg_ + self.sub_info[substation_id])
            topo_sub = self.topo_vect[beg_:end_]
            if np.any(topo_sub > 0):
                nb_bus = np.max(topo_sub[topo_sub > 0]) - np.min(topo_sub[topo_sub > 0]) + 1
            else:
                nb_bus = 0
            res = {"topo_vect": topo_sub,
                   "nb_bus": nb_bus,
                   "cooldown_time": self.time_before_cooldown_sub[substation_id]
                   }

        return res

    def reset(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Resetting a single observation is unlikely to do what you want to do.

        Reset the :class:`BaseObservation` to a blank state, where everything is set to either ``None`` or to its default
        value.

        """
        # vecorized _grid
        self.timestep_overflow[:] = 0

        # 0. (line is disconnected) / 1. (line is connected)
        self.line_status[:] = True

        # topological vector
        self.topo_vect[:] = 0

        # generators information
        self.prod_p[:] = np.NaN
        self.prod_q[:] = np.NaN
        self.prod_v[:] = np.NaN
        # loads information
        self.load_p[:] = np.NaN
        self.load_q[:] = np.NaN
        self.load_v[:] = np.NaN
        # lines origin information
        self.p_or[:] = np.NaN
        self.q_or[:] = np.NaN
        self.v_or[:] = np.NaN
        self.a_or[:] = np.NaN
        # lines extremity information
        self.p_ex[:] = np.NaN
        self.q_ex[:] = np.NaN
        self.v_ex[:] = np.NaN
        self.a_ex[:] = np.NaN
        # lines relative flows
        self.rho[:] = np.NaN

        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line[:] = -1
        self.time_before_cooldown_sub[:] = -1
        self.time_next_maintenance[:] = -1
        self.duration_next_maintenance[:] = -1

        # calendar data
        self.year = dt_int(1970)
        self.month = dt_int(0)
        self.day = dt_int(0)
        self.hour_of_day = dt_int(0)
        self.minute_of_hour = dt_int(0)
        self.day_of_week = dt_int(0)

        # forecasts
        self._forecasted_inj = []
        self._forecasted_grid_act = {}

        # redispatching
        self.target_dispatch[:] = np.NaN
        self.actual_dispatch[:] = np.NaN

        # to save up computation time
        self._dictionnarized = None
        self._connectivity_matrix_ = None
        self._bus_connectivity_matrix_ = None

    def set_game_over(self):
        """
        Set the observation to the "game over" state:

        - all powerlines are disconnected
        - all loads are 0.
        - all prods are 0.
        - etc.
        """
        self.prod_p[:] = 0.
        self.prod_q[:] = 0.
        self.prod_v[:] = 0.
        # loads information
        self.load_p[:] = 0.
        self.load_q[:] = 0.
        self.load_v[:] = 0.
        # lines origin information
        self.p_or[:] = 0.
        self.q_or[:] = 0.
        self.v_or[:] = 0.
        self.a_or[:] = 0.
        # lines extremity information
        self.p_ex[:] = 0.
        self.q_ex[:] = 0.
        self.v_ex[:] = 0.
        self.a_ex[:] = 0.
        # lines relative flows
        self.rho[:] = 0.
        # line status
        self.line_status[:] = False
        # topological vector
        self.topo_vect[:] = -1

        # forecasts
        self._forecasted_inj = []
        self._forecasted_grid_act = {}

        # redispatching
        self.target_dispatch[:] = 0.
        self.actual_dispatch[:] = 0.

        # cooldown
        self.time_before_cooldown_line[:] = 99999
        self.time_before_cooldown_sub[:] = 99999
        self.time_next_maintenance[:] = 99999
        self.duration_next_maintenance[:] = 99999

    def __compare_stats(self, other, name):
        attr_me = getattr(self, name)
        attr_other = getattr(other, name)
        if attr_me is None and attr_other is not None:
            return False
        if attr_me is not None and attr_other is None:
            return False
        if attr_me is not None:
            if attr_me.shape != attr_other.shape:
                return False

            if attr_me.dtype != attr_other.dtype:
                return False
            if np.issubdtype(attr_me.dtype, np.dtype(dt_float).type):
                # special case of floating points, otherwise vector are never equal
                if not np.all(np.abs(attr_me - attr_other) <= self._tol_equal):
                    return False
            else:
                if not np.all(attr_me == attr_other):
                    return False
        return True

    def __eq__(self, other):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Test the equality of two observations.

        2 actions are said to be identical if the have the same impact on the powergrid. This is unlrelated to their
        respective class. For example, if an BaseAction is of class :class:`BaseAction` and doesn't act on the _injection, it
        can be equal to a an BaseAction of derived class :class:`TopologyAction` (if the topological modification are the
        same of course).

        This implies that the attributes :attr:`BaseAction.authorized_keys` is not checked in this method.

        Note that if 2 actions doesn't act on the same powergrid, or on the same backend (eg number of loads, or
        generators is not the same in *self* and *other*, or they are not in the same order) then action will be
        declared as different.

        **Known issue** if two backend are different, but the description of the _grid are identical (ie all
        n_gen, n_load, n_line, sub_info, dim_topo, all vectors \*_to_subid, and \*_pos_topo_vect are
        identical) then this method will not detect the backend are different, and the action could be declared
        as identical. For now, this is only a theoretical behaviour: if everything is the same, then probably, up to
        the naming convention, then the powergrid are identical too.

        Parameters
        ----------
        other: :class:`BaseObservation`
            An instance of class BaseAction to which "self" will be compared.

        Returns
        -------
        ``True`` if the action are equal, ``False`` otherwise.
        """

        if self.year != other.year:
            return False
        if self.month != other.month:
            return False
        if self.day != other.day:
            return False
        if self.day_of_week != other.day_of_week:
            return False
        if self.hour_of_day != other.hour_of_day:
            return False
        if self.minute_of_hour != other.minute_of_hour:
            return False

        # check that the _grid is the same in both instances
        same_grid = True
        same_grid = same_grid and self.n_gen == other.n_gen
        same_grid = same_grid and self.n_load == other.n_load
        same_grid = same_grid and self.n_line == other.n_line
        same_grid = same_grid and np.all(self.sub_info == other.sub_info)
        same_grid = same_grid and self.dim_topo == other.dim_topo
        # to which substation is connected each element
        same_grid = same_grid and np.all(self.load_to_subid == other.load_to_subid)
        same_grid = same_grid and np.all(self.gen_to_subid == other.gen_to_subid)
        same_grid = same_grid and np.all(self.line_or_to_subid == other.line_or_to_subid)
        same_grid = same_grid and np.all(self.line_ex_to_subid == other.line_ex_to_subid)
        # which index has this element in the substation vector
        same_grid = same_grid and np.all(self.load_to_sub_pos == other.load_to_sub_pos)
        same_grid = same_grid and np.all(self.gen_to_sub_pos == other.gen_to_sub_pos)
        same_grid = same_grid and np.all(self.line_or_to_sub_pos == other.line_or_to_sub_pos)
        same_grid = same_grid and np.all(self.line_ex_to_sub_pos == other.line_ex_to_sub_pos)
        # which index has this element in the topology vector
        same_grid = same_grid and np.all(self.load_pos_topo_vect == other.load_pos_topo_vect)
        same_grid = same_grid and np.all(self.gen_pos_topo_vect == other.gen_pos_topo_vect)
        same_grid = same_grid and np.all(self.line_or_pos_topo_vect == other.line_or_pos_topo_vect)
        same_grid = same_grid and np.all(self.line_ex_pos_topo_vect == other.line_ex_pos_topo_vect)

        if not same_grid:
            return False

        for stat_nm in ["line_status", "topo_vect",
                        "timestep_overflow",
                        "prod_p", "prod_q", "prod_v",
                        "load_p", "load_q", "load_v",
                        "p_or", "q_or", "v_or", "a_or",
                        "p_ex", "q_ex", "v_ex", "a_ex",
                        "time_before_cooldown_line",
                        "time_before_cooldown_sub",
                        "time_next_maintenance",
                        "duration_next_maintenance",
                        "target_dispatch", "actual_dispatch"
                        ]:
            if not self.__compare_stats(other, stat_nm):
                print("error for {}".format(stat_nm))
                # one of the above stat is not equal in this and in other
                return False

        return True

    @abstractmethod
    def update(self, env, with_forecast=True):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            This is carried out automatically by the environment in `env.step`

        Update the actual instance of BaseObservation with the new received value from the environment.

        An observation is a description of the powergrid perceived by an agent. The agent takes his decision based on
        the current observation and the past rewards.

        This method `update` receive complete detailed information about the powergrid, but that does not mean an
        agent sees everything.
        For example, it is possible to derive this class to implement some noise in the generator or load, or flows to
        mimic sensor inaccuracy.

        It is also possible to give fake information about the topology, the line status etc.

        In the Grid2Op framework it's also through the observation that the agent has access to some forecast (the way
        forecast are handled depends are implemented in this class). For example, forecast data (retrieved thanks to
        `chronics_handler`) are processed, but can be processed differently. One can apply load / production forecast to
        each _grid state, or to make forecast for one "reference" _grid state valid a whole day and update this one
        only etc.
        All these different mechanisms can be implemented in Grid2Op framework by overloading the `update` observation
        method.

        This class is really what a dispatcher observes from it environment.
        It can also include some temperatures, nebulosity, wind etc. can also be included in this class.
        """
        pass

    def connectivity_matrix(self):
        """
        Computes and return the "connectivity matrix" `con_mat`.
        if "dim_topo = 2 * n_line + n_prod + n_conso"
        It is a matrix of size dim_topo, dim_topo, with values 0 or 1.
        For two objects (lines extremity, generator unit, load) i,j :

            - if i and j are connected on the same substation:
                - if `conn_mat[i,j] = 0` it means the objects id'ed i and j are not connected to the same bus.
                - if `conn_mat[i,j] = 1` it means the objects id'ed i and j are connected to the same bus, are both end
                  of the same powerline

            - if i and j are not connected on the same substation then`conn_mat[i,j] = 0` except if i and j are
              the two extremities of the same power line, in this case `conn_mat[i,j] = 1`.

        By definition, the diagonal is made of 0.

        Returns
        -------
        res: ``numpy.ndarray``, shape:dim_topo,dim_topo, dtype:float
            The connectivity matrix, as defined above

        Notes
        -------
        For now this matrix is stored as a dense matrix. Support for sparse matrix will be added in the future.

        Examples
        ---------
        If you want to know if powerline 0 is connected at its "extremity" side with the load of id 0 you can do

        .. code-block:: python

            import grid2op
            env = grid2op.make()
            obs = env.reset()

            # retrieve the id of extremity of powerline 1:
            id_lineex_0 = obs.line_ex_pos_topo_vect[0]
            id_load_1 = obs.load_pos_topo_vect[0]

            # get the connectivity matrix
            connectivity_matrix = obs.connectivity_matrix()

            # know if the objects are connected or not
            are_connected = connectivity_matrix[id_lineex_0, id_load_1]
            # as `are_connected` is 1.0 then these objects are indeed connected

        And now, supposes we do an action that changes the topology of the substation to which these
        two objects are connected, then we get (same example continues)

        .. code-block:: python

            topo_action = env.action_space({"set_bus": {"substations_id": [(1, [1,1,1,2,2,2])]}})
            print(topo_action)
            # This action will:
            #   - NOT change anything to the injections
            #   - NOT perform any redispatching action
            #   - NOT force any line status
            #   - NOT switch any line status
            #   - NOT switch anything in the topology
            #   - Set the bus of the following element:
            #     - assign bus 1 to line (extremity) 0 [on substation 1]
            #     - assign bus 1 to line (origin) 2 [on substation 1]
            #     - assign bus 1 to line (origin) 3 [on substation 1]
            #     - assign bus 2 to line (origin) 4 [on substation 1]
            #     - assign bus 2 to generator 0 [on substation 1]
            #     - assign bus 2 to load 0 [on substation 1]

            obs, reward, done, info = env.step(topo_action)
            # and now retrieve the matrix
            connectivity_matrix = obs.connectivity_matrix()

            # know if the objects are connected or not
            are_connected = connectivity_matrix[id_lineex_0, id_load_1]
            # as `are_connected` is 0.0 then these objects are not connected anymore
            # this is visible when you "print" the action (see above) in the two following lines:
            #     - assign bus 1 to line (extremity) 0 [on substation 1]
            #     - assign bus 2 to load 0 [on substation 1]
            # -> one of them is on bus 1 [line (extremity) 0] and the other on bus 2 [load 0]
        """
        if self._connectivity_matrix_ is None:
            self._connectivity_matrix_ = np.zeros(shape=(self.dim_topo, self.dim_topo), dtype=dt_float)
            # fill it by block for the objects
            beg_ = 0
            end_ = 0
            for sub_id, nb_obj in enumerate(self.sub_info):
                # it must be a vanilla python integer, otherwise it's not handled by some backend
                # especially if written in c++
                nb_obj = int(nb_obj)
                end_ += nb_obj
                tmp = np.zeros(shape=(nb_obj, nb_obj), dtype=dt_float)
                for obj1 in range(nb_obj):
                    for obj2 in range(obj1+1, nb_obj):
                        if self.topo_vect[beg_+obj1] == self.topo_vect[beg_+obj2]:
                            # objects are on the same bus
                            tmp[obj1, obj2] = 1
                            tmp[obj2, obj1] = 1

                self._connectivity_matrix_[beg_:end_, beg_:end_] = tmp
                beg_ += nb_obj
            # connect the objects together with the lines (both ends of a lines are connected together)
            for q_id in range(self.n_line):
                self._connectivity_matrix_[self.line_or_pos_topo_vect[q_id], self.line_ex_pos_topo_vect[q_id]] = 1
                self._connectivity_matrix_[self.line_ex_pos_topo_vect[q_id], self.line_or_pos_topo_vect[q_id]] = 1

        return self._connectivity_matrix_

    def bus_connectivity_matrix(self):
        """
        If we denote by `nb_bus` the total number bus of the powergrid (you can think of a "bus" being
        a "node" if you represent a powergrid as a graph [mathematical object, not a plot] with the lines
        being the "edges"].

        The `bus_connectivity_matrix` will have a size nb_bus, nb_bus and will be made of 0 and 1.

        If `bus_connectivity_matrix[i,j] = 1` then at least a power line connects bus i and bus j.
        Otherwise, nothing connects it.

        Returns
        -------
        res: ``numpy.ndarray``, shape:nb_bus,nb_bus dtype:float
            The bus connectivity matrix

        Notes
        ------
        By convention we say that a bus is connected to itself. So the diagonal of this matrix is 1.

        For now this matrix is stored as a dense matrix. Support for sparse matrix will be added in the future.
        """
        if self._bus_connectivity_matrix_ is None:
            bus_or = self.topo_vect[self.line_or_pos_topo_vect]
            bus_ex = self.topo_vect[self.line_ex_pos_topo_vect]
            connected = (bus_or > 0) & (bus_ex > 0)
            bus_or = bus_or[connected]
            bus_ex = bus_ex[connected]
            bus_or += self.line_or_to_subid[connected] + (bus_or == 2) * self.n_sub
            bus_ex += self.line_ex_to_subid[connected] + (bus_ex == 2) * self.n_sub
            unique_bus = np.unique(np.concatenate((bus_or, bus_ex)))
            unique_bus = np.sort(unique_bus)
            nb_bus = unique_bus.shape[0]
            all_indx = np.arange(nb_bus)
            tmplate = np.arange(np.max(unique_bus)+1)
            tmplate[unique_bus] = all_indx
            self._bus_connectivity_matrix_ = np.zeros(shape=(nb_bus, nb_bus), dtype=dt_float)
            self._bus_connectivity_matrix_[tmplate[bus_or], tmplate[bus_ex]] = 1.0
            self._bus_connectivity_matrix_[tmplate[bus_ex], tmplate[bus_or]] = 1.0
            self._bus_connectivity_matrix_[all_indx, all_indx] = 1.0
        return self._bus_connectivity_matrix_


    def get_forecasted_inj(self, time_step=1):
        """
        This function allows you to retrieve directly the "forecast" injections for the step `time_step`.

        We remind that the environment, under some conditions, can produce these forecasts automatically.
        This function allows to retrieve what has been forecast.

        Parameters
        ----------
        time_step: ``int``
            The horizon of the forecast (given in number of time steps)

        Returns
        -------
        prod_p_f: ``numpy.ndarray``
            The forecast generators active values
        prod_v_f: ``numpy.ndarray``
            The forecast generators voltage setpoins
        load_p_f: ``numpy.ndarray``
            The forecast load active consumption
        load_q_f: ``numpy.ndarray``
            The forecast load reactive consumption
        """
        if time_step >= len(self._forecasted_inj):
            raise NoForecastAvailable("Forecast for {} timestep ahead is not possible with your chronics.".format(time_step))
        t, a = self._forecasted_inj[time_step]
        prod_p_f = np.full(self.n_gen, fill_value=np.NaN, dtype=dt_float)
        prod_v_f = np.full(self.n_gen, fill_value=np.NaN, dtype=dt_float)
        load_p_f = np.full(self.n_load, fill_value=np.NaN, dtype=dt_float)
        load_q_f = np.full(self.n_load, fill_value=np.NaN, dtype=dt_float)

        if "prod_p" in a["injection"]:
            prod_p_f = a["injection"]["prod_p"]
        if "prod_v" in a["injection"]:
            prod_v_f = a["injection"]["prod_v"]
        if "load_p" in a["injection"]:
            load_p_f = a["injection"]["load_p"]
        if "load_q" in a["injection"]:
            load_q_f = a["injection"]["load_q"]
        tmp_arg = ~np.isfinite(prod_p_f)
        prod_p_f[tmp_arg] = self.prod_p[tmp_arg]
        tmp_arg = ~np.isfinite(prod_v_f)
        prod_v_f[tmp_arg] = self.prod_v[tmp_arg]
        tmp_arg = ~np.isfinite(load_p_f)
        load_p_f[tmp_arg] = self.load_p[tmp_arg]
        tmp_arg = ~np.isfinite(load_q_f)
        load_q_f[tmp_arg] = self.load_q[tmp_arg]
        return prod_p_f, prod_v_f, load_p_f, load_q_f

    def get_time_stamp(self):
        """
        Get the time stamp of the current observation as a `datetime.datetime` object
        """
        res = datetime.datetime(year=self.year, month=self.month, day=self.day,
                                hour=self.hour_of_day, minute=self.minute_of_hour)
        return res

    def simulate(self, action, time_step=1):
        """
        This method is used to simulate the effect of an action on a forecast powergrid state. This forecast
        state is built upon the current observation.

        The forecast are pre computed by the environment.

        More concretely, if not deactivated by the environment
        (see :func:`grid2op.Environment.BaseEnv.deactivate_forecast`) and the environment has the capacity to
        generate these forecasts (which is the case in most grid2op environments) this function will simulate
        the effect of doing an action now and return the "next state" (often the state you would get at
        time `t + 5` mins) if you were to do the action at this step.

        It has the same return
        value as the :func:`grid2op.Environment.Environment.step` function.

        Parameters
        ----------
        action: :class:`grid2op.Action.Action`
            The action to simulate

        time_step: ``int``
            The time step of the forecasted grid to perform the action on. If no forecast are available for this
            time step, a :class:`grid2op.Exceptions.NoForecastAvailable` is thrown.

        Raises
        ------
        :class:`grid2op.Exceptions.NoForecastAvailable`
            if no forecast are available for the time_step querried.

        Returns
        -------
        simulated_observation: :class:`grid2op.Observation.Observation`
            agent's observation of the current environment after the application of the action "act" on the
            the current state.

        reward: ``float``
            amount of reward returned after previous action

        done: ``bool``
            whether the episode has ended, in which case further step() calls will return undefined results

        info: ``dict``
            contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        Notes
        ------
        This is a simulation in the sense that the "next grid state" is not the real grid state you will get. As you
        don't know the future, the "injections you forecast for the next step" will not be the real injection you
        will get in the next step.

        Also, in some circumstances, the "Backend" (ie the powerflow) used to do the simulation may not be the
        same one as the one used by the environment. This is to model a real fact: as accurate your powerflow is, it does
        not model all the reality (*"all models are wrong"*). Having a different solver for the environment (
        "the reality") than the one used to anticipate the impact of the action (this "simulate" function)
        is a way to represent this fact.

        Examples
        --------
        To simulate what would be the effect of the action "act" if you were to take this action at this step
        you can do the following:

        .. code-block:: python

            import grid2op
            # retrieve an environment
            env = grid2op.make()

            # retrieve an observation, this is the same for all observations
            obs = env.reset()

            # and now you can simulate the effect of doing nothing in the next time step
            act = env.action_space()  # this can be any action that grid2op understands
            simulated_obs, simulated_reward, simulated_done, simulated_info = obs.simulate(act)

            # `simulated_obs` will be the "observation" after the application of action `act` on the
            #                 " forecast of the grid state (it will be the "forecast state at time t+5mins usually)
            # `simulated_reward` will be the reward for the same action on the same forecast state
            # `simulated_done` will indicate whether or not the simulation ended up in a "game over"
            # `simulated_info` gives extra information on this forecast state

        """
        if self.action_helper is None or self._obs_env is None:
            raise NoForecastAvailable("No forecasts are available for this instance of BaseObservation "
                                      "(no action_space "
                                      "and no simulated environment are set).")

        if time_step < 0:
            raise NoForecastAvailable("Impossible to forecast in the past.")

        if time_step >= len(self._forecasted_inj):
            raise NoForecastAvailable("Forecast for {} timestep(s) ahead is not possible with your chronics."
                                      "".format(time_step))

        if time_step not in self._forecasted_grid_act:
            timestamp, inj_forecasted = self._forecasted_inj[time_step]
            self._forecasted_grid_act[time_step] = {
                "timestamp": timestamp,
                "inj_action": self.action_helper(inj_forecasted)
            }

        timestamp = self._forecasted_grid_act[time_step]["timestamp"]
        inj_action = self._forecasted_grid_act[time_step]["inj_action"]
        self._obs_env.init(inj_action,
                           time_stamp=timestamp,
                           timestep_overflow=self.timestep_overflow,
                           topo_vect=self.topo_vect,
                           time_step=time_step)

        sim_obs, *rest = self._obs_env.simulate(action)
        sim_obs = copy.deepcopy(sim_obs)
        return (sim_obs, *rest)

    def copy(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Make a (deep) copy of the observation.

        Returns
        -------
        res: :class:`BaseObservation`
            The deep copy of the observation

        """
        obs_env = self._obs_env
        self._obs_env = None
        res = copy.deepcopy(self)
        self._obs_env = obs_env
        res._obs_env = obs_env.copy()
        return res
