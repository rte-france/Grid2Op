# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import copy
import numpy as np
from abc import ABC, abstractmethod
import pdb

from grid2op.Exceptions import *
from grid2op.Space import GridObjects

# TODO be able to change reward here

# TODO make an action with the difference between the observation that would be an action.
# TODO have a method that could do "forecast" by giving the _injection by the agent, if he wants to make custom forecasts

# TODO finish documentation


# TODO fix "bug" when action not initalized, return nan in to_vect

class BaseObservation(GridObjects):
    """
    Basic class representing an observation.

    All observation must derive from this class and implement all its abstract methods.

    Attributes
    ----------
    action_helper: :class:`grid2op.Action.ActionSpace`
        A reprensentation of the possible action space.

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
        (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_LINE_STATUS_REMODIF` for more information). 0 means the
        an action will be able to act on this same powerline, a number > 0 (eg 1) means that an action at this time step
        cannot act on this powerline (in the example the agent have to wait 1 time step)

    time_before_cooldown_sub: :class:`numpy.ndarray`, dtype:int
        Same as :attr:`BaseObservation.time_before_cooldown_line` but for substations. For each substation, it gives the
        number of timesteps to wait before acting on this substation (see
        see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_TOPOLOGY_REMODIF` for more information).

    time_before_line_reconnectable: :class:`numpy.ndarray`, dtype:int
        For each powerline, it gives the number of timesteps before the powerline can be reconnected. This only
        concerns the maintenance, outage (hazards) and disconnection due to cascading failures (including overflow). The
        same convention as for :attr:`BaseObservation.time_before_cooldown_line` and
        :attr:`BaseObservation.time_before_cooldown_sub` is adopted: 0 at position `i` means that the powerline can be
        reconnected. It there is 2 (for example) it means that the powerline `i` is unavailable for 2 timesteps (we
        will be able to re connect it not this time, not next time, but the following timestep)

    time_next_maintenance: :class:`numpy.ndarray`, dtype:int
        For each powerline, it gives the time of the next planned maintenance. For example if there is:

            - `1` at position `i` it means that the powerline `i` will be disconnected for maintenance operation at the next time step. A
            - `0` at position `i` means that powerline `i` is disconnected from the powergrid for maintenance operation
              at the current time step.
            - `-1` at position `i` means that powerline `i` will not be disconnected for maintenance reason for this
              episode.

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
    def __init__(self, gridobj,
                 obs_env=None,
                 action_helper=None,
                 seed=None):
        GridObjects.__init__(self)
        self.init_grid(gridobj)

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
        self._forecasted_grid = []
        self._forecasted_inj = []

        self._obs_env = obs_env

        self.timestep_overflow = None

        # 0. (line is disconnected) / 1. (line is connected)
        self.line_status = None

        # topological vector
        self.topo_vect = None

        # generators information
        self.prod_p = None
        self.prod_q = None
        self.prod_v = None
        # loads information
        self.load_p = None
        self.load_q = None
        self.load_v = None
        # lines origin information
        self.p_or = None
        self.q_or = None
        self.v_or = None
        self.a_or = None
        # lines extremity information
        self.p_ex = None
        self.q_ex = None
        self.v_ex = None
        self.a_ex = None
        # lines relative flows
        self.rho = None

        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line = None
        self.time_before_cooldown_sub = None
        self.time_before_line_reconnectable = None
        self.time_next_maintenance = None
        self.duration_next_maintenance = None

        # matrices
        self.connectivity_matrix_ = None
        self.bus_connectivity_matrix_ = None
        self.vectorized = None

        # redispatching
        self.target_dispatch = None
        self.actual_dispatch = None

        # value to assess if two observations are equal
        self._tol_equal = 5e-1

        self.attr_list_vect = None
        self.reset()

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
            ID of the powerline we want to inspect

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
                  (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_LINE_STATUS_REMODIF` for more information)
                - "indisponibility": for how many timestep the powerline is unavailable (disconnected, and it's
                  impossible to reconnect it) due to hazards, maintenance or overflow (incl. cascading failure)

            - if a substation is inspected, it returns the topology to this substation in a dictionary with keys:

                - "topo_vect": the representation of which object is connected where
                - "nb_bus": number of active buses in this substations
                - "cooldown_time": for how many timestep i am not supposed to act on the substation due to cooldown
                  (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_TOPOLOGY_REMODIF` for more information)

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

            # indisponibility
            res["indisponibility"] = self.time_before_line_reconnectable[line_id]

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
            res = {
                "topo_vect": topo_sub,
                "nb_bus": nb_bus,
                "cooldown_time": self.time_before_cooldown_sub[substation_id]
                   }

        return res

    def reset(self):
        """
        Reset the :class:`BaseObservation` to a blank state, where everything is set to either ``None`` or to its default
        value.

        """
        # vecorized _grid
        self.timestep_overflow = np.zeros(shape=(self.n_line,), dtype=np.int)

        # 0. (line is disconnected) / 1. (line is connected)
        self.line_status = np.ones(shape=self.n_line, dtype=np.bool)

        # topological vector
        self.topo_vect = np.full(shape=self.dim_topo, dtype=np.int, fill_value=0)

        # generators information
        self.prod_p = np.full(shape=self.n_gen, dtype=np.float, fill_value=np.NaN)
        self.prod_q = np.full(shape=self.n_gen, dtype=np.float, fill_value=np.NaN)
        self.prod_v = np.full(shape=self.n_gen, dtype=np.float, fill_value=np.NaN)
        # loads information
        self.load_p = np.full(shape=self.n_load, dtype=np.float, fill_value=np.NaN)
        self.load_q = np.full(shape=self.n_load, dtype=np.float, fill_value=np.NaN)
        self.load_v = np.full(shape=self.n_load, dtype=np.float, fill_value=np.NaN)
        # lines origin information
        self.p_or = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        self.q_or = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        self.v_or = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        self.a_or = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        # lines extremity information
        self.p_ex = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        self.q_ex = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        self.v_ex = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        self.a_ex = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        # lines relative flows
        self.rho = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)

        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line = np.full(shape=self.n_line, dtype=np.int, fill_value=-1)
        self.time_before_cooldown_sub = np.full(shape=self.n_sub, dtype=np.int, fill_value=-1)
        self.time_before_line_reconnectable = np.full(shape=self.n_line, dtype=np.int, fill_value=-1)
        self.time_next_maintenance = np.full(shape=self.n_line, dtype=np.int, fill_value=-1)
        self.duration_next_maintenance = np.full(shape=self.n_line, dtype=np.int, fill_value=-1)

        # calendar data
        self.year = 1970
        self.month = 0
        self.day = 0
        self.hour_of_day = 0
        self.minute_of_hour = 0
        self.day_of_week = 0

        # forecasts
        self._forecasted_inj = []
        self._forecasted_grid = []

        # redispatching
        self.target_dispatch = np.full(shape=self.n_gen, dtype=np.float, fill_value=np.NaN)
        self.actual_dispatch = np.full(shape=self.n_gen, dtype=np.float, fill_value=np.NaN)

    def __compare_stats(self, other, name):
        if self.__dict__[name] is None and other.__dict__[name] is not None:
            return False
        if self.__dict__[name] is not None and other.__dict__[name] is None:
            return False
        if self.__dict__[name] is not None:
            if self.__dict__[name].shape != other.__dict__[name].shape:
                return False
            if self.__dict__[name].dtype != other.__dict__[name].dtype:
                return False
            if np.issubdtype(self.__dict__[name].dtype, np.dtype(float).type):
                # special case of floating points, otherwise vector are never equal
                if not np.all(np.abs(self.__dict__[name] - other.__dict__[name]) <= self._tol_equal):
                    return False
            else:
                if not np.all(self.__dict__[name] == other.__dict__[name]):
                    return False
        return True

    def __eq__(self, other):
        """
        Test the equality of two actions.

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

        """
        # TODO doc above

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
                        "time_before_line_reconnectable",
                        "time_next_maintenance",
                        "duration_next_maintenance",
                        "target_dispatch", "actual_dispatch"
                        ]:
            if not self.__compare_stats(other, stat_nm):
                # one of the above stat is not equal in this and in other
                return False

        return True

    @abstractmethod
    def update(self, env):
        """
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
        """
        raise NotImplementedError("This method is not implemented")

    def bus_connectivity_matrix(self):
        """
        If we denote by `nb_bus` the total number bus of the powergrid.

        The `bus_connectivity_matrix` will have a size nb_bus, nb_bus and will be made of 0 and 1.

        If `bus_connectivity_matrix[i,j] = 1` then at least a power line connects bus i and bus j.
        Otherwise, nothing connects it.

        Returns
        -------
        res: ``numpy.ndarray``, shape:nb_bus,nb_bus dtype:float
            The bus connectivity matrix
        """
        raise NotImplementedError("This method is not implemented. ")

    def simulate(self, action, time_step=0):
        """
        This method is used to simulate the effect of an action on a forecasted powergrid state. It has the same return
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
            observation: :class:`grid2op.Observation.Observation`
                agent's observation of the current environment
            reward: ``float``
                amount of reward returned after previous action
            done: ``bool``
                whether the episode has ended, in which case further step() calls will return undefined results
            info: ``dict``
                contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """
        if self.action_helper is None or self._obs_env is None:
            raise NoForecastAvailable("No forecasts are available for this instance of BaseObservation (no action_space "
                                      "and no simulated environment are set).")

        if time_step >= len(self._forecasted_inj):
            raise NoForecastAvailable("Forecast for {} timestep ahead is not possible with your chronics.".format(time_step))

        timestamp, inj_forecasted = self._forecasted_inj[time_step]
        inj_action = self.action_helper(inj_forecasted)
        # initialize the "simulation environment" with the proper injections
        self._forecasted_grid[time_step] = self._obs_env.copy()
        # TODO avoid un necessary copy above. Have one backend for all "simulate" and save instead the
        # TODO obs_env._action that set the backend to the sate we want to simulate
        self._forecasted_grid[time_step].init(inj_action, time_stamp=timestamp,
                                              timestep_overflow=self.timestep_overflow)

        return self._forecasted_grid[time_step].simulate(action)

    def copy(self):
        """
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
