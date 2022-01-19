# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import datetime
import warnings
import networkx
from abc import abstractmethod
import numpy as np
from scipy.sparse import csr_matrix

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import Grid2OpException, NoForecastAvailable, EnvError
from grid2op.Space import GridObjects

# TODO have a method that could do "forecast" by giving the _injection by the agent,
# TODO if he wants to make custom forecasts

# TODO fix "bug" when action not initalized it should return nan in to_vect

# TODO be consistent with gen_* and prod_* also in dictionaries


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
        The current month (1 = january, 12 = december)

    day: ``int``
        The current day of the month (1 = first day of the month)

    hour_of_day: ``int``
        The current hour of the day (from O to 23)

    minute_of_hour: ``int``
        The current minute of the current hour (from 0 to 59)

    day_of_week: ``int``
        The current day of the week (monday = 0 and sunday = 6)

    gen_p: :class:`numpy.ndarray`, dtype:float
        The active production value of each generator (expressed in MW).
        (the old name "prod_p" is still usable)

    gen_q: :class:`numpy.ndarray`, dtype:float
        The reactive production value of each generator (expressed in MVar).
        (the old name "prod_q" is still usable)

    gen_v: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude of the bus to which each generator is connected (expressed in kV).
        (the old name "prod_v" is still usable)

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

        When a powerline is "in maintenance", it cannot be reconnected by the `Agent` before the end of this
        maintenance.

    duration_next_maintenance: :class:`numpy.ndarray`, dtype:int
        For each powerline, it gives the number of time step that the maintenance will last (if any). This means that,
        if at position `i` of this vector:

            - there is a `0`: the powerline is not disconnected from the grid for maintenance
            - there is a `1`, `2`, ... the powerline will be disconnected for at least `1`, `2`, ... timestep (**NB**
              in all case, the powerline will stay disconnected until a :class:`grid2op.BaseAgent.BaseAgent` performs the
              proper :class:`grid2op.BaseAction.BaseAction` to reconnect it).

        When a powerline is "in maintenance", it cannot be reconnected by the `Agent` before the end of this
        maintenance.

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

    storage_charge: :class:`numpy.ndarray`, dtype:float
        The actual 'state of charge' of each storage unit, expressed in MWh.

    storage_power_target: :class:`numpy.ndarray`, dtype:float
        For each storage units, give the setpoint of production / consumption as given by the agent

    storage_power: :class:`numpy.ndarray`, dtype:float
        Give the actual storage production / loads at the given state.

    gen_p_before_curtail:  :class:`numpy.ndarray`, dtype:float
        Give the production of renewable generator there would have been
        if no curtailment were applied (**NB** it returns 0.0 for non renewable
        generators that cannot be curtailed)

    curtailment: :class:`numpy.ndarray`, dtype:float
        Give the actual curtailment setpoint for generator using renewable energy sources (1.0 means no curtailment).
        This is the results of all curtailment actions applied up to the

    curtailment_limit: :class:`numpy.ndarray`, dtype:float
        Limit (in ratio of gen_pmax) imposed on each renewable generator.

    current_step: ``int``
        Current number of step performed up until this observation (NB this is not given in the observation if
        it is transformed into a vector)

    max_step: ``int``
        Maximum number of steps possible for this episode

    delta_time: ``float``
        Time (in minutes) between the last step and the current step (usually constant in an episode, even in an environment)

    is_alarm_illegal: ``bool``
        whether the last alarm has been illegal (due to budget constraint). It can only be ``True`` if an alarm
        was raised by the agent on the previous step. Otherwise it is always ``False``

    time_since_last_alarm: ``int``
        Number of steps since the last successful alarm has been raised. It is `-1` if no alarm has been raised yet.

    last_alarm: :class:`numpy.ndarray`, dtype:int
        For each zones, gives how many steps since the last alarm was raised successfully for this zone

    attention_budget: ``int``
        The current attention budget

    was_alarm_used_after_game_over: ``bool``
        Was the last alarm used to compute anything related
        to the attention budget when there was a game over (can only be set to ``True`` if the observation
        corresponds to a game over, but not necessarily)

    _shunt_p: :class:`numpy.ndarray`, dtype:float
        Shunt active value (only available if shunts are available) (in MW)

    _shunt_q: :class:`numpy.ndarray`, dtype:float
        Shunt reactive value (only available if shunts are available) (in MVAr)

    _shunt_v: :class:`numpy.ndarray`, dtype:float
        Shunt voltage (only available if shunts are available) (in kV)

    _shunt_bus: :class:`numpy.ndarray`, dtype:float
        Bus (-1 disconnected, 1 for bus 1, 2 for bus 2) at which each shunt is connected
        (only available if shunts are available)

    """

    _attr_eq = ["line_status",
                "topo_vect",
                "timestep_overflow",
                "gen_p", "gen_q", "gen_v",
                "load_p", "load_q", "load_v",
                "p_or", "q_or", "v_or", "a_or",
                "p_ex", "q_ex", "v_ex", "a_ex",
                "time_before_cooldown_line",
                "time_before_cooldown_sub",
                "time_next_maintenance",
                "duration_next_maintenance",
                "target_dispatch", "actual_dispatch",
                "_shunt_p", "_shunt_q", "_shunt_v", "_shunt_bus",
                # storage
                "storage_charge", "storage_power_target", "storage_power",
                # curtailment
                "gen_p_before_curtail", "curtailment", "curtailment_limit",
                # attention budget
                "is_alarm_illegal", "time_since_last_alarm", "last_alarm", "attention_budget",
                "was_alarm_used_after_game_over",
                ]

    attr_list_vect = None
    # value to assess if two observations are equal
    _tol_equal = 1e-3

    def __init__(self,
                 obs_env=None,
                 action_helper=None,
                 random_prng=None):
        GridObjects.__init__(self)
        self.random_prng = random_prng
        
        self.action_helper = action_helper
        # handles the forecasts here
        self._forecasted_grid_act = {}
        self._forecasted_inj = []
        self._obs_env = obs_env

        # calendar data
        self.year = dt_int(1970)
        self.month = dt_int(1)
        self.day = dt_int(1)
        self.hour_of_day = dt_int(0)
        self.minute_of_hour = dt_int(0)
        self.day_of_week = dt_int(0)

        self.timestep_overflow = np.empty(shape=(self.n_line,), dtype=dt_int)

        # 0. (line is disconnected) / 1. (line is connected)
        self.line_status = np.empty(shape=self.n_line, dtype=dt_bool)

        # topological vector
        self.topo_vect = np.empty(shape=self.dim_topo, dtype=dt_int)

        # generators information
        self.gen_p = np.empty(shape=self.n_gen, dtype=dt_float)
        self.gen_q = np.empty(shape=self.n_gen, dtype=dt_float)
        self.gen_v = np.empty(shape=self.n_gen, dtype=dt_float)
        # loads information
        self.load_p = np.empty(shape=self.n_load, dtype=dt_float)
        self.load_q = np.empty(shape=self.n_load, dtype=dt_float)
        self.load_v = np.empty(shape=self.n_load, dtype=dt_float)
        # lines origin information
        self.p_or = np.empty(shape=self.n_line, dtype=dt_float)
        self.q_or = np.empty(shape=self.n_line, dtype=dt_float)
        self.v_or = np.empty(shape=self.n_line, dtype=dt_float)
        self.a_or = np.empty(shape=self.n_line, dtype=dt_float)
        # lines extremity information
        self.p_ex = np.empty(shape=self.n_line, dtype=dt_float)
        self.q_ex = np.empty(shape=self.n_line, dtype=dt_float)
        self.v_ex = np.empty(shape=self.n_line, dtype=dt_float)
        self.a_ex = np.empty(shape=self.n_line, dtype=dt_float)
        # lines relative flows
        self.rho = np.empty(shape=self.n_line, dtype=dt_float)

        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line = np.empty(shape=self.n_line, dtype=dt_int)
        self.time_before_cooldown_sub = np.empty(shape=self.n_sub, dtype=dt_int)
        self.time_next_maintenance = 1 * self.time_before_cooldown_line
        self.duration_next_maintenance = 1 * self.time_before_cooldown_line

        # redispatching
        self.target_dispatch = np.empty(shape=self.n_gen, dtype=dt_float)
        self.actual_dispatch = np.empty(shape=self.n_gen, dtype=dt_float)

        # storage unit
        self.storage_charge = np.empty(shape=self.n_storage, dtype=dt_float)  # in MWh
        self.storage_power_target = np.empty(shape=self.n_storage, dtype=dt_float)  # in MW
        self.storage_power = np.empty(shape=self.n_storage, dtype=dt_float)  # in MW

        # attention budget
        self.is_alarm_illegal = np.ones(shape=1, dtype=dt_bool)
        self.time_since_last_alarm = np.empty(shape=1, dtype=dt_int)
        self.last_alarm = np.empty(shape=self.dim_alarms, dtype=dt_int)
        self.attention_budget = np.empty(shape=1, dtype=dt_float)
        self.was_alarm_used_after_game_over = np.zeros(shape=1, dtype=dt_bool)

        # to save some computation time
        self._connectivity_matrix_ = None
        self._bus_connectivity_matrix_ = None
        self._dictionnarized = None
        self._vectorized = None

        # for shunt (these are not stored!)
        if self.shunts_data_available:
            self._shunt_p = np.empty(shape=self.n_shunt, dtype=dt_float)
            self._shunt_q = np.empty(shape=self.n_shunt, dtype=dt_float)
            self._shunt_v = np.empty(shape=self.n_shunt, dtype=dt_float)
            self._shunt_bus = np.empty(shape=self.n_shunt, dtype=dt_int)

        self._thermal_limit = np.empty(shape=self.n_line, dtype=dt_float)

        self.gen_p_before_curtail = np.empty(shape=self.n_gen, dtype=dt_float)
        self.curtailment = np.empty(shape=self.n_gen, dtype=dt_float)
        self.curtailment_limit = np.empty(shape=self.n_gen, dtype=dt_float)

        # the "theta" (voltage angle, in degree)
        self.support_theta = False
        self.theta_or = np.empty(shape=self.n_line, dtype=dt_float)
        self.theta_ex = np.empty(shape=self.n_line, dtype=dt_float)
        self.load_theta = np.empty(shape=self.n_load, dtype=dt_float)
        self.gen_theta = np.empty(shape=self.n_gen, dtype=dt_float)
        self.storage_theta = np.empty(shape=self.n_storage, dtype=dt_float)

        # counter
        self.current_step = dt_int(0)
        self.max_step = dt_int(np.iinfo(dt_int).max)
        self.delta_time = dt_float(5.)

    def _aux_copy(self, other):
        attr_simple = ["max_step", "current_step", "support_theta", "day_of_week",
                       "minute_of_hour", "hour_of_day", "day", "month", "year", "delta_time"]

        attr_vect = ["storage_theta", "gen_theta", "load_theta", "theta_ex", "theta_or", "curtailment_limit",
                     "curtailment", "gen_p_before_curtail", "_thermal_limit", "is_alarm_illegal",
                     "time_since_last_alarm", "last_alarm", "attention_budget", "was_alarm_used_after_game_over",
                     "storage_power", "storage_power_target", "storage_charge",
                     "actual_dispatch", "target_dispatch",
                     "duration_next_maintenance", "time_next_maintenance",
                     "time_before_cooldown_sub", "time_before_cooldown_line",
                     "rho", "a_ex", "v_ex", "q_ex", "p_ex",
                     "a_or", "v_or", "q_or", "p_or",
                     "load_p", "load_q", "load_v",
                     "gen_p", "gen_q", "gen_v",
                     "topo_vect", "line_status", "timestep_overflow"
                     ]

        if self.shunts_data_available:
            attr_vect += ["_shunt_bus", "_shunt_v", "_shunt_q", "_shunt_p"]

        for attr_nm in attr_simple:
            setattr(other, attr_nm, getattr(self, attr_nm))

        for attr_nm in attr_vect:
            getattr(other, attr_nm)[:] = getattr(self, attr_nm)

    def __copy__(self):
        res = type(self)(obs_env=self._obs_env,
                         action_helper=self.action_helper)

        # copy regular attributes
        self._aux_copy(other=res)

        # just copy
        res._connectivity_matrix_ = copy.copy(self._connectivity_matrix_)
        res._bus_connectivity_matrix_ = copy.copy(self._bus_connectivity_matrix_)
        res._dictionnarized = copy.copy(self._dictionnarized)
        res._vectorized = copy.copy(self._vectorized)

        # handles the forecasts here
        res._forecasted_grid_act = copy.copy(self._forecasted_grid_act)
        res._forecasted_inj = copy.copy(self._forecasted_inj)

        return res

    def __deepcopy__(self, memodict={}):
        res = type(self)(obs_env=self._obs_env,
                         action_helper=self.action_helper)

        # copy regular attributes
        self._aux_copy(other=res)

        # just deepcopy
        res._connectivity_matrix_ = copy.deepcopy(self._connectivity_matrix_, memodict)
        res._bus_connectivity_matrix_ = copy.deepcopy(self._bus_connectivity_matrix_, memodict)
        res._dictionnarized = copy.deepcopy(self._dictionnarized, memodict)
        res._vectorized = copy.deepcopy(self._vectorized, memodict)

        # handles the forecasts here
        res._forecasted_grid_act = copy.deepcopy(self._forecasted_grid_act, memodict)
        res._forecasted_inj = copy.deepcopy(self._forecasted_inj, memodict)

        return res

    def state_of(self,
                 _sentinel=None,
                 load_id=None,
                 gen_id=None,
                 line_id=None,
                 storage_id=None,
                 substation_id=None):
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

        line_id: ``int``
            ID of the powerline we want to inspect

        storage_id: ``int``
            ID of the storage unit we want to inspect

        substation_id: ``int``
            ID of the substation unit we want to inspect

        Returns
        -------
        res: :class:`dict`
            A dictionary with keys and value depending on which object needs to be inspected:

            - if a load is inspected, then the keys are:

                - "p" the active value consumed by the load
                - "q" the reactive value consumed by the load
                - "v" the voltage magnitude of the bus to which the load is connected
                - "theta" (optional) the voltage angle (in degree) of the bus to which the load is connected
                - "bus" on which bus the load is connected in the substation
                - "sub_id" the id of the substation to which the load is connected

            - if a generator is inspected, then the keys are:

                - "p" the active value produced by the generator
                - "q" the reactive value consumed by the generator
                - "v" the voltage magnitude of the bus to which the generator is connected
                - "theta" (optional) the voltage angle (in degree) of the bus to which the gen. is connected
                - "bus" on which bus the generator is connected in the substation
                - "sub_id" the id of the substation to which the generator is connected
                - "actual_dispatch" the actual dispatch implemented for this generator
                - "target_dispatch" the target dispatch (cumulation of all previously asked dispatch by the agent)
                  for this generator

            - if a powerline is inspected then the keys are "origin" and "extremity" each being dictionary with keys:

                - "p" the active flow on line side (extremity or origin)
                - "q" the reactive flow on line side (extremity or origin)
                - "v" the voltage magnitude of the bus to which the line side (extremity or origin) is connected
                - "theta" (optional) the voltage angle (in degree) of the bus to which line side (extremity or origin)
                   is connected
                - "bus" on which bus the line side (extremity or origin) is connected in the substation
                - "sub_id" the id of the substation to which the line side is connected
                - "a" the current flow on the line side (extremity or origin)

                In the case of a powerline, additional information are:

                - "maintenance": information about the maintenance operation (time of the next maintenance and duration
                  of this next maintenance.
                - "cooldown_time": for how many timestep i am not supposed to act on the powerline due to cooldown
                  (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_COOLDOWN_LINE` for more information)

            - if a storage unit is inspected, information are:

                - "storage_power": the power the unit actually produced / absorbed
                - "storage_charge": the state of the charge of the storage unit
                - "storage_power_target": the power production / absorbtion targer
                - "storage_theta": (optional) the voltage angle of the bus at which the storage unit is connected
                - "bus": the bus (1 or 2) to which the storage unit is connected
                - "sub_id" : the id of the substation to which the sotrage unit is connected

            - if a substation is inspected, it returns the topology to this substation in a dictionary with keys:

                - "topo_vect": the representation of which object is connected where
                - "nb_bus": number of active buses in this substations
                - "cooldown_time": for how many timestep i am not supposed to act on the substation due to cooldown
                  (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_COOLDOWN_SUB` for more information)

        Notes
        -----
        This function can only be used to retrieve the state of the element of the grid, and not the alarm sent
        or not, to the operator.

        Raises
        ------
        Grid2OpException
            If _sentinel is modified, or if None of the arguments are set or alternatively if 2 or more of the
            parameters are being set.

        """
        if _sentinel is not None:
            raise Grid2OpException("action.effect_on should only be called with named argument.")

        if load_id is None and gen_id is None and line_id is None and substation_id is None and storage_id is None:
            raise Grid2OpException("You ask the state of an object in a observation without specifying the object id. "
                                   "Please provide \"load_id\", \"gen_id\", \"line_id\", \"storage_id\" or "
                                   "\"substation_id\"")

        if load_id is not None:
            if gen_id is not None or line_id is not None or substation_id is not None or storage_id is not None:
                raise Grid2OpException("You can only the inspect the effect of an action on one single element")
            if load_id >= len(self.load_p):
                raise Grid2OpException("There are no load of id \"load_id={}\" in this grid.".format(load_id))
            if load_id < 0:
                raise Grid2OpException("`load_id` should be a positive integer")

            res = {"p": self.load_p[load_id],
                   "q": self.load_q[load_id],
                   "v": self.load_v[load_id],
                   "bus": self.topo_vect[self.load_pos_topo_vect[load_id]],
                   "sub_id": self.load_to_subid[load_id]
                   }
            if self.support_theta:
                res["theta"] = self.load_theta[load_id]
        elif gen_id is not None:
            if line_id is not None or substation_id is not None or storage_id is not None:
                raise Grid2OpException("You can only the inspect the effect of an action on one single element")
            if gen_id >= len(self.gen_p):
                raise Grid2OpException("There are no generator of id \"gen_id={}\" in this grid.".format(gen_id))
            if gen_id < 0:
                raise Grid2OpException("`gen_id` should be a positive integer")

            res = {"p": self.gen_p[gen_id],
                   "q": self.gen_q[gen_id],
                   "v": self.gen_v[gen_id],
                   "bus": self.topo_vect[self.gen_pos_topo_vect[gen_id]],
                   "sub_id": self.gen_to_subid[gen_id],
                   "target_dispatch": self.target_dispatch[gen_id],
                   "actual_dispatch": self.target_dispatch[gen_id],
                   "curtailment": self.curtailment[gen_id],
                   "curtailment_limit": self.curtailment_limit[gen_id],
                   "p_before_curtail": self.gen_p_before_curtail[gen_id],
                   }
            if self.support_theta:
                res["theta"] = self.gen_theta[gen_id]
        elif line_id is not None:
            if substation_id is not None or storage_id is not None:
                raise Grid2OpException("You can only the inspect the effect of an action on one single element")
            if line_id >= len(self.p_or):
                raise Grid2OpException("There are no powerline of id \"line_id={}\" in this grid.".format(line_id))
            if line_id < 0:
                raise Grid2OpException("`line_id` should be a positive integer")

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
            if self.support_theta:
                res["origin"]["theta"] = self.theta_or[line_id]
            # extremity information
            res["extremity"] = {
                "p": self.p_ex[line_id],
                "q": self.q_ex[line_id],
                "v": self.v_ex[line_id],
                "a": self.a_ex[line_id],
                "bus": self.topo_vect[self.line_ex_pos_topo_vect[line_id]],
                "sub_id": self.line_ex_to_subid[line_id]
            }
            if self.support_theta:
                res["origin"]["theta"] = self.theta_ex[line_id]

            # maintenance information
            res["maintenance"] = {"next": self.time_next_maintenance[line_id],
                                  "duration_next": self.duration_next_maintenance[line_id]}

            # cooldown
            res["cooldown_time"] = self.time_before_cooldown_line[line_id]

        elif storage_id is not None:
            if substation_id is not None:
                raise Grid2OpException("You can only the inspect the effect of an action on one single element")
            if storage_id >= self.n_storage:
                raise Grid2OpException("There are no storage unit with id \"storage_id={}\" in this grid.".format(line_id))
            if storage_id < 0:
                raise Grid2OpException("`storage_id` should be a positive integer")

            res = {}
            res["storage_power"] = self.storage_power[storage_id]
            res["storage_charge"] = self.storage_charge[storage_id]
            res["storage_power_target"] = self.storage_power_target[storage_id]
            res["bus"] = self.topo_vect[self.storage_pos_topo_vect[storage_id]]
            res["sub_id"] = self.storage_to_subid[storage_id]
            if self.support_theta:
                res["origin"]["theta"] = self.storage_theta[storage_id]
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

    @classmethod
    def process_grid2op_compat(cls):
        if cls.glop_version == cls.BEFORE_COMPAT_VERSION:
            # oldest version: no storage and no curtailment available

            # this is really important, otherwise things from grid2op base types will be affected
            cls.attr_list_vect = copy.deepcopy(cls.attr_list_vect)
            cls.attr_list_set = copy.deepcopy(cls.attr_list_set)

            # deactivate storage
            cls.set_no_storage()
            for el in ["storage_charge", "storage_power_target", "storage_power"]:
                if el in cls.attr_list_vect:
                    try:
                        cls.attr_list_vect.remove(el)
                    except ValueError:
                        pass

            # remove the curtailment
            for el in ["gen_p_before_curtail", "curtailment", "curtailment_limit"]:
                if el in cls.attr_list_vect:
                    try:
                        cls.attr_list_vect.remove(el)
                    except ValueError:
                        pass

            cls.attr_list_set = set(cls.attr_list_vect)

        if cls.glop_version < "1.6.0" or cls.glop_version == cls.BEFORE_COMPAT_VERSION:
            # this feature did not exist before and was introduced in grid2op 1.6.0
            cls.attr_list_vect = copy.deepcopy(cls.attr_list_vect)
            cls.attr_list_set = copy.deepcopy(cls.attr_list_set)
            cls.dim_alarms = 0
            for el in ["is_alarm_illegal", "time_since_last_alarm", "last_alarm", "attention_budget",
                       "was_alarm_used_after_game_over"]:
                try:
                    cls.attr_list_vect.remove(el)
                except ValueError as exc_:
                    # this attribute was not there in the first place
                    pass

            for el in ["_shunt_p", "_shunt_q", "_shunt_v", "_shunt_bus"]:
                # added in grid2op 1.6.0 mainly for the EpisodeReboot
                try:
                    cls.attr_list_vect.remove(el)
                except ValueError as exc_:
                    # this attribute was not there in the first place
                    pass
            cls.attr_list_set = set(cls.attr_list_vect)

        if cls.glop_version < "1.6.4" or cls.glop_version == cls.BEFORE_COMPAT_VERSION:
            # "current_step", "max_step" were added in grid2Op 1.6.4
            cls.attr_list_vect = copy.deepcopy(cls.attr_list_vect)
            cls.attr_list_set = copy.deepcopy(cls.attr_list_set)

            for el in ["max_step", "current_step"]:
                try:
                    cls.attr_list_vect.remove(el)
                except ValueError as exc_:
                    # this attribute was not there in the first place
                    pass
            cls.attr_list_set = set(cls.attr_list_vect)

        if cls.glop_version < "1.6.5" or cls.glop_version == cls.BEFORE_COMPAT_VERSION:
            # "current_step", "max_step" were added in grid2Op 1.6.5
            cls.attr_list_vect = copy.deepcopy(cls.attr_list_vect)
            cls.attr_list_set = copy.deepcopy(cls.attr_list_set)

            for el in ["delta_time"]:
                try:
                    cls.attr_list_vect.remove(el)
                except ValueError as exc_:
                    # this attribute was not there in the first place
                    pass
            cls.attr_list_set = set(cls.attr_list_vect)

    def reset(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Resetting a single observation is unlikely to do what you want to do.

        Reset the :class:`BaseObservation` to a blank state, where everything is set to either ``None`` or to its default
        value.

        """
        # 0. (line is disconnected) / 1. (line is connected)
        self.line_status[:] = True

        # topological vector
        self.topo_vect[:] = 0

        # generators information
        self.gen_p[:] = np.NaN
        self.gen_q[:] = np.NaN
        self.gen_v[:] = np.NaN
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
        self.timestep_overflow[:] = 0

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

        # storage units
        self.storage_charge[:] = np.NaN
        self.storage_power_target[:] = np.NaN
        self.storage_power[:] = np.NaN

        # to save up computation time
        self._dictionnarized = None
        self._connectivity_matrix_ = None
        self._bus_connectivity_matrix_ = None

        if self.shunts_data_available:
            self._shunt_p[:] = np.NaN
            self._shunt_q[:] = np.NaN
            self._shunt_v[:] = np.NaN
            self._shunt_bus[:] = -1

        self.support_theta = False
        self.theta_or[:] = np.NaN
        self.theta_ex[:] = np.NaN
        self.load_theta[:] = np.NaN
        self.gen_theta[:] = np.NaN
        self.storage_theta[:] = np.NaN

        # alarm feature
        self.is_alarm_illegal[:] = False
        self.time_since_last_alarm[:] = -1
        self.last_alarm[:] = False
        self.attention_budget[:] = 0
        self.was_alarm_used_after_game_over[:] = False

        self.current_step = dt_int(0)
        self.max_step = dt_int(np.iinfo(dt_int).max)
        self.delta_time = dt_float(5.)

    def set_game_over(self, env=None):
        """
        Set the observation to the "game over" state:

        - all powerlines are disconnected
        - all loads are 0.
        - all prods are 0.
        - etc.

        Notes
        -----
        As some attributes are initialized with `np.empty` it is recommended to reset here all attributes to avoid
        non deterministic behaviour.
        """
        self.gen_p[:] = 0.
        self.gen_q[:] = 0.
        self.gen_v[:] = 0.
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

        # storage
        self.storage_charge[:] = 0.
        self.storage_power_target[:] = 0.
        self.storage_power[:] = 0.

        # curtailment
        self.curtailment[:] = 0.
        self.curtailment_limit[:] = 1.
        self.gen_p_before_curtail[:] = 0.

        # cooldown
        self.time_before_cooldown_line[:] = 0
        self.time_before_cooldown_sub[:] = 0
        self.time_next_maintenance[:] = -1
        self.duration_next_maintenance[:] = 0

        # overflow
        self.timestep_overflow[:] = 0

        if self.shunts_data_available:
            self._shunt_p[:] = 0.
            self._shunt_q[:] = 0.
            self._shunt_v[:] = 0.
            self._shunt_bus[:] = -1

        if env is None:
            # set an old date (as i don't know anything about the env)
            self.year = 1970
            self.month = 1
            self.day = 1
            self.hour_of_day = 0
            self.minute_of_hour = 0
            self.day_of_week = 1
        else:
            # retrieve the date from the environment
            self.year = dt_int(env.time_stamp.year)
            self.month = dt_int(env.time_stamp.month)
            self.day = dt_int(env.time_stamp.day)
            self.hour_of_day = dt_int(env.time_stamp.hour)
            self.minute_of_hour = dt_int(env.time_stamp.minute)
            self.day_of_week = dt_int(env.time_stamp.weekday())

        if self.support_theta:
            # by convention, I say it's 0 if the grid is in total blackout
            self.theta_or[:] = 0.
            self.theta_ex[:] = 0.
            self.load_theta[:] = 0.
            self.gen_theta[:] = 0.
            self.storage_theta[:] = 0.

        # counter
        if env is not None:
            self.current_step = dt_int(env.nb_time_step)
            self.max_step = dt_int(env.max_episode_duration())

        # stuff related to alarm
        self.is_alarm_illegal[:] = False
        self.time_since_last_alarm[:] = -1
        self.last_alarm[:] = False
        self.attention_budget[:] = 0
        if env is not None:
            self.was_alarm_used_after_game_over[:] = env._is_alarm_used_in_reward
        else:
            self.was_alarm_used_after_game_over[:] = False

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
                # first special case: there can be Nan there
                me_finite = np.isfinite(attr_me)
                oth_finite = np.isfinite(attr_other)
                if np.any(me_finite != oth_finite):
                    return False
                # special case of floating points, otherwise vector are never equal
                if not np.all(np.abs(attr_me[me_finite] - attr_other[oth_finite]) <= self._tol_equal):
                    return False
            else:
                if not np.all(attr_me == attr_other):
                    return False
        return True

    def __eq__(self, other):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Test the equality of two observations.

        2 actions are said to be identical if the have the same impact on the powergrid. This is unlrelated to their
        respective class. For example, if an BaseAction is of class :class:`BaseAction` and doesn't act on the
        _injection, it
        can be equal to a an BaseAction of derived class :class:`TopologyAction` (if the topological modification
        are the same of course).

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

        # check that the underlying grid is the same in both instances
        same_grid = type(self).same_grid_class(type(other))
        if not same_grid:
            return False

        for stat_nm in self._attr_eq:
            if not self.__compare_stats(other, stat_nm):
                # one of the above stat is not equal in this and in other
                return False

        return True

    def __sub__(self, other):
        """
        computes the difference between two observation, and return an observation corresponding to
        this difference.

        This can be used to easily plot the difference between two observations at different step for
        example.
        """
        same_grid = type(self).same_grid_class(type(other))
        if not same_grid:
            raise Grid2OpException("Cannot compare to observation not coming from the same powergrid.")
        tmp_obs_env = self._obs_env
        self._obs_env = None  # keep aside the backend
        res = copy.deepcopy(self)
        self._obs_env = tmp_obs_env
        for stat_nm in self._attr_eq:
            me_ = getattr(self, stat_nm)
            oth_ = getattr(other, stat_nm)
            if me_ is None and oth_ is None:
                diff_ = None
            elif me_ is not None and oth_ is None:
                diff_ = me_
            elif me_ is None and oth_ is not None:
                if oth_.dtype == dt_bool:
                    diff_ = np.full(oth_.shape, fill_value=False, dtype=dt_bool)
                else:
                    diff_ = -oth_
            else:
                # both are not None
                if oth_.dtype == dt_bool:
                    diff_ = ~np.logical_xor(me_, oth_)
                else:
                    diff_ = me_ - oth_
            res.__setattr__(stat_nm,  diff_)
        return res

    def where_different(self, other):
        """
        Returns the difference between two observation.

        Parameters
        ----------
        other:
            Other action to compare

        Returns
        -------
        diff_: :class:`grid2op.Observation.BaseObservation`
            The observation showing the difference between `self` and `other`
        attr_nm: ``list``
            List of string representing the names of the different attributes. It's [] if the two observations
            are identical.

        """
        diff_ = self - other
        res = []
        for attr_nm in self._attr_eq:
            array_ = getattr(diff_, attr_nm)
            if array_.dtype == dt_bool:
                if np.any(~array_):
                    res.append(attr_nm)
            else:
                if (array_.shape[0] > 0) and np.max(np.abs(array_)):
                    res.append(attr_nm)
        return diff_, res

    @abstractmethod
    def update(self, env, with_forecast=True):
        """
        INTERNAL

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

        Notes
        -----
        We strongly recommend to call :attr:`BaseObservation.reset` when implementing this function.

        """
        pass

    def connectivity_matrix(self, as_csr_matrix=False):
        """
        Computes and return the "connectivity matrix" `con_mat`.
        Let "dim_topo := 2 * n_line + n_prod + n_conso + n_storage" (the total number of elements on the grid)

        It is a matrix of size dim_topo, dim_topo, with values 0 or 1.
        For two objects (lines extremity, generator unit, load) i,j :

            - if i and j are connected on the same substation:
                - if `conn_mat[i,j] = 0` it means the objects id'ed i and j are not connected to the same bus.
                - if `conn_mat[i,j] = 1` it means the objects id'ed i and j are connected to the same bus

            - if i and j are not connected on the same substation then`conn_mat[i,j] = 0` except if i and j are
              the two extremities of the same power line, in this case `conn_mat[i,j] = 1` (if the powerline is
              in service or 0 otherwise).

        By definition, the diagonal is made of 0.

        Returns
        -------
        res: ``numpy.ndarray``, shape:dim_topo,dim_topo, dtype:float
            The connectivity matrix, as defined above

        Notes
        -------
        Matrix can be either a sparse matrix or a dense matrix depending on the argument `as_csr_matrix`

        An object, is not disconnected, is always connected to itself.

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
        if self._connectivity_matrix_ is None or \
                (isinstance(self._connectivity_matrix_, csr_matrix) and not as_csr_matrix) or \
                ((not isinstance(self._connectivity_matrix_, csr_matrix)) and as_csr_matrix ):
            # self._connectivity_matrix_ = np.zeros(shape=(self.dim_topo, self.dim_topo), dtype=dt_float)
            # fill it by block for the objects
            beg_ = 0
            end_ = 0
            row_ind = []
            col_ind = []
            for sub_id, nb_obj in enumerate(self.sub_info):
                # it must be a vanilla python integer, otherwise it's not handled by some backend
                # especially if written in c++
                nb_obj = int(nb_obj)
                end_ += nb_obj
                # tmp = np.zeros(shape=(nb_obj, nb_obj), dtype=dt_float)
                for obj1 in range(nb_obj):
                    my_bus = self.topo_vect[beg_+obj1]
                    if my_bus == -1:
                        # object is disconnected, nothing is done
                        continue
                    # connect an object to itself
                    row_ind.append(beg_ + obj1)
                    col_ind.append(beg_ + obj1)

                    # connect the other objects to it
                    for obj2 in range(obj1+1, nb_obj):
                        my_bus2 = self.topo_vect[beg_+obj2]
                        if my_bus2 == -1:
                            # object is disconnected, nothing is done
                            continue
                        if my_bus == my_bus2:
                            # objects are on the same bus
                            # tmp[obj1, obj2] = 1
                            # tmp[obj2, obj1] = 1
                            row_ind.append(beg_ + obj2)
                            col_ind.append(beg_ + obj1)
                            row_ind.append(beg_ + obj1)
                            col_ind.append(beg_ + obj2)
                beg_ += nb_obj

            # both ends of a line are connected together (if line is connected)
            for q_id in range(self.n_line):
                if self.line_status[q_id]:
                    # if powerline is connected connect both its side
                    row_ind.append(self.line_or_pos_topo_vect[q_id])
                    col_ind.append(self.line_ex_pos_topo_vect[q_id])
                    row_ind.append(self.line_ex_pos_topo_vect[q_id])
                    col_ind.append(self.line_or_pos_topo_vect[q_id])
            row_ind = np.array(row_ind).astype(dt_int)
            col_ind = np.array(col_ind).astype(dt_int)
            if not as_csr_matrix:
                self._connectivity_matrix_ = np.zeros(shape=(self.dim_topo, self.dim_topo), dtype=dt_float)
                self._connectivity_matrix_[row_ind.T, col_ind] = 1.0
            else:
                data = np.ones(row_ind.shape[0], dtype=dt_float)
                self._connectivity_matrix_ = csr_matrix((data, (row_ind, col_ind)),
                                                        shape=(self.dim_topo, self.dim_topo),
                                                        dtype=dt_float)
        return self._connectivity_matrix_

    def _aux_fun_get_bus(self):
        """see in bus_connectivity matrix"""
        bus_or = self.topo_vect[self.line_or_pos_topo_vect]
        bus_ex = self.topo_vect[self.line_ex_pos_topo_vect]
        connected = (bus_or > 0) & (bus_ex > 0)
        bus_or = bus_or[connected]
        bus_ex = bus_ex[connected]
        bus_or += self.line_or_to_subid[connected] + (bus_or - 1) * self.n_sub
        bus_ex += self.line_ex_to_subid[connected] + (bus_ex - 1) * self.n_sub
        bus_or -= 1
        bus_ex -= 1
        unique_bus = np.unique(np.concatenate((bus_or, bus_ex)))
        unique_bus = np.sort(unique_bus)
        nb_bus = unique_bus.shape[0]
        return nb_bus, unique_bus, bus_or, bus_ex

    def bus_connectivity_matrix(self, as_csr_matrix=False, return_lines_index=False):
        """
        If we denote by `nb_bus` the total number bus of the powergrid (you can think of a "bus" being
        a "node" if you represent a powergrid as a graph [mathematical object, not a plot] with the lines
        being the "edges"].

        The `bus_connectivity_matrix` will have a size nb_bus, nb_bus and will be made of 0 and 1.

        If `bus_connectivity_matrix[i,j] = 1` then at least a power line connects bus i and bus j.
        Otherwise, nothing connects it.

        Parameters
        ----------
        as_csr_matrix: ``bool``
            Whether to return the bus connectivity matrix as a sparse matrix (csr format) or as a
            dense matrix. By default it's ``False`` meaning a dense matrix is returned.

        return_lines_index: ``bool``
            Whether to also return the bus index associated to both side of each powerline.

        Returns
        -------
        res: ``numpy.ndarray``, shape: (nb_bus, nb_bus) dtype:float
            The bus connectivity matrix defined above.

        Notes
        ------
        By convention we say that a bus is connected to itself. So the diagonal of this matrix is 1.

        Examples
        --------

        Here is how you can use this function:

        .. code-block:: python

            bus_bus_graph, (line_or_bus, line_ex_bus) = obs.bus_connectivity_matrix(return_lines_index=True)

            # bus_bus_graph is the matrix described above.
            # line_or_bus[0] give the id of the bus to which the origin side of powerline 0 is connected
            # line_ex_bus[0] give the id of the bus to which the extremity side of powerline 0 is connected
            # (NB: if the powerline is disconnected, both are -1)
            # this means that if line 0 is connected: bus_bus_graph[line_or_bus[0], line_ex_bus[0]] = 1
            # and bus_bus_graph[line_ex_bus[0], line_or_bus[0]] = 1
            # (of course you can replace 0 with any integer `0 <= l_id < obs.n_line`

        """
        if self._bus_connectivity_matrix_ is None or \
                (isinstance(self._bus_connectivity_matrix_, csr_matrix) and not as_csr_matrix) or \
                ((not isinstance(self._bus_connectivity_matrix_, csr_matrix)) and as_csr_matrix ) or \
                return_lines_index:
            nb_bus, unique_bus, bus_or, bus_ex = self._aux_fun_get_bus()

            # convert the bus id (from 0 to 2 * n_sub) to the row / column in the matrix (number between 0 and nb_bus)
            all_indx = np.arange(nb_bus)
            tmplate = np.arange(np.max(unique_bus)+1)
            tmplate[unique_bus] = all_indx
            bus_or_in_mat = tmplate[bus_or]
            bus_ex_in_mat = tmplate[bus_ex]

            if not as_csr_matrix:
                self._bus_connectivity_matrix_ = np.zeros(shape=(nb_bus, nb_bus), dtype=dt_float)
                self._bus_connectivity_matrix_[bus_or_in_mat, bus_ex_in_mat] = 1.0
                self._bus_connectivity_matrix_[bus_ex_in_mat, bus_or_in_mat] = 1.0
                self._bus_connectivity_matrix_[all_indx, all_indx] = 1.0
            else:
                data = np.ones(nb_bus + bus_or_in_mat.shape[0] + bus_ex_in_mat.shape[0], dtype=dt_float)
                row_ind = np.concatenate((all_indx, bus_or_in_mat, bus_ex_in_mat))
                col_ind = np.concatenate((all_indx, bus_ex_in_mat, bus_or_in_mat))
                self._bus_connectivity_matrix_ = csr_matrix((data, (row_ind, col_ind)),
                                                            shape=(nb_bus, nb_bus),
                                                            dtype=dt_float)
        if not return_lines_index:
            res = self._bus_connectivity_matrix_
        else:
            # bus or and bus ex are defined above is return_line_index is True
            lor_bus, _ = self._get_bus_id(self.line_or_pos_topo_vect, self.line_or_to_subid)
            lex_bus, _ = self._get_bus_id(self.line_ex_pos_topo_vect, self.line_ex_to_subid)
            res = (self._bus_connectivity_matrix_, (tmplate[lor_bus], tmplate[lex_bus]))
        return res

    def _get_bus_id(self, id_topo_vect, sub_id):
        """
        get the bus id with the internal convention that:

        - if object on bus 1, its bus is `sub_id`
        - if object on bus 2, its bus is `sub_id` + n_sub
        - if object on bus 3, its bus is `sub_id` + 2 * n_sub
        - etc.

        """
        bus_id = 1 * self.topo_vect[id_topo_vect]
        connected = bus_id > 0
        # bus_id[connected] = sub_id[connected] + (bus_id[connected] - 1) * self.n_sub
        bus_id[connected] += sub_id[connected] + (bus_id[connected] - 1) * self.n_sub
        bus_id -= 1  # because its label 1-2 and not 0-1
        bus_id[~connected] = -1
        return bus_id, connected

    def flow_bus_matrix(self, active_flow=True, as_csr_matrix=False):
        """
        A matrix of size "nb bus" "nb bus". Each row and columns represent a "bus" of the grid ("bus" is a power
        system word that for computer scientist means "nodes" if the powergrid is represented as a graph)

        The diagonal will sum the power produced and consumed at each bus.

        The other  element of each **row** of this matrix will be the flow of power from the bus represented
        by the line i to the bus represented by column j.

        Parameters
        ----------
        active_flow: ``bool``
            Whether to get the active flow (in MW) or the reactive flow (in MVAr). Defaults to active flow.

        as_csr_matrix: ``bool``
            Whether to retrieve the results as a scipy csr sparse matrix or as a dense matrix (default)

        Returns
        -------
        res: ``matrix``
            Which can either be a sparse matrix or a dense matrix depending on the value of the argument
            "as_csr_matrix".

        mappings: ``tuple``
            The mapping that makes the correspondence between each object and the bus to which it is connected.
            It is made of 4 elements: (load_bus, prod_bus, lor_bus, lex_bus).

            For example if `load_bus[i] = 14` it means that the load with id `i` is connected to the
            bus 14. If `load_bus[i] = -1` then the object is disconnected.

        Examples
        --------

        Here is how you can use this function:

        .. code-block:: python

            flow_mat, (load, prod, stor, ind_lor, ind_lex) = obs.flow_bus_matrix()

            # flow_mat is the matrix described above.

        Lots of information can be deduce from this matrix. For example if you want to know
        how much power goes from one bus say bus `i` to another bus (say bus `j` )
        you can look at the associated coefficient `flow_mat[i,j]` which will also be related to the
        flow on the origin (or extremity) side of the powerline connecting bus `i` to bus `j`

        You can also know how much power
        (total generation + total storage discharging - total load - total storage charging - )
        is injected at each bus `i`
        by looking at the `i` th diagonal coefficient.

        Another use would be to check if the current powergrid state (as seen by grid2op) meet
        the Kirchhoff circuit laws (conservation of energy), by doing the sum (row by row) of this
        matrix. `flow_mat.sum(axis=1)`

        """
        nb_bus, unique_bus, bus_or, bus_ex = self._aux_fun_get_bus()
        prod_bus, prod_conn = self._get_bus_id(self.gen_pos_topo_vect, self.gen_to_subid)
        load_bus, load_conn = self._get_bus_id(self.load_pos_topo_vect, self.load_to_subid)
        stor_bus, stor_conn = self._get_bus_id(self.storage_pos_topo_vect, self.storage_to_subid)
        lor_bus, lor_conn = self._get_bus_id(self.line_or_pos_topo_vect, self.line_or_to_subid)
        lex_bus, lex_conn = self._get_bus_id(self.line_ex_pos_topo_vect, self.line_ex_to_subid)

        if self.shunts_data_available:
            sh_bus = 1 * self._shunt_bus
            sh_bus[sh_bus > 0] = self.shunt_to_subid[sh_bus > 0]*(sh_bus[sh_bus > 0] - 1) + \
                                 self.shunt_to_subid[sh_bus > 0]
            sh_conn = self._shunt_bus != -1

        # convert the bus to be "id of row or column in the matrix" instead of the bus id with
        # the "grid2op convention"
        all_indx = np.arange(nb_bus)
        tmplate = np.arange(np.max(unique_bus)+1)
        tmplate[unique_bus] = all_indx
        prod_bus = tmplate[prod_bus]
        load_bus = tmplate[load_bus]
        lor_bus = tmplate[lor_bus]
        lex_bus = tmplate[lex_bus]
        stor_bus = tmplate[stor_bus]
        if active_flow:
            prod_vect = self.gen_p
            load_vect = self.load_p
            or_vect = self.p_or
            ex_vect = self.p_ex
            stor_vect = self.storage_power
            if self.shunts_data_available:
                sh_vect = self._shunt_p
        else:
            prod_vect = self.gen_q
            load_vect = self.load_q
            or_vect = self.q_or
            ex_vect = self.q_ex
            stor_vect = np.zeros(self.n_storage, dtype=dt_float)
            if self.shunts_data_available:
                sh_vect = self._shunt_q

        nb_lor = np.sum(lor_conn)
        nb_lex = np.sum(lex_conn)
        data = np.zeros(nb_bus + nb_lor + nb_lex, dtype=dt_float)

        # if two generators / loads / storage unit are connected at the same bus
        # this is why i go with matrix product and sparse matrices
        nb_prod = np.sum(prod_conn)
        if nb_prod:
            bus_prod = np.arange(prod_bus[prod_conn].max() + 1)
            map_mat = csr_matrix((np.ones(nb_prod), (prod_bus[prod_conn], np.arange(nb_prod))),
                                 shape=(bus_prod.shape[0], nb_prod),
                                 dtype=dt_float
                                 )
            data[bus_prod] += map_mat.dot(prod_vect[prod_conn])

        # handle load
        nb_load = np.sum(load_conn)
        if nb_load:
            bus_load = np.arange(load_bus[load_conn].max() + 1)
            map_mat = csr_matrix((np.ones(nb_load), (load_bus[load_conn], np.arange(nb_load))),
                                 shape=(bus_load.shape[0], nb_load),
                                 dtype=dt_float
                                 )
            data[bus_load] -= map_mat.dot(load_vect[load_conn])

        # handle storage
        nb_stor = np.sum(stor_conn)
        if nb_stor:
            bus_stor = np.arange(stor_bus[stor_conn].max() + 1)
            map_mat = csr_matrix((np.ones(nb_stor), (stor_bus[stor_conn], np.arange(nb_stor))),
                                 shape=(bus_stor.shape[0], nb_stor),
                                 dtype=dt_float
                                 )
            data[bus_stor] -= map_mat.dot(stor_vect[stor_conn])

        if self.shunts_data_available:
            # handle shunts
            nb_shunt = np.sum(sh_conn)
            if nb_shunt:
                bus_shunt = np.arange(sh_bus[sh_conn].max() + 1)
                map_mat = csr_matrix((np.ones(nb_shunt), (sh_bus[sh_conn], np.arange(nb_shunt))),
                                     shape=(bus_shunt.shape[0], nb_shunt),
                                     dtype=dt_float
                                     )
                data[bus_shunt] -= map_mat.dot(sh_vect[sh_conn])

        # powerlines
        data[np.arange(nb_lor) + nb_bus] -= or_vect[lor_conn]
        data[np.arange(nb_lex) + nb_bus + nb_lor] -= ex_vect[lex_conn]
        row_ind = np.concatenate((all_indx, lor_bus[lor_conn], lex_bus[lex_conn]))
        col_ind = np.concatenate((all_indx, lex_bus[lex_conn], lor_bus[lor_conn]))
        res = csr_matrix((data, (row_ind, col_ind)),
                         shape=(nb_bus, nb_bus),
                         dtype=dt_float)
        if not as_csr_matrix:
            res = res.toarray()

        return res, (load_bus, prod_bus, stor_bus, lor_bus, lex_bus)

    def _add_edges_simple(self, vector, attr_nm, lor_bus, lex_bus, graph):
        """add the edges, when the attributes are common for the all the powerline"""
        dict_ = {(lor_bus[lid], lex_bus[lid]): val for lid, val in enumerate(vector)}
        dict_2 = {}
        for (k1, k2), val in dict_.items():
            dict_2[(k2, k1)] = val
        dict_.update(dict_2)
        networkx.set_edge_attributes(graph, dict_, attr_nm)

    def _add_edges_multi(self,
                         vector_or,
                         vector_ex,
                         attr_nm,
                         lor_bus,
                         lex_bus,
                         graph):
        """
        Utilities to add attributes of the edges of the graph in networkx, because edges are not necessarily
        "oriented" the same way (so we need to reverse or / ex if networkx oriented it in the same way)
        """
        dict_or_glop = {}
        for lid, val in enumerate(vector_or):
            tup_ = (lor_bus[lid], lex_bus[lid])
            if tup_ in dict_or_glop:
                dict_or_glop[tup_] += val
            else:
                dict_or_glop[tup_] = val

        dict_ex_glop = {}
        for lid, val in enumerate(vector_ex):
            tup_ = (lor_bus[lid], lex_bus[lid])
            if tup_ in dict_ex_glop:
                dict_ex_glop[tup_] += val
            else:
                dict_ex_glop[tup_] = val

        dict_or = {}
        dict_ex = {}
        for (k1, k2), val in dict_or_glop.items():
            if k1 < k2:
                # networkx put it in the right "direction"
                dict_or[(k1, k2)] = val
            else:
                # networkx and grid2op do not share the same "direction"
                dict_or[(k2, k1)] = dict_ex_glop[(k1, k2)]

        for (k1, k2), val in dict_ex_glop.items():
            if k1 < k2:
                # networkx put it in the right "direction"
                dict_ex[(k1, k2)] = val
            else:
                # networkx and grid2op do not share the same "direction"
                dict_ex[(k2, k1)] = dict_or_glop[(k1, k2)]

        networkx.set_edge_attributes(graph, dict_or, "{}_or".format(attr_nm))
        networkx.set_edge_attributes(graph, dict_ex, "{}_ex".format(attr_nm))

    def as_networkx(self):
        """
        Convert this observation as a networkx graph.

        Notes
        ------
        The resulting graph is "frozen" this means that you cannot add / remove attribute on nodes or edges, nor add /
        remove edges or nodes.

        This graphs has the following properties:

        - it counts as many nodes as the number of buses of the grid
        - it counts less edges than the number of lines of the grid (two lines connecting the same buses are "merged"
          into one single edge - this is the case for parallel line, that are hence "merged" into the same edge)
        - nodes have attributes:

            - `p`: the active power produced at this node (negative means the sum of power produce minus power absorbed
              is negative)
            - `q`: the reactive power produced at this node
            - `v`: the voltage magnitude at this node
            - `cooldown`: how much longer you need to wait before being able to merge / split or change this node

        - edges have attributes too:

            - `rho`: the relative flow on this powerline
            - `cooldown`: the number of step you need to wait before being able to act on this powerline
            - `status`: whether this powerline is connected or not
            - `thermal_limit`: maximum flow allowed on the the powerline (this is the "a_or" flow)
            - `timestep_overflow`: number of time steps during which the powerline is on overflow
            - `p_or`: active power injected at this node at the "origin side".
            - `p_ex`: active power injected at this node at the "extremity side".
            - `q_or`: reactive power injected at this node at the "origin side".
            - `q_ex`: reactive power injected at this node at the "extremity side".
            - `a_or`: current flow injected at this node at the "origin side".
            - `a_ex`: current flow injected at this node at the "extremity side".

        .. danger::
            **IMPORTANT NOTE** the "origin" and "extremity" of the networkx graph is not necessarily the same as the one
            in grid2op. The "origin" side will always be the nodes with the lowest id. For example, if an edges connects
            the bus 6 to the bus 8, then the "origin" of this powerline is bus 6 (**eg** the active power
            injected at node 6 from this edge will be *p_or*) and the "extremity" side is bus 8
            (**eg** the active power injected at node 8 from this edge will be *p_ex*).

        .. note::
            The graph returned by this function is "frozen" to prevent its modification. If you really want to modify
            it you can

        Returns
        -------
        graph: ``networkx graph``
            A possible representation of the observation as a networkx graph

        Examples
        --------
        The following code explains how to check that a grid meet the kirchoffs law (conservation of energy)

        .. code-block:: python

            # create an environment and get the observation
            import grid2op
            env_name = ...
            env = grid2op.make(env_name)
            obs = env.reset()

            # retrieve the networkx graph
            graph = obs.as_networkx()

            # perform the check for every nodes
            for node_id in graph.nodes:
                # retrieve power (active and reactive) produced at this node
                p_ = graph.nodes[node_id]["p"]
                q_ = graph.nodes[node_id]["q"]

                # get the edges
                edges = graph.edges(node_id)
                p_lines = 0
                q_lines = 0
                # get the power that is "evacuated" at each nodes on all the edges connecting it to the other nodes
                # of the network
                for (k1, k2) in edges:
                    # now retrieve the active / reactive power injected at this node (looking at either *_or or *_ex
                    # depending on the direction of the powerline: remember that the "origin" is always the lowest
                    # bus id.
                    if k1 < k2:
                        # the current inspected node is the lowest, so on the "origin" side
                        p_lines += graph.edges[(k1, k2)]["p_or"]
                        q_lines += graph.edges[(k1, k2)]["q_or"]
                    else:
                        # the current node is the largest, so on the "extremity" side
                        p_lines += graph.edges[(k1, k2)]["p_ex"]
                        q_lines += graph.edges[(k1, k2)]["q_ex"]
                assert abs(p_line - p_) <= 1e-5, "error for kirchoff's law for graph for P"
                assert abs(q_line - q_) <= 1e-5, "error for kirchoff's law for graph for Q"

        """
        # TODO save this graph somewhere, in a self._as_networkx attributes for example
        mat_p, (load_bus, gen_bus, stor_bus, lor_bus, lex_bus) = self.flow_bus_matrix(active_flow=True,
                                                                                      as_csr_matrix=True)
        mat_q, *_ = self.flow_bus_matrix(active_flow=False, as_csr_matrix=True)
        # bus voltage
        bus_v = np.zeros(mat_p.shape[0])
        bus_v[lor_bus] = self.v_or
        bus_v[lex_bus] = self.v_ex
        bus_theta = np.zeros(mat_p.shape[0])
        bus_subid = np.zeros(mat_p.shape[0], dtype=dt_int)
        bus_subid[lor_bus] = self.line_or_to_subid
        bus_subid[lex_bus] = self.line_ex_to_subid
        if self.support_theta:
            bus_theta[lor_bus] = self.theta_or
            bus_theta[lex_bus] = self.theta_ex

        # bus active injection
        bus_p = mat_p.diagonal().copy()
        mat_p.setdiag(0.)
        mat_p.eliminate_zeros()

        # create the graph
        graph = networkx.from_scipy_sparse_matrix(mat_p, edge_attribute="p")

        # add the nodes attributes
        networkx.set_node_attributes(graph, {el: val for el, val in enumerate(bus_p)}, "p")
        networkx.set_node_attributes(graph, {el: val for el, val in enumerate(mat_q.diagonal())}, "q")
        networkx.set_node_attributes(graph, {el: val for el, val in enumerate(bus_v)}, "v")
        networkx.set_node_attributes(graph, {el: val for el, val in enumerate(bus_subid)}, "sub_id")
        if self.support_theta:
            networkx.set_node_attributes(graph, {el: val for el, val in enumerate(bus_theta)}, "theta")

        dict_cooldown = {el: val for el, val in enumerate(self.time_before_cooldown_sub)}
        dict_cooldown2 = {}
        for k, v in dict_cooldown.items():
            dict_cooldown2[k + self.n_sub] = v
        dict_cooldown.update(dict_cooldown2)
        networkx.set_node_attributes(graph, dict_cooldown, "cooldown")

        # add the edges attributes
        self._add_edges_multi(self.p_or, self.p_ex, "p", lor_bus, lex_bus, graph)
        self._add_edges_multi(self.q_or, self.q_ex, "q", lor_bus, lex_bus, graph)
        self._add_edges_multi(self.a_or, self.a_ex, "a", lor_bus, lex_bus, graph)
        if self.support_theta:
            self._add_edges_multi(self.theta_or, self.theta_ex, "theta", lor_bus, lex_bus, graph)

        self._add_edges_simple(self.rho, "rho", lor_bus, lex_bus, graph)
        self._add_edges_simple(self.time_before_cooldown_line, "cooldown", lor_bus, lex_bus, graph)
        self._add_edges_simple(self.line_status, "status", lor_bus, lex_bus, graph)
        self._add_edges_simple(self.thermal_limit, "thermal_limit", lor_bus, lex_bus, graph)
        self._add_edges_simple(self.timestep_overflow, "timestep_overflow", lor_bus, lex_bus, graph)
        networkx.freeze(graph)  # extra layer of security: prevent accidental modification of this graph
        return graph

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
        gen_p_f: ``numpy.ndarray``
            The forecast generators active values
        gen_v_f: ``numpy.ndarray``
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
        prod_p_f[tmp_arg] = self.gen_p[tmp_arg]
        tmp_arg = ~np.isfinite(prod_v_f)
        prod_v_f[tmp_arg] = self.gen_v[tmp_arg]
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
        if self.action_helper is None:
            raise NoForecastAvailable("No forecasts are available for this instance of BaseObservation "
                                      "(no action_space "
                                      "and no simulated environment are set).")
        if self._obs_env is None:
            raise EnvError("This observation has no \"environment used for simulation\" (_obs_env) is not created. "
                           "This is the case if you loaded this observation from a disk (for example using "
                           "EpisodeData) "
                           "or used a Runner with multi processing with the \"add_detailed_output=True\" "
                           "flag. This is a feature of grid2op: you cannot serialize backends.")

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
        return (sim_obs, *rest)  # parentheses are needed for python 3.6 at least.

    def copy(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Make a copy of the observation.

        Returns
        -------
        res: :class:`BaseObservation`
            The deep copy of the observation

        Notes
        --------
        The "obs_env" attributes
        """
        obs_env = self._obs_env
        self._obs_env = None  # _obs_env is a pointer, it is not held by this !

        action_helper = self.action_helper
        self.action_helper = None

        res = copy.deepcopy(self)

        self._obs_env = obs_env
        res._obs_env = obs_env

        self.action_helper = action_helper
        res.action_helper = action_helper
        return res

    @property
    def line_or_bus(self):
        """
        Retrieve the busbar at which each origin end of powerline is connected.

        The result follow grid2op convention:

        - -1 means the powerline is disconnected
        - 1 means it is connected to busbar 1
        - 2 means it is connected to busbar 2
        - etc.

        Notes
        -----
        In a same substation, two objects are connected together if (and only if) they are connected
        to the same busbar.

        """
        res = self.topo_vect[self.line_or_pos_topo_vect]
        res.flags.writeable = False
        return res


    @property
    def line_ex_bus(self):
        """
        Retrieve the busbar at which each extremity end of powerline is connected.

        The result follow grid2op convention:

        - -1 means the powerline is disconnected
        - 1 means it is connected to busbar 1
        - 2 means it is connected to busbar 2
        - etc.

        Notes
        -----
        In a same substation, two objects are connected together if (and only if) they are connected
        to the same busbar.

        """
        res = self.topo_vect[self.line_ex_pos_topo_vect]
        res.flags.writeable = False
        return res

    @property
    def gen_bus(self):
        """
        Retrieve the busbar at which each generator is connected.

        The result follow grid2op convention:

        - -1 means the generator is disconnected
        - 1 means it is generator to busbar 1
        - 2 means it is connected to busbar 2
        - etc.

        Notes
        -----
        In a same substation, two objects are connected together if (and only if) they are connected
        to the same busbar.

        """
        res = self.topo_vect[self.gen_pos_topo_vect]
        res.flags.writeable = False
        return res

    @property
    def load_bus(self):
        """
        Retrieve the busbar at which each load is connected.

        The result follow grid2op convention:

        - -1 means the load is disconnected
        - 1 means it is load to busbar 1
        - 2 means it is load to busbar 2
        - etc.

        Notes
        -----
        In a same substation, two objects are connected together if (and only if) they are connected
        to the same busbar.

        """
        res = self.topo_vect[self.load_pos_topo_vect]
        res.flags.writeable = False
        return res

    @property
    def storage_bus(self):
        """
        Retrieve the busbar at which each storage unit is connected.

        The result follow grid2op convention:

        - -1 means the storage unit is disconnected
        - 1 means it is storage unit to busbar 1
        - 2 means it is connected to busbar 2
        - etc.

        Notes
        -----
        In a same substation, two objects are connected together if (and only if) they are connected
        to the same busbar.

        """
        res = self.topo_vect[self.storage_pos_topo_vect]
        res.flags.writeable = False
        return res

    @property
    def prod_p(self):
        """
        As of grid2op version 1.5.0, for better consistency, the "prod_p" attribute has been renamed "gen_p".

        This property is present to maintain the backward compatibility.

        Returns
        -------
        :attr:`BaseObservation.gen_p`

        """
        return self.gen_p

    @property
    def prod_q(self):
        """
        As of grid2op version 1.5.0, for better consistency, the "prod_q" attribute has been renamed "gen_q".

        This property is present to maintain the backward compatibility.

        Returns
        -------
        :attr:`BaseObservation.gen_q`

        """
        return self.gen_q

    @property
    def prod_v(self):
        """
        As of grid2op version 1.5.0, for better consistency, the "prod_v" attribute has been renamed "gen_v".

        This property is present to maintain the backward compatibility.

        Returns
        -------
        :attr:`BaseObservation.gen_v`

        """
        return self.gen_v

    def sub_topology(self, sub_id):
        """
        Returns the topology of the given substation
        Returns
        -------

        """
        tmp = self.topo_vect[self._topo_vect_to_sub == sub_id]
        tmp.flags.writeable = False
        return tmp

    def _reset_matrices(self):
        self._vectorized = None

    def from_vect(self, vect, check_legit=True):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            To reload an observation from a vector, use the "env.observation_space.from_vect()".

        Convert back an observation represented as a vector into a proper observation.

        Some conversion are done silently from float to the type of the corresponding observation attribute.

        Parameters
        ----------
        vect: ``numpy.ndarray``
            A representation of an BaseObservation in the form of a vector that is used to convert back the current
            observation to be equal to the vect.

        """

        # reset the matrices
        self._reset_matrices()
        # and ensure everything is reloaded properly
        super().from_vect(vect, check_legit=check_legit)

    def to_dict(self):
        """
        Transform this observation as a dictionary. This dictionary allows you to inspect the state of this
        observation and is simply a shortcut of the class instance.

        Returns
        -------
        A dictionary representing the observation.

        Notes
        -------
        The returned dictionary is not necessarily json serializable. To have a grid2op observation that you can
        serialize in a json fashion, please use the :func:`grid2op.Space.GridObjects.to_json` function.

        """
        if self._dictionnarized is None:
            self._dictionnarized = {}
            self._dictionnarized["timestep_overflow"] = self.timestep_overflow
            self._dictionnarized["line_status"] = self.line_status
            self._dictionnarized["topo_vect"] = self.topo_vect
            self._dictionnarized["loads"] = {}
            self._dictionnarized["loads"]["p"] = self.load_p
            self._dictionnarized["loads"]["q"] = self.load_q
            self._dictionnarized["loads"]["v"] = self.load_v
            self._dictionnarized["prods"] = {}  # TODO will be removed in future versions
            self._dictionnarized["prods"]["p"] = self.gen_p  # TODO will be removed in future versions
            self._dictionnarized["prods"]["q"] = self.gen_q  # TODO will be removed in future versions
            self._dictionnarized["prods"]["v"] = self.gen_v  # TODO will be removed in future versions
            self._dictionnarized["gens"] = {}
            self._dictionnarized["gens"]["p"] = self.gen_p
            self._dictionnarized["gens"]["q"] = self.gen_q
            self._dictionnarized["gens"]["v"] = self.gen_v
            self._dictionnarized["lines_or"] = {}
            self._dictionnarized["lines_or"]["p"] = self.p_or
            self._dictionnarized["lines_or"]["q"] = self.q_or
            self._dictionnarized["lines_or"]["v"] = self.v_or
            self._dictionnarized["lines_or"]["a"] = self.a_or
            self._dictionnarized["lines_ex"] = {}
            self._dictionnarized["lines_ex"]["p"] = self.p_ex
            self._dictionnarized["lines_ex"]["q"] = self.q_ex
            self._dictionnarized["lines_ex"]["v"] = self.v_ex
            self._dictionnarized["lines_ex"]["a"] = self.a_ex
            self._dictionnarized["rho"] = self.rho

            self._dictionnarized["maintenance"] = {}
            self._dictionnarized["maintenance"]['time_next_maintenance'] = self.time_next_maintenance
            self._dictionnarized["maintenance"]['duration_next_maintenance'] = self.duration_next_maintenance
            self._dictionnarized["cooldown"] = {}
            self._dictionnarized["cooldown"]['line'] = self.time_before_cooldown_line
            self._dictionnarized["cooldown"]['substation'] = self.time_before_cooldown_sub
            self._dictionnarized["redispatching"] = {}
            self._dictionnarized["redispatching"]["target_redispatch"] = self.target_dispatch
            self._dictionnarized["redispatching"]["actual_dispatch"] = self.actual_dispatch

            # storage
            self._dictionnarized["storage_charge"] = 1.0 * self.storage_charge
            self._dictionnarized["storage_power_target"] = 1.0 * self.storage_power_target
            self._dictionnarized["storage_power"] = 1.0 * self.storage_power

            # curtailment
            self._dictionnarized["gen_p_before_curtail"] = 1.0 * self.gen_p_before_curtail
            self._dictionnarized["curtailment"] = 1.0 * self.curtailment
            self._dictionnarized["curtailment_limit"] = 1.0 * self.curtailment_limit

            # alarm / attention budget
            self._dictionnarized["is_alarm_illegal"] = self.is_alarm_illegal[0]
            self._dictionnarized["time_since_last_alarm"] = self.time_since_last_alarm[0]
            self._dictionnarized["last_alarm"] = copy.deepcopy(self.last_alarm)
            self._dictionnarized["attention_budget"] = self.attention_budget[0]
            self._dictionnarized["was_alarm_used_after_game_over"] = self.was_alarm_used_after_game_over[0]

            # current_step / max step
            self._dictionnarized["current_step"] = self.current_step
            self._dictionnarized["max_step"] = self.max_step

        return self._dictionnarized

    def add_act(self, act, issue_warn=True):
        """
        Easier access to the impact on the observation if an action were applied.

        This is for now only useful to get a topology in which the grid would be without
        doing an expensive `obs.simuulate`

        Notes
        -----
        This will not give the real topology of the grid in all cases for many reasons amongst:

        1) past topologies are not known by the observation. If you reconnect a powerline in the action
           without having specified on which bus, it has no way to know (but the environment does!)
           on which bus it should be reconnected (which is the last known bus)
        2) some "protections" are emulated in the environment. This means that the environment
           can disconnect some powerline under certain conditions. This is absolutely not
           taken into account here.
        3) the environment is stochastic, for example there can be maintenance or attacks (hazards)
           and the generators and loads change each step. This is not taken into account
           in this function.
        4) no checks are performed to see if the action meets the rules of the game (number of elements
           you can modify in the action, cooldowns etc.) This method **supposes** that the action
           is legal and non ambiguous.
        5) It do not check for possible "game over", for example due to isolated elements or non-connected
           grid (grid with 2 or more connex components)

        If these issues are important for you, you will need to use the
        :func:`grid2op.Observation.BaseObservation.simulate` method. It can be used like
        `obs.simulate(act, time_step=0)` but it is much more expensive.

        Parameters
        ----------
        act: :class:`grid2op.Action.BaseAction`
            The action you want to add to the observation

        issue_warn: ``bool``
            Issue a warning when this method might not compute the proper resulting topologies. Default to ``True``:
            it issues warning when something not supported is done in the action.

        Returns
        -------
        res: :class:`grid2op.Observation.Observation`
            The resulting observation. Note that this observation is not initialized with everything.
            It is only relevant when you want to study the resulting topology after you applied an
            action. Lots of `res` attributes are empty.

        Examples
        --------
        You can use it this way, for example if you want to retrieve the topology you would get (see the restriction
        in the above description) after applying an action:

        .. code-block:: python

            import grid2op

            # create the environment
            env_name = ...
            env = grid2op.make(env_name)

            # generate the first observation
            obs = env.reset()

            # make some action
            act = ...  # see the dedicated page

            # have a look at the impact on the action on the topology
            partial_obs = obs + act
            # or `partial_obs = obs.add_act(act, issue_warn=False)` if you want to silence the warnings

            # and now you can inspect the topology with any method you want:
            partial_obs.topo_vect
            partial_obs.load_bus
            bus_mat = partial_obs.bus_connectivity_matrix()
            # or even
            elem_mat = partial_obs.connectivity_matrix()

            # but you cannot use
            partial_obs.prod_p
            # or
            partial_obs.load_q
            etc.

        """

        from grid2op.Action import BaseAction
        if not isinstance(act, BaseAction):
            raise RuntimeError("You can only add actions to observation at the moment")

        act = copy.deepcopy(act)
        res = type(self)()
        res.set_game_over(env=None)

        res.topo_vect[:] = self.topo_vect
        res.line_status[:] = self.line_status

        ambiguous, except_tmp = act.is_ambiguous()
        if ambiguous:
            raise RuntimeError(f"Impossible to add an ambiguous action to an observation. Your action was "
                               f"ambiguous because: \"{except_tmp}\"")

        # if a powerline has been reconnected without specific bus, i issue a warning
        if "set_line_status" in act.authorized_keys:
            reco_powerline = act.line_set_status
            if "set_bus" in act.authorized_keys:
                line_ex_set_bus = act.line_ex_set_bus
                line_or_set_bus = act.line_or_set_bus
            else:
                line_ex_set_bus = np.zeros(res.n_line, dtype=dt_int)
                line_or_set_bus = np.zeros(res.n_line, dtype=dt_int)
            error_no_bus_set = "You reconnected a powerline with your action but did not specify on which bus " \
                               "to reconnect both its end. This behaviour, also perfectly fine for an environment " \
                               "will not be accurate in the method obs + act. Consult the documentation for more " \
                               "information. Problem arose for powerlines with id {}"

            tmp = (reco_powerline == 1) & (line_ex_set_bus <= 0) & (res.topo_vect[self.line_ex_pos_topo_vect] == -1)
            if np.any(tmp):
                id_issue_ex = np.where(tmp)[0]
                if issue_warn:
                    warnings.warn(error_no_bus_set.format(id_issue_ex))
                if "set_bus" in act.authorized_keys:
                    # assign 1 in the bus in this case
                    act.line_ex_set_bus = [(el, 1) for el in id_issue_ex]
            tmp = (reco_powerline == 1) & (line_or_set_bus <= 0) & (res.topo_vect[self.line_or_pos_topo_vect] == -1)
            if np.any(tmp):
                id_issue_or = np.where(tmp)[0]
                if issue_warn:
                    warnings.warn(error_no_bus_set.format(id_issue_or))
                if "set_bus" in act.authorized_keys:
                    # assign 1 in the bus in this case
                    act.line_or_set_bus = [(el, 1) for el in id_issue_or]

        # topo vect
        if "set_bus" in act.authorized_keys:
            res.topo_vect[act.set_bus != 0] = act.set_bus[act.set_bus != 0]

        if "change_bus" in act.authorized_keys:
            do_change_bus_on = act.change_bus & (res.topo_vect > 0)  # change bus of elements that were on
            res.topo_vect[do_change_bus_on] = 3 - res.topo_vect[do_change_bus_on]

        # topo vect: reco of powerline that should be
        res.line_status = (res.topo_vect[self.line_or_pos_topo_vect] >= 1) & \
                          (res.topo_vect[self.line_ex_pos_topo_vect] >= 1)

        # powerline status
        if "set_line_status" in act.authorized_keys:
            disco_line = (act.line_set_status == -1) & res.line_status
            res.topo_vect[res.line_or_pos_topo_vect[disco_line]] = -1
            res.topo_vect[res.line_ex_pos_topo_vect[disco_line]] = -1
            res.line_status[disco_line] = False

            reco_line = (act.line_set_status >= 1) & (~res.line_status)
            # i can do that because i already "fixed" the action to have it put 1 in case it
            # bus were not provided
            if "set_bus" in act.authorized_keys:
                # I assign previous bus (because it could have been modified)
                res.topo_vect[res.line_or_pos_topo_vect[reco_line]] = act.line_or_set_bus[reco_line]
                res.topo_vect[res.line_ex_pos_topo_vect[reco_line]] = act.line_ex_set_bus[reco_line]
            else:
                # I assign one (action do not allow me to modify the bus)
                res.topo_vect[res.line_or_pos_topo_vect[reco_line]] = 1
                res.topo_vect[res.line_ex_pos_topo_vect[reco_line]] = 1

            res.line_status[reco_line] = True

        if "change_line_status" in act.authorized_keys:
            disco_line = act.line_change_status & res.line_status
            reco_line = act.line_change_status & (~res.line_status)

            # handle disconnected powerlines
            res.topo_vect[res.line_or_pos_topo_vect[disco_line]] = -1
            res.topo_vect[res.line_ex_pos_topo_vect[disco_line]] = -1
            res.line_status[disco_line] = False

            # handle reconnected powerlines
            if np.any(reco_line):
                if "set_bus" in act.authorized_keys:
                    line_ex_set_bus = 1 * act.line_ex_set_bus
                    line_or_set_bus = 1 * act.line_or_set_bus
                else:
                    line_ex_set_bus = np.zeros(res.n_line, dtype=dt_int)
                    line_or_set_bus = np.zeros(res.n_line, dtype=dt_int)

                if issue_warn and (np.any(line_or_set_bus[reco_line] == 0) or
                                   np.any(line_ex_set_bus[reco_line] == 0)):
                    warnings.warn("A powerline has been reconnected with a \"change_status\" action without "
                                  "specifying on which bus it was supposed to be reconnected. This is "
                                  "perfectly fine in regular grid2op environment, but this behaviour "
                                  "cannot be properly implemented with the only information in the "
                                  "observation. Please see the documentation for more information."
                                  )
                    line_or_set_bus[reco_line & (line_or_set_bus == 0)] = 1
                    line_ex_set_bus[reco_line & (line_ex_set_bus == 0)] = 1

                res.topo_vect[res.line_or_pos_topo_vect[reco_line]] = line_or_set_bus[reco_line]
                res.topo_vect[res.line_ex_pos_topo_vect[reco_line]] = line_ex_set_bus[reco_line]
                res.line_status[reco_line] = True

        if 'redispatch' in act.authorized_keys:
            redisp = act.redispatch
            if np.any(redisp != 0) and issue_warn:
                warnings.warn("You did redispatching on this action. Redispatching is heavily transformed "
                              "by the environment (consult the documentation about the modeling of the "
                              "generators for example) so we will not even try to mimic this here.")

        if 'set_storage' in act.authorized_keys:
            storage_p = act.storage_p
            if np.any(storage_p != 0) and issue_warn:
                warnings.warn("You did action on storage units in this action. This implies performing some "
                              "redispatching which is heavily transformed "
                              "by the environment (consult the documentation about the modeling of the "
                              "generators for example) so we will not even try to mimic this here.")
        return res

    def __add__(self, act):
        from grid2op.Action import BaseAction
        if isinstance(act, BaseAction):
            return self.add_act(act, issue_warn=True)
        raise Grid2OpException("Only grid2op action can be added to grid2op observation at the moment.")

    @property
    def thermal_limit(self):
        """
        Return the thermal limit of the powergrid, given in Amps (A)

        Examples
        --------
        .. code-block:: python

            import grid2op
            env_name = ...
            env = grid2op.make(env_name)

            obs = env.reset()
            thermal_limit = obs.thermal_limit

        """
        res = 1.0 * self._thermal_limit
        res.flags.writeable = False
        return res

    @property
    def curtailment_mw(self):
        """
        return the curtailment, expressed in MW rather than in ratio of pmax.

        Examples
        --------
        .. code-block:: python

            import grid2op
            env_name = ...
            env = grid2op.make(env_name)

            obs = env.reset()
            curtailment_mw = obs.curtailment_mw

        """
        return self.curtailment * self.gen_pmax

    @property
    def curtailment_limit_mw(self):
        """
        return the limit of production of a generator in MW rather in ratio

        Examples
        --------
        .. code-block:: python

            import grid2op
            env_name = ...
            env = grid2op.make(env_name)

            obs = env.reset()
            curtailment_limit_mw = obs.curtailment_limit_mw

        """
        return self.curtailment_limit * self.gen_pmax

    def _update_obs_complete(self, env, with_forecast=True):
        """
        update all the observation attributes as if it was a complete, fully
        observable and without noise observation
        """
        # counter
        self.current_step = dt_int(env.nb_time_step)
        self.max_step = dt_int(env.max_episode_duration())

        # extract the time stamps
        self.year = dt_int(env.time_stamp.year)
        self.month = dt_int(env.time_stamp.month)
        self.day = dt_int(env.time_stamp.day)
        self.hour_of_day = dt_int(env.time_stamp.hour)
        self.minute_of_hour = dt_int(env.time_stamp.minute)
        self.day_of_week = dt_int(env.time_stamp.weekday())

        # get the values related to topology
        self.timestep_overflow[:] = env._timestep_overflow
        self.line_status[:] = env.backend.get_line_status()
        self.topo_vect[:] = env.backend.get_topo_vect()

        # get the values related to continuous values
        self.gen_p[:], self.gen_q[:], self.gen_v[:] = env.backend.generators_info()
        self.load_p[:], self.load_q[:], self.load_v[:] = env.backend.loads_info()
        self.p_or[:], self.q_or[:], self.v_or[:], self.a_or[:] = env.backend.lines_or_info()
        self.p_ex[:], self.q_ex[:], self.v_ex[:], self.a_ex[:] = env.backend.lines_ex_info()

        # storage units
        self.storage_charge[:] = env._storage_current_charge
        self.storage_power_target[:] = env._action_storage
        self.storage_power[:] = env._storage_power

        # handles forecasts here
        if with_forecast:
            inj_action = {}
            dict_ = {}
            dict_["load_p"] = dt_float(1.0 * self.load_p)
            dict_["load_q"] = dt_float(1.0 * self.load_q)
            dict_["prod_p"] = dt_float(1.0 * self.gen_p)
            dict_["prod_v"] = dt_float(1.0 * self.gen_v)
            inj_action["injection"] = dict_
            # inj_action = self.action_helper(inj_action)
            timestamp = self.get_time_stamp()
            self._forecasted_inj = [(timestamp, inj_action)]
            self._forecasted_inj += env.chronics_handler.forecasts()
            self._forecasted_grid = [None for _ in self._forecasted_inj]

        self.rho[:] = env.backend.get_relative_flow().astype(dt_float)

        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line[:] = env._times_before_line_status_actionable
        self.time_before_cooldown_sub[:] = env._times_before_topology_actionable
        self.time_next_maintenance[:] = env._time_next_maintenance
        self.duration_next_maintenance[:] = env._duration_next_maintenance

        # redispatching
        self.target_dispatch[:] = env._target_dispatch
        self.actual_dispatch[:] = env._actual_dispatch

        # handle shunts (if avaialble)
        if self.shunts_data_available:
            sh_p, sh_q, sh_v, sh_bus = env.backend.shunt_info()
            self._shunt_p[:] = sh_p
            self._shunt_q[:] = sh_q
            self._shunt_v[:] = sh_v
            self._shunt_bus[:] = sh_bus

        self._thermal_limit[:] = env.get_thermal_limit()

        if self.redispatching_unit_commitment_availble:
            self.gen_p_before_curtail[:] = env._gen_before_curtailment
            self.curtailment[:] = (self.gen_p_before_curtail - self.gen_p) / self.gen_pmax
            self.curtailment[~self.gen_renewable] = 0.
            self.curtailment_limit[:] = env._limit_curtailment
            self.curtailment_limit[self.curtailment_limit >= 1.] = 1.0
        else:
            self.curtailment[:] = 0.
            self.gen_p_before_curtail[:] = self.gen_p
            self.curtailment_limit[:] = 1.0

        if env.backend.can_output_theta:
            self.support_theta = True  # backend supports the computation of theta
            self.theta_or[:], self.theta_ex[:], self.load_theta[:], self.gen_theta[:], self.storage_theta[:] = \
                env.backend.get_theta()

        if self.dim_alarms and env._has_attention_budget:
            self.is_alarm_illegal[:] = env._is_alarm_illegal
            if env._attention_budget.time_last_successful_alarm_raised > 0:
                self.time_since_last_alarm[:] = self.current_step - env._attention_budget.time_last_successful_alarm_raised
            else:
                self.time_since_last_alarm[:] = -1
            self.last_alarm[:] = env._attention_budget.last_successful_alarm_raised
            self.attention_budget[:] = env._attention_budget.current_budget
            
        self.delta_time = dt_float(1.0 * env.delta_time_seconds / 60.)
