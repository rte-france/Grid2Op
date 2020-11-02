# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.dtypes import dt_int, dt_float
from grid2op.Observation.BaseObservation import BaseObservation


class CompleteObservation(BaseObservation):
    """
    This class represent a complete observation, where everything on the powergrid can be observed without
    any noise.

    This is the only :class:`BaseObservation` implemented (and used) in Grid2Op. Other type of observation, for other
    usage can of course be implemented following this example.

    It has the same attributes as the :class:`BaseObservation` class. Only one is added here.

    For a :class:`CompleteObservation` the unique representation as a vector is:

        1. :attr:`BaseObservation.year` the year [1 element]
        2. :attr:`BaseObservation.month` the month [1 element]
        3. :attr:`BaseObservation.day` the day [1 element]
        4. :attr:`BaseObservation.hour_of_day` the hour of the day [1 element]
        5. :attr:`BaseObservation.minute_of_hour` minute of the hour  [1 element]
        6. :attr:`BaseObservation.day_of_week` the day of the week. Monday = 0, Sunday = 6 [1 element]
        7. :attr:`BaseObservation.prod_p` the active value of the productions
           [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        8. :attr:`BaseObservation.prod_q` the reactive value of the productions
           [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        9. :attr:`BaseObservation.prod_q` the voltage setpoint of the productions
           [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        10. :attr:`BaseObservation.load_p` the active value of the loads
            [:attr:`grid2op.Space.GridObjects.n_load` elements]
        11. :attr:`BaseObservation.load_q` the reactive value of the loads
            [:attr:`grid2op.Space.GridObjects.n_load` elements]
        12. :attr:`BaseObservation.load_v` the voltage setpoint of the loads
            [:attr:`grid2op.Space.GridObjects.n_load` elements]
        13. :attr:`BaseObservation.p_or` active flow at origin of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        14. :attr:`BaseObservation.q_or` reactive flow at origin of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        15. :attr:`BaseObservation.v_or` voltage at origin of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        16. :attr:`BaseObservation.a_or` current flow at origin of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        17. :attr:`BaseObservation.p_ex` active flow at extremity of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        18. :attr:`BaseObservation.q_ex` reactive flow at extremity of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        19. :attr:`BaseObservation.v_ex` voltage at extremity of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        20. :attr:`BaseObservation.a_ex` current flow at extremity of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        21. :attr:`BaseObservation.rho` line capacity used (current flow / thermal limit)
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        22. :attr:`BaseObservation.line_status` line status [:attr:`grid2op.Space.GridObjects.n_line` elements]
        23. :attr:`BaseObservation.timestep_overflow` number of timestep since the powerline was on overflow
            (0 if the line is not on overflow)[:attr:`grid2op.Space.GridObjects.n_line` elements]
        24. :attr:`BaseObservation.topo_vect` representation as a vector of the topology [for each element
            it gives its bus]. See :func:`grid2op.Backend.Backend.get_topo_vect` for more information.
        25. :attr:`BaseObservation.time_before_cooldown_line` representation of the cooldown time on the powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        26. :attr:`BaseObservation.time_before_cooldown_sub` representation of the cooldown time on the substations
            [:attr:`grid2op.Space.GridObjects.n_sub` elements]
        27. :attr:`BaseObservation.time_next_maintenance` number of timestep before the next maintenance (-1 means
            no maintenance are planned, 0 a maintenance is in operation) [:attr:`BaseObservation.n_line` elements]
        28. :attr:`BaseObservation.duration_next_maintenance` duration of the next maintenance. If a maintenance
            is taking place, this is the number of timestep before it ends. [:attr:`BaseObservation.n_line` elements]
        29. :attr:`BaseObservation.target_dispatch` the target dispatch for each generator
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        30. :attr:`BaseObservation.actual_dispatch` the actual dispatch for each generator
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]

    This behavior is specified in the :attr:`BaseObservation.attr_list_vect` vector.

    Attributes
    ----------
    dictionnarized: ``dict``
        The representation of the action in a form of a dictionnary. See the definition of
        :func:`CompleteObservation.to_dict` for a description of this dictionnary.

    """
    def __init__(self,
                 obs_env=None,
                 action_helper=None,
                 seed=None):

        BaseObservation.__init__(self,
                                 obs_env=obs_env,
                                 action_helper=action_helper,
                                 seed=seed)
        self._dictionnarized = None
        self.attr_list_vect = [
            "year", "month", "day", "hour_of_day",
            "minute_of_hour", "day_of_week",
            "prod_p", "prod_q", "prod_v",
            "load_p", "load_q", "load_v",
            "p_or", "q_or", "v_or", "a_or",
            "p_ex", "q_ex", "v_ex", "a_ex",
            "rho",
            "line_status", "timestep_overflow",
            "topo_vect",
            "time_before_cooldown_line", "time_before_cooldown_sub",
            "time_next_maintenance", "duration_next_maintenance",
            "target_dispatch", "actual_dispatch"
        ]

    def _reset_matrices(self):
        self.vectorized = None

    def update(self, env, with_forecast=True):
        # reset the matrices
        self._reset_matrices()
        self.reset()

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
        self.prod_p[:], self.prod_q[:], self.prod_v[:] = env.backend.generators_info()
        self.load_p[:], self.load_q[:], self.load_v[:] = env.backend.loads_info()
        self.p_or[:], self.q_or[:], self.v_or[:], self.a_or[:] = env.backend.lines_or_info()
        self.p_ex[:], self.q_ex[:], self.v_ex[:], self.a_ex[:] = env.backend.lines_ex_info()

        # handles forecasts here
        if with_forecast:
            inj_action = {}
            dict_ = {}
            dict_["load_p"] = dt_float(1.0 * self.load_p)
            dict_["load_q"] = dt_float(1.0 * self.load_q)
            dict_["prod_p"] = dt_float(1.0 * self.prod_p)
            dict_["prod_v"] = dt_float(1.0 * self.prod_v)
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

    def from_vect(self, vect, check_legit=True):
        """
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
            self._dictionnarized["prods"] = {}
            self._dictionnarized["prods"]["p"] = self.prod_p
            self._dictionnarized["prods"]["q"] = self.prod_q
            self._dictionnarized["prods"]["v"] = self.prod_v
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

        return self._dictionnarized
