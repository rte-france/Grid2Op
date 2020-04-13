# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import copy

from grid2op.Observation.BaseObservation import BaseObservation


class CompleteObservation(BaseObservation):
    """
    This class represent a complete observation, where everything on the powergrid can be observed without
    any noise.

    This is the only :class:`BaseObservation` implemented (and used) in Grid2Op. Other type of observation, for other
    usage can of course be implemented following this example.

    It has the same attributes as the :class:`BaseObservation` class. Only one is added here.

    For a :class:`CompleteObservation` the unique representation as a vector is:

        1. the year [1 element]
        2. the month [1 element]
        3. the day [1 element]
        4. the day of the week. Monday = 0, Sunday = 6 [1 element]
        5. the hour of the day [1 element]
        6. minute of the hour  [1 element]
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
        27. :attr:`BaseObservation.time_before_line_reconnectable` number of timestep to wait before a powerline
            can be reconnected (it is disconnected due to maintenance, cascading failure or overflow)
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        28. :attr:`BaseObservation.time_next_maintenance` number of timestep before the next maintenance (-1 means
            no maintenance are planned, 0 a maintenance is in operation) [:attr:`BaseObservation.n_line` elements]
        29. :attr:`BaseObservation.duration_next_maintenance` duration of the next maintenance. If a maintenance
            is taking place, this is the number of timestep before it ends. [:attr:`BaseObservation.n_line` elements]
        30. :attr:`BaseObservation.target_dispatch` the target dispatch for each generator
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        31. :attr:`BaseObservation.actual_dispatch` the actual dispatch for each generator
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]

    This behavior is specified in the :attr:`BaseObservation.attr_list_vect` vector.

    Attributes
    ----------
    dictionnarized: ``dict``
        The representation of the action in a form of a dictionnary. See the definition of
        :func:`CompleteObservation.to_dict` for a description of this dictionnary.

    """
    def __init__(self, gridobj,
                 obs_env=None,
                 action_helper=None,
                 seed=None):

        BaseObservation.__init__(self, gridobj,
                                 obs_env=obs_env,
                                 action_helper=action_helper,
                                 seed=seed)
        self.dictionnarized = None
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
            "time_before_line_reconnectable",
            "time_next_maintenance", "duration_next_maintenance",
            "target_dispatch", "actual_dispatch"
        ]

    def _reset_matrices(self):
        self.connectivity_matrix_ = None
        self.bus_connectivity_matrix_ = None
        self.vectorized = None
        self.dictionnarized = None

    def update(self, env):
        """
        This use the environement to update properly the BaseObservation.

        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The environment from which to update this observation.

        """
        # reset the matrices
        self._reset_matrices()
        self.reset()

        # extract the time stamps
        self.year = env.time_stamp.year
        self.month = env.time_stamp.month
        self.day = env.time_stamp.day
        self.hour_of_day = env.time_stamp.hour
        self.minute_of_hour = env.time_stamp.minute
        self.day_of_week = env.time_stamp.weekday()

        # get the values related to topology
        self.timestep_overflow = copy.copy(env.timestep_overflow)
        self.line_status = copy.copy(env.backend.get_line_status())
        self.topo_vect = copy.copy(env.backend.get_topo_vect())

        # get the values related to continuous values
        self.prod_p[:], self.prod_q[:], self.prod_v[:] = env.backend.generators_info()
        self.load_p[:], self.load_q[:], self.load_v[:] = env.backend.loads_info()
        self.p_or[:], self.q_or[:], self.v_or[:], self.a_or[:] = env.backend.lines_or_info()
        self.p_ex[:], self.q_ex[:], self.v_ex[:], self.a_ex[:] = env.backend.lines_ex_info()

        # handles forecasts here
        self._forecasted_inj = env.chronics_handler.forecasts()
        for i in range(len(self._forecasted_grid)):
            # in the action, i assign the lat topology known, it's a choice here...
            self._forecasted_grid[i]["setbus"] = self.topo_vect

        self._forecasted_grid = [None for _ in self._forecasted_inj]
        self.rho = env.backend.get_relative_flow()

        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line[:] = env.times_before_line_status_actionable
        self.time_before_cooldown_sub[:] = env.times_before_topology_actionable
        self.time_before_line_reconnectable[:] = env.time_remaining_before_line_reconnection
        self.time_next_maintenance[:] = env.time_next_maintenance
        self.duration_next_maintenance[:] = env.duration_next_maintenance

        # redispatching
        self.target_dispatch[:] = env.target_dispatch
        self.actual_dispatch[:] = env.actual_dispatch

    def from_vect(self, vect):
        """
        Convert back an observation represented as a vector into a proper observation.

        Some convertion are done silently from float to the type of the corresponding observation attribute.

        Parameters
        ----------
        vect: ``numpy.ndarray``
            A representation of an BaseObservation in the form of a vector that is used to convert back the current
            observation to be equal to the vect.

        """

        # reset the matrices
        self._reset_matrices()
        # and ensure everything is reloaded properly
        super().from_vect(vect)

    def to_dict(self):
        """

        Returns
        -------

        """
        # TODO doc
        if self.dictionnarized is None:
            self.dictionnarized = {}
            self.dictionnarized["timestep_overflow"] = self.timestep_overflow
            self.dictionnarized["line_status"] = self.line_status
            self.dictionnarized["topo_vect"] = self.topo_vect
            self.dictionnarized["loads"] = {}
            self.dictionnarized["loads"]["p"] = self.load_p
            self.dictionnarized["loads"]["q"] = self.load_q
            self.dictionnarized["loads"]["v"] = self.load_v
            self.dictionnarized["prods"] = {}
            self.dictionnarized["prods"]["p"] = self.prod_p
            self.dictionnarized["prods"]["q"] = self.prod_q
            self.dictionnarized["prods"]["v"] = self.prod_v
            self.dictionnarized["lines_or"] = {}
            self.dictionnarized["lines_or"]["p"] = self.p_or
            self.dictionnarized["lines_or"]["q"] = self.q_or
            self.dictionnarized["lines_or"]["v"] = self.v_or
            self.dictionnarized["lines_or"]["a"] = self.a_or
            self.dictionnarized["lines_ex"] = {}
            self.dictionnarized["lines_ex"]["p"] = self.p_ex
            self.dictionnarized["lines_ex"]["q"] = self.q_ex
            self.dictionnarized["lines_ex"]["v"] = self.v_ex
            self.dictionnarized["lines_ex"]["a"] = self.a_ex
            self.dictionnarized["rho"] = self.rho

            self.dictionnarized["maintenance"] = {}
            self.dictionnarized["maintenance"]['time_next_maintenance'] = self.time_next_maintenance
            self.dictionnarized["maintenance"]['duration_next_maintenance'] = self.duration_next_maintenance
            self.dictionnarized["cooldown"] = {}
            self.dictionnarized["cooldown"]['line'] = self.time_before_cooldown_line
            self.dictionnarized["cooldown"]['substation'] = self.time_before_cooldown_sub
            self.dictionnarized["time_before_line_reconnectable"] = self.time_before_line_reconnectable
            self.dictionnarized["redispatching"] = {}
            self.dictionnarized["redispatching"]["target_redispatch"] = self.target_dispatch
            self.dictionnarized["redispatching"]["actual_dispatch"] = self.actual_dispatch

        return self.dictionnarized

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
        if self.connectivity_matrix_ is None:
            self.connectivity_matrix_ = np.zeros(shape=(self.dim_topo, self.dim_topo),dtype=np.float)
            # fill it by block for the objects
            beg_ = 0
            end_ = 0
            for sub_id, nb_obj in enumerate(self.sub_info):
                # it must be a vanilla python integer, otherwise it's not handled by some backend
                # especially if written in c++
                nb_obj = int(nb_obj)
                end_ += nb_obj
                tmp = np.zeros(shape=(nb_obj, nb_obj), dtype=np.float)
                for obj1 in range(nb_obj):
                    for obj2 in range(obj1+1, nb_obj):
                        if self.topo_vect[beg_+obj1] == self.topo_vect[beg_+obj2]:
                            # objects are on the same bus
                            tmp[obj1, obj2] = 1
                            tmp[obj2, obj1] = 1

                self.connectivity_matrix_[beg_:end_, beg_:end_] = tmp
                beg_ += nb_obj
            # connect the objects together with the lines (both ends of a lines are connected together)
            for q_id in range(self.n_line):
                self.connectivity_matrix_[self.line_or_pos_topo_vect[q_id], self.line_ex_pos_topo_vect[q_id]] = 1
                self.connectivity_matrix_[self.line_ex_pos_topo_vect[q_id], self.line_or_pos_topo_vect[q_id]] = 1

        return self.connectivity_matrix_

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
        # TODO voir avec Antoine pour les r,x,h ici !! (surtout les x)
        if self.bus_connectivity_matrix_ is None:
            # computes the number of buses in the powergrid.
            nb_bus = 0
            nb_bus_per_sub = np.zeros(self.sub_info.shape[0])
            beg_ = 0
            end_ = 0
            for sub_id, nb_obj in enumerate(self.sub_info):
                nb_obj = int(nb_obj)
                end_ += nb_obj

                tmp = len(np.unique(self.topo_vect[beg_:end_]))
                nb_bus_per_sub[sub_id] = tmp
                nb_bus += tmp

                beg_ += nb_obj

            # define the bus_connectivity_matrix
            self.bus_connectivity_matrix_ = np.zeros(shape=(nb_bus, nb_bus), dtype=np.float)
            np.fill_diagonal(self.bus_connectivity_matrix_, 1)

            for q_id in range(self.n_line):
                bus_or = int(self.topo_vect[self.line_or_pos_topo_vect[q_id]])
                sub_id_or = int(self.line_or_to_subid[q_id])

                bus_ex = int(self.topo_vect[self.line_ex_pos_topo_vect[q_id]])
                sub_id_ex = int(self.line_ex_to_subid[q_id])

                # try:
                bus_id_or = int(np.sum(nb_bus_per_sub[:sub_id_or])+(bus_or-1))
                bus_id_ex = int(np.sum(nb_bus_per_sub[:sub_id_ex])+(bus_ex-1))

                self.bus_connectivity_matrix_[bus_id_or, bus_id_ex] = 1
                self.bus_connectivity_matrix_[bus_id_ex, bus_id_or] = 1
                # except:
                #     pdb.set_trace()
        return self.bus_connectivity_matrix_

