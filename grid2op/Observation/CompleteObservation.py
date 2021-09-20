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
        7. :attr:`BaseObservation.gen_p` the active value of the productions
           [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        8. :attr:`BaseObservation.gen_q` the reactive value of the productions
           [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        9. :attr:`BaseObservation.gen_v` the voltage setpoint of the productions
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
        31. :attr:`BaseObservation.storage_charge` the actual state of charge of each storage unit
            [:attr:`grid2op.Space.GridObjects.n_storage` elements]
        32. :attr:`BaseObservation.storage_power_target` the production / consumption of setpoint of each storage unit
            [:attr:`grid2op.Space.GridObjects.n_storage` elements]
        33. :attr:`BaseObservation.storage_power` the realized production / consumption of each storage unit
            [:attr:`grid2op.Space.GridObjects.n_storage` elements]
        34. :attr:`BaseObservation.gen_p_before_curtail` : the theoretical generation that would have happened
            if no generator from renewable energy sources have been performed (in MW)
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        35. :attr:`BaseObservation.curtailment` : the current curtailment applied
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        36. :attr:`BaseObservation.is_alarm_illegal` whether the last alarm has been illegal (due to budget
            constraint) [``bool``]
        37. :attr:`BaseObservation.curtailment_limit` : the current curtailment limit (if any)
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        38. :attr:`BaseObservation.time_since_last_alarm` number of step since the last alarm has been raised
            successfully [``int``]
        39. :attr:`BaseObservation.last_alarm` : for each alarm zone, gives the last step at which an alarm has
            been successfully raised at this zone
            [:attr:`grid2op.Space.GridObjects.dim_alarms` elements]
        40. :attr:`BaseObservation.attention_budget` : the current attention budget
            [``int``]
        41. :attr:`BaseObservation.was_alarm_used_after_game_over` : was the last alarm used to compute anything related
            to the attention budget when there was a game over (can only be set to ``True`` if the observation
            corresponds to a game over)
            [``bool``]

    """
    attr_list_vect = [
        "year", "month", "day", "hour_of_day",
        "minute_of_hour", "day_of_week",
        "gen_p", "gen_q", "gen_v",
        "load_p", "load_q", "load_v",
        "p_or", "q_or", "v_or", "a_or",
        "p_ex", "q_ex", "v_ex", "a_ex",
        "rho",
        "line_status", "timestep_overflow",
        "topo_vect",
        "time_before_cooldown_line", "time_before_cooldown_sub",
        "time_next_maintenance", "duration_next_maintenance",
        "target_dispatch", "actual_dispatch",
        "storage_charge", "storage_power_target", "storage_power",
        "gen_p_before_curtail", "curtailment", "curtailment_limit",
        "is_alarm_illegal", "time_since_last_alarm", "last_alarm", "attention_budget",
        "was_alarm_used_after_game_over",
        "_shunt_p", "_shunt_q", "_shunt_v", "_shunt_bus",  # starting from grid2op version 1.6.0
        "current_step", "max_step"  # starting from grid2op version 1.6.4
    ]
    attr_list_json = ["_thermal_limit",
                      "support_theta",
                      "theta_or", "theta_ex", "load_theta", "gen_theta", "storage_theta"]
    attr_list_set = set(attr_list_vect)

    def __init__(self,
                 obs_env=None,
                 action_helper=None,
                 seed=None):

        BaseObservation.__init__(self,
                                 obs_env=obs_env,
                                 action_helper=action_helper,
                                 seed=seed)
        self._dictionnarized = None

    def update(self, env, with_forecast=True):
        # reset the matrices
        self._reset_matrices()
        self.reset()

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
