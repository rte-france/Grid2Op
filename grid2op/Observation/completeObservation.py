# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Observation.baseObservation import BaseObservation


class CompleteObservation(BaseObservation):
    """
    This class represent a complete observation, where everything on the powergrid can be observed without
    any noise.

    This is the only :class:`BaseObservation` implemented (and used) in Grid2Op. Other type of observation, for other
    usage can of course be implemented following this example.

    It has the same attributes as the :class:`BaseObservation` class. Only one is added here.

    For a :class:`CompleteObservation` the unique representation as a vector is:

        #. :attr:`BaseObservation.year` the year [1 element]
        #. :attr:`BaseObservation.month` the month [1 element]
        #. :attr:`BaseObservation.day` the day [1 element]
        #. :attr:`BaseObservation.hour_of_day` the hour of the day [1 element]
        #. :attr:`BaseObservation.minute_of_hour` minute of the hour  [1 element]
        #. :attr:`BaseObservation.day_of_week` the day of the week. Monday = 0, Sunday = 6 [1 element]
        #. :attr:`BaseObservation.gen_p` the active value of the productions
           [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        #. :attr:`BaseObservation.gen_q` the reactive value of the productions
           [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        #. :attr:`BaseObservation.gen_v` the voltage setpoint of the productions
           [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        #. :attr:`BaseObservation.load_p` the active value of the loads
            [:attr:`grid2op.Space.GridObjects.n_load` elements]
        #. :attr:`BaseObservation.load_q` the reactive value of the loads
            [:attr:`grid2op.Space.GridObjects.n_load` elements]
        #. :attr:`BaseObservation.load_v` the voltage setpoint of the loads
            [:attr:`grid2op.Space.GridObjects.n_load` elements]
        #. :attr:`BaseObservation.p_or` active flow at origin of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.q_or` reactive flow at origin of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.v_or` voltage at origin of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.a_or` current flow at origin of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.p_ex` active flow at extremity of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.q_ex` reactive flow at extremity of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.v_ex` voltage at extremity of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.a_ex` current flow at extremity of powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.rho` line capacity used (current flow / thermal limit)
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.line_status` line status [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.timestep_overflow` number of timestep since the powerline was on overflow
            (0 if the line is not on overflow)[:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.timestep_protection_engaged` number of timestep since the powerline was on overflow
            (0 if the line is not on overflow)[:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.topo_vect` representation as a vector of the topology [for each element
            it gives its bus]. See :func:`grid2op.Backend.Backend.get_topo_vect` for more information.
        #. :attr:`BaseObservation.time_before_cooldown_line` representation of the cooldown time on the powerlines
            [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.time_before_cooldown_sub` representation of the cooldown time on the substations
            [:attr:`grid2op.Space.GridObjects.n_sub` elements]
        #. :attr:`BaseObservation.time_next_maintenance` number of timestep before the next maintenance (-1 means
            no maintenance are planned, 0 a maintenance is in operation) [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.duration_next_maintenance` duration of the next maintenance. If a maintenance
            is taking place, this is the number of timestep before it ends. [:attr:`grid2op.Space.GridObjects.n_line` elements]
        #. :attr:`BaseObservation.target_dispatch` the target dispatch for each generator
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        #. :attr:`BaseObservation.actual_dispatch` the actual dispatch for each generator
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        #. :attr:`BaseObservation.storage_charge` the actual state of charge of each storage unit
            [:attr:`grid2op.Space.GridObjects.n_storage` elements]
        #. :attr:`BaseObservation.storage_power_target` the production / consumption of setpoint of each storage unit
            [:attr:`grid2op.Space.GridObjects.n_storage` elements]
        #. :attr:`BaseObservation.storage_power` the realized production / consumption of each storage unit
            [:attr:`grid2op.Space.GridObjects.n_storage` elements]
        #. :attr:`BaseObservation.gen_p_before_curtail` : the theoretical generation that would have happened
            if no generator from renewable energy sources have been performed (in MW)
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        #. :attr:`BaseObservation.curtailment` : the current curtailment applied
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        #. :attr:`BaseObservation.is_alarm_illegal` whether the last alarm has been illegal (due to budget
            constraint) [``bool``], 
            .. warning: /!\\\\ Only valid with "l2rpn_icaps_2021" environment /!\\\\ 
        #. :attr:`BaseObservation.curtailment_limit` : the current curtailment limit (if any)
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        #. :attr:`BaseObservation.time_since_last_alarm` number of step since the last alarm has been raised
            successfully [``int``]
            .. warning: /!\\\\ Only valid with "l2rpn_icaps_2021" environment /!\\\\ 
        #. :attr:`BaseObservation.last_alarm` : for each alarm zone, gives the last step at which an alarm has
            been successfully raised at this zone 
            .. warning: /!\\\\ Only valid with "l2rpn_icaps_2021" environment /!\\\\
            [:attr:`grid2op.Space.GridObjects.dim_alarms` elements]
        #. :attr:`BaseObservation.attention_budget` : the current attention budget
            [``int``]
        #. :attr:`BaseObservation.was_alarm_used_after_game_over` : was the last alarm used to compute anything related
            to the attention budget when there was a game over (can only be set to ``True`` if the observation
            corresponds to a game over), warning: /!\\\\ Only valid with "l2rpn_icaps_2021" environment /!\\\\ 
            [``bool``]
        #. :attr:`BaseObservation.is_alarm_illegal` whether the last alert has been illegal (due to budget
            constraint) [``bool``]
        #. :attr:`BaseObservation.curtailment_limit` : the current curtailment limit (if any)
            [:attr:`grid2op.Space.GridObjects.n_gen` elements]
        #. :attr:`BaseObservation.curtailment_limit_effective` Limit (in ratio of gen_pmax) imposed on 
            each renewable generator effectively imposed by the environment.
        #. :attr:`BaseObservation.current_step` the number of steps since the beginning of the episode (it's
            0 for the observation after a call to `env.reset()`)
        #. :attr:`BaseObservation.max_step` maximum number of steps that can be done by the environment.
            When :attr:`BaseObservation.current_step` is  :attr:`BaseObservation.max_step` the the environment
            is done.
        #. :attr:`BaseObservation.delta_time` Amount of time (in minutes) represented by a step. In general, there
            are the equivalent of 5 minutes between two steps.
        #. :attr:`BaseObservation.gen_margin_up` From how much can you increase each generators production between this
            step and the next.
        #. :attr:`BaseObservation.gen_margin_down` From how much can you decrease each generators production between this
            step and the next.   
        #. :attr:`BaseObservation.active_alert` This attribute gives the lines "under alert" at the given observation.
        #. :attr:`BaseObservation.time_since_last_alert`  Give the time since an alert has been raised for each powerline.
        #. :attr:`BaseObservation.alert_duration` Give the time since an alert has started for all attackable line.
        #. :attr:`BaseObservation.total_number_of_alert` Total number of alerts since the beginning of the episode sent by 
            the agent
        #. :attr:`BaseObservation.time_since_last_attack` For each attackable line `i` it counts the number of steps since the powerline has
            been attacked
        #. :attr:`BaseObservation.was_alert_used_after_attack` For each attackable line `i` it says if an alert has been used or not
            for the computation of the reward: +1 means "used and the alert was correct", -1 means "used and the alert was not correct"
            and 0 means "not used" 
        #. :attr:`BaseObservation.attack_under_alert` For each attackable line `i` it says if an alert has been sent (+1) or not (-1)
            for each attackable line currently under attack.
        #. :attr:`BaseObservation.theta_or` voltage angle at the origin of each powerline
            [:attr:`grid2op.Space.GridObjects.n_line` elements].
        #. :attr:`BaseObservation.theta_ex` voltage angle at the extremity of each powerline
            [:attr:`grid2op.Space.GridObjects.n_line` elements].
        #. :attr:`BaseObservation.load_theta` voltage angle at each load
            [:attr:`grid2op.Space.GridObjects.n_load` elements].
        #. :attr:`BaseObservation.gen_theta` voltage angle at each generator
            [:attr:`grid2op.Space.GridObjects.n_gen` elements].
        #. :attr:`BaseObservation.storage_theta` voltage angle at each storage unit
            [:attr:`grid2op.Space.GridObjects.n_storage` elements].

            These attributes are set to zero when the backend does not support voltage
            angles; see :attr:`BaseObservation.support_theta`.

            Voltage angles use the reference selected by the backend. A uniform shift
            of every angle represents the same grid state, so users can subtract a
            common reference or statistic when reference-invariant inputs are needed.

    """

    attr_list_vect = [
        "year",
        "month",
        "day",
        "hour_of_day",
        "minute_of_hour",
        "day_of_week",
        "gen_p",
        "gen_q",
        "gen_v",
        "load_p",
        "load_q",
        "load_v",
        "p_or",
        "q_or",
        "v_or",
        "a_or",
        "p_ex",
        "q_ex",
        "v_ex",
        "a_ex",
        "rho",
        "line_status",
        "timestep_overflow",
        "topo_vect",
        "time_before_cooldown_line",
        "time_before_cooldown_sub",
        "time_next_maintenance",
        "duration_next_maintenance",
        "target_dispatch",
        "actual_dispatch",
        "storage_charge",
        "storage_power_target",
        "storage_power",
        "gen_p_before_curtail",
        "curtailment",
        "curtailment_limit",
        "curtailment_limit_effective",  # starting grid2op version 1.6.6
        "is_alarm_illegal",
        "time_since_last_alarm",
        "last_alarm",
        "attention_budget",
        "was_alarm_used_after_game_over",
        "_shunt_p",
        "_shunt_q",
        "_shunt_v",
        "_shunt_bus",  # starting from grid2op version 1.6.0
        "current_step",
        "max_step",  # starting from grid2op version 1.6.4
        "delta_time",  # starting grid2op version 1.6.5
        "gen_margin_up",
        "gen_margin_down",  # starting grid2op version 1.6.6
        # line alert (starting grid2Op 1.9.1, for compatible envs)
        "active_alert",
        "attack_under_alert",
        "time_since_last_alert",
        "alert_duration",
        "total_number_of_alert",
        "time_since_last_attack",
        "was_alert_used_after_attack",
        # ~slack (>= 1.11.0)
        "gen_p_delta",
        # detachment (>= 1.11.0)
        "load_detached",
        "gen_detached",
        "storage_detached",
        "load_p_detached",
        "load_q_detached",
        "gen_p_detached",
        "storage_p_detached",
        # protections (>= 1.11.0)
        "timestep_protection_engaged",
        # voltage angles (>= 1.12.6)
        "theta_or",
        "theta_ex",
        "load_theta",
        "gen_theta",
        "storage_theta",
    ]
    attr_list_json = [
        "_thermal_limit",
        "support_theta",
    ]
    attr_list_set = set(attr_list_vect)

    def __init__(self,
                 obs_env=None,
                 action_helper=None,
                 random_prng=None,
                 kwargs_env=None,
                 **kwargs):

        BaseObservation.__init__(
            self,
            obs_env=obs_env,
            action_helper=action_helper,
            random_prng=random_prng,
            kwargs_env=kwargs_env,
            **kwargs
        )
        self._dictionnarized = None

    def update(self, env, with_forecast=True):
        # reset the matrices
        self._reset_matrices()
        self.reset()
        self._update_obs_complete(env, with_forecast=with_forecast)
