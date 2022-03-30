# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation


class ActionICAPS2021(PlayableAction):
    authorized_keys = {
        "set_line_status",
        "change_line_status",
        "set_bus",
        "change_bus",
        "redispatch",
        "curtail",
        "raise_alarm",
        }

    attr_list_vect = ['_set_line_status',
                      '_switch_line_status',
                      '_set_topo_vect',
                      '_change_bus_vect', 
                      '_redispatch',
                      '_storage_power',
                      '_curtail',
                      '_raise_alarm']
    attr_list_set = set(attr_list_vect)
    pass


class ObservationICAPS2021(CompleteObservation):
    attr_list_vect = ['year',
                      'month',
                      'day',
                      'hour_of_day',
                      'minute_of_hour',
                      'day_of_week',
                      'gen_p',
                      'gen_q',
                      'gen_v',
                      'load_p',
                      'load_q',
                      'load_v',
                      'p_or',
                      'q_or',
                      'v_or',
                      'a_or',
                      'p_ex',
                      'q_ex',
                      'v_ex',
                      'a_ex',
                      'rho',
                      'line_status',
                      'timestep_overflow',
                      'topo_vect',
                      'time_before_cooldown_line',
                      'time_before_cooldown_sub',
                      'time_next_maintenance',
                      'duration_next_maintenance',
                      'target_dispatch',
                      'actual_dispatch',
                      'storage_charge',
                      'storage_power_target',
                      'storage_power',
                      'gen_p_before_curtail',
                      'curtailment',
                      'curtailment_limit',
                      'is_alarm_illegal',
                      'time_since_last_alarm',
                      'last_alarm',
                      'attention_budget',
                      'was_alarm_used_after_game_over',
                      '_shunt_p',
                      '_shunt_q',
                      '_shunt_v',
                      '_shunt_bus'
    ]
    
    attr_list_json = [
        "current_step",
        "max_step",
        "delta_time",
        "gen_margin_up",
        "gen_margin_down",
        "_thermal_limit",
        "support_theta",
        "theta_or",
        "theta_ex",
        "load_theta",
        "gen_theta",
        "storage_theta",
    ]
    attr_list_set = set(attr_list_vect)
    