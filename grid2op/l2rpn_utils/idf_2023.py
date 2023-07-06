# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation

class ActionIDF2023(PlayableAction):
    authorized_keys = {
        "set_line_status",
        "change_line_status",
        "set_bus",
        "change_bus",
        "redispatch",
        "set_storage",
        "curtail",
        "raise_alert",
        }

    attr_list_vect = [
        "_set_line_status",
        "_switch_line_status",
        "_set_topo_vect",
        "_change_bus_vect",
        "_redispatch",
        "_storage_power",
        "_curtail",
        "_raise_alert",
        ]
    attr_list_set = set(attr_list_vect)
    pass
    
class ObservationIDF2023(CompleteObservation):
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
    ]
    attr_list_json = [
        "_thermal_limit",
        "support_theta",
        "theta_or",
        "theta_ex",
        "load_theta",
        "gen_theta",
        "storage_theta"
    ]
    attr_list_set = set(attr_list_vect)
