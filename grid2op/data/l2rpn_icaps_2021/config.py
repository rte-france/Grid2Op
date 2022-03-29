# Copyright (c) 2019-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Action import PlayableAction, PowerlineSetAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import AlarmReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend
from grid2op.Opponent import GeometricOpponent, BaseActionBudget
from grid2op.operator_attention import LinearAttentionBudget


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
    
    
lines_attacked = [
    "62_58_180",
    "62_63_160",
    "48_50_136",
    "48_53_141",
    "41_48_131",
    "39_41_121",
    "43_44_125",
    "44_45_126",
    "34_35_110",
    "54_58_154",
]

opponent_attack_cooldown = 12  # 1 hour, 1 hour being 12 time steps
opponent_attack_duration = 96  # 8 hours at maximum
opponent_budget_per_ts = (
    0.17  # opponent_attack_duration / opponent_attack_cooldown + epsilon
)
opponent_init_budget = 144.0  # no need to attack straightfully, it can attack starting at midday the first day

config = {
    "backend": PandaPowerBackend,
    "action_class": ActionICAPS2021,
    "observation_class": ObservationICAPS2021,
    "reward_class": AlarmReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "names_chronics_to_grid": None,
    "thermal_limits": [
        60.9,
        231.9,
        272.6,
        212.8,
        749.2,
        332.4,
        348.0,
        414.4,
        310.1,
        371.4,
        401.2,
        124.3,
        298.5,
        86.4,
        213.9,
        160.8,
        112.2,
        291.4,
        489.0,
        489.0,
        124.6,
        196.7,
        191.9,
        238.4,
        174.2,
        105.6,
        143.7,
        293.4,
        288.9,
        107.7,
        415.5,
        148.2,
        124.2,
        154.4,
        85.9,
        106.5,
        142.0,
        124.0,
        130.2,
        86.2,
        278.1,
        182.0,
        592.1,
        173.1,
        249.8,
        441.0,
        344.2,
        722.8,
        494.6,
        494.6,
        196.7,
        151.8,
        263.4,
        364.1,
        327.0,
        370.5,
        441.0,
        300.3,
        656.2,
    ],
    "opponent_attack_cooldown": opponent_attack_cooldown,
    "opponent_attack_duration": opponent_attack_duration,
    "opponent_budget_per_ts": opponent_budget_per_ts,
    "opponent_init_budget": opponent_init_budget,
    "opponent_action_class": PowerlineSetAction,
    "opponent_class": GeometricOpponent,
    "opponent_budget_class": BaseActionBudget,
    "kwargs_opponent": {
        "lines_attacked": lines_attacked,
        "attack_every_xxx_hour": 24,
        "average_attack_duration_hour": 4,
        "minimum_attack_duration_hour": 1,
    },
    "has_attention_budget": True,
    "attention_budget_class": LinearAttentionBudget,
    "kwargs_attention_budget": {
        "max_budget": 3.0,
        "budget_per_ts": 1.0 / (12.0 * 16),
        "alarm_cost": 1.0,
        "init_budget": 2.0,
    },
}
