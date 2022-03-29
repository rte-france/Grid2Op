from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecastsWithMaintenance
from grid2op.Backend import PandaPowerBackend


class ActionWCCI2020(PlayableAction):
    authorized_keys = {
        "set_line_status",
        "change_line_status",
        "set_bus",
        "change_bus",
        }

    attr_list_vect = [
        "_set_line_status",
        "_switch_line_status",
        "_set_topo_vect",
        "_change_bus_vect"
        ]
    attr_list_set = set(attr_list_vect)
    pass
    
    
class ObservationWCCI2020(CompleteObservation):
    attr_list_vect = [
        'year',
        'month',
        'day',
        'hour_of_day',
        'minute_of_hour',
        'day_of_week',
        "gen_p",
        "gen_q",
        "gen_v",
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
        'actual_dispatch'
    ]
    attr_list_json = [
        "storage_charge",
        "storage_power_target",
        "storage_power",
        "gen_p_before_curtail",
        "curtailment",
        "curtailment_limit",
        "curtailment_limit_effective",
        "_shunt_p",
        "_shunt_q",
        "_shunt_v",
        "_shunt_bus",
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
    
    
config = {
    "backend": PandaPowerBackend,
    "action_class": ActionWCCI2020,
    "observation_class": ObservationWCCI2020,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecastsWithMaintenance,
    "volagecontroler_class": None,
    "names_chronics_to_grid": {},
    "thermal_limits": [
        43.3,
        205.2,
        341.2,
        204.0,
        601.4,
        347.1,
        319.6,
        301.4,
        330.3,
        274.1,
        307.4,
        172.3,
        354.3,
        127.9,
        174.9,
        152.6,
        81.8,
        204.3,
        561.5,
        561.5,
        98.7,
        179.8,
        193.4,
        239.9,
        164.8,
        100.4,
        125.7,
        278.2,
        274.0,
        89.9,
        352.1,
        157.1,
        124.4,
        154.6,
        86.1,
        106.7,
        148.5,
        129.6,
        136.1,
        86.0,
        313.2,
        198.5,
        599.1,
        206.8,
        233.7,
        395.8,
        516.7,
        656.4,
        583.0,
        583.0,
        263.1,
        222.6,
        322.8,
        340.6,
        305.2,
        360.1,
        395.8,
        274.2,
        605.5,
    ],
}
