from grid2op.Action import PlayableAction, PowerlineSetAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecastsWithMaintenance
from grid2op.Backend import PandaPowerBackend
from grid2op.Opponent import GeometricOpponent, BaseActionBudget

try:
    from grid2op.l2rpn_utils import ActionWCCI2022, ObservationWCCI2022
except ImportError:
    from grid2op.Action import PlayableAction
    from grid2op.Observation import CompleteObservation
    import warnings
    warnings.warn("The grid2op version you are trying to use is too old for this environment. Please upgrade it.")
    ActionWCCI2022 = PlayableAction
    ObservationWCCI2022 = CompleteObservation
    
lines_attacked = ["26_31_106",
                  "21_22_93",
                  "17_18_88",
                  "4_10_162",
                  "12_14_68",
                  "14_32_108",
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
                  "74_117_81",
                  "80_79_175",
                  "93_95_43",
                  "88_91_33",
                  "91_92_37",
                  "99_105_62",
                  "102_104_61"]

opponent_attack_cooldown = 12  # 1 hour, 1 hour being 12 time steps
opponent_attack_duration = 96  # 8 hours at maximum
opponent_budget_per_ts = (
    0.17  # opponent_attack_duration / opponent_attack_cooldown + epsilon
)
opponent_init_budget = 144.0  # no need to attack straightfully, it can attack starting at midday the first day

config = {
    "backend": PandaPowerBackend,
    "action_class": ActionWCCI2022,
    "observation_class": ObservationWCCI2022,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecastsWithMaintenance,
    "volagecontroler_class": None,
    "names_chronics_to_grid": None,
    "thermal_limits": [ 233.4,  354.4,  792.7,  550.2,  572.2,  557.2,    8. ,  480. ,
                        567.4,  681.8,  357.6,  336.9,  819. ,  419.2,  304.2,  626.2,
                        256.1,  300.1,  132.7,  165.9,  841. , 1105.5,  428.2,  555.2,
                        224.2,  374.4,  285.6,  429.8,  253.1,  479.6,  238.3,  452.6,
                        312.9,  627.8,  196.1,  360.9,  317.1,  325.1,  352.6,  347.3,
                        565.5,  495.7, 1422.9,  479.8,  646.9, 1603.9,  364.1, 1498.4,
                        278. ,  866.2, 1667.7,  569.6, 1350.2, 1478. ,  380.8,  282.4,
                        246.9,  301.3,  766.9,  401.2,  306.9,  314.4,  333.4,  748.9,
                        513.4,  255.8,  513. ,  268.5,  219. ,  492. ,  420.4,  417.4,
                        637.8,  571.9,  593.8,  273.7,  247. ,  385.3,  283.4,  251.2,
                        210.8,  473.9,  408.5,  162.7,  602.2, 1098.6,  205. ,  546. ,
                        435.9,  191.4,  424.1,  106.2,  149.2,  184.9, 1146.1, 1117.8,
                        569.6,  800.2,  380.3,  292.1,  636.5,  487.5,  490.9,  207.4,
                        590.6,  243.8,  466. ,  698.2,  385. ,  351.7,   60.9,  231.9,
                        340.8,  212.8,  749.2,  332.4,  348. ,  798. ,  398.3,  414.4,
                        341.1,  371.4,  401.2,  298.3,  343.3,  267.8,  213.9,  160.8,
                        112.2,  458.9,  349.7,  489. ,  489. ,  180.7,  196.7,  191.9,
                        238.4,  174.2,  105.6,  143.7,  393.6,  293.4,  288.9,  107.7,
                        623.2,  252.9,  118.3,  154.4,  111.7,  106.5,  177.5,  655.8,
                        161.2,  169.3,  120.7,  389.3,  291.2,  592.1,  277. ,  412.2,
                        441. ,  671.2,  609. ,  867.4,  494.6,  494.6,  196.7,  167. ,
                        263.4,  364.1,  359.7,  803.2,  589. ,  887.2,  615.2, 1096.8,
                        306.9,  472.6,  546.6,  370.5,  441. ,  300.3,  656.2, 1346. ,
                        1246.5, 1196.5],
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
}
