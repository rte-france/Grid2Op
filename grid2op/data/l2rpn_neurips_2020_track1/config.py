from grid2op.Action import TopologyAndDispatchAction, PowerlineSetAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend
from grid2op.Opponent import RandomLineOpponent, BaseActionBudget
config = {
    "backend": PandaPowerBackend,
    "action_class": TopologyAndDispatchAction,
    "observation_class": None,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "names_chronics_to_grid": None,
    "thermal_limits": [60.9, 231.9, 272.6, 212.8, 749.2, 332.4, 348., 414.4, 310.1,
                       371.4, 401.2, 124.3, 298.5, 86.4, 213.9, 160.8, 112.2, 291.4,
                       489., 489., 124.6, 196.7, 191.9, 238.4, 174.2, 105.6, 143.7,
                       293.4, 288.9, 107.7, 415.5, 148.2, 124.2, 154.4, 85.9, 106.5,
                       142., 124., 130.2, 86.2, 278.1, 182., 592.1, 173.1, 249.8,
                       441., 344.2, 722.8, 494.6, 494.6, 196.7, 151.8, 263.4, 364.1,
                       327., 370.5, 441., 300.3, 656.2],
    "opponent_attack_cooldown": 12*24,
    "opponent_attack_duration": 12*4,
    "opponent_budget_per_ts": 0.5,
    "opponent_init_budget": 0.,
    "opponent_action_class": PowerlineSetAction,
    "opponent_class": RandomLineOpponent,
    "opponent_budget_class": BaseActionBudget,
    "kwargs_opponent": {"lines_attacked": ["62_58_180",
        "62_63_160",
        "48_50_136",
        "48_53_141",
        "41_48_131",
        "39_41_121",
        "43_44_125",
        "44_45_126",
        "34_35_110",
        "54_58_154"]}
}
