from grid2op.Action import TopologyAction
from grid2op.Reward import L2RPNReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import ChangeNothing
from grid2op.Backend import PandaPowerBackend

config = {
    "backend": PandaPowerBackend,
    "action_class": TopologyAction,
    "observation_class": None,
    "reward_class": L2RPNReward,
    "gamerules_class": DefaultRules,
    "chronics_class": ChangeNothing,
    "volagecontroler_class": None,
    "thermal_limits": {
        "0_1_0": 200.0,
        "0_2_1": 300.0,
        "0_3_2": 500.0,
        "0_4_3": 600.0,
        "1_2_4": 700.0,
        "2_3_5": 800.0,
        "2_3_6": 900.0,
        "3_4_7": 1000.0,
    },
    "names_chronics_to_grid": None,
}
