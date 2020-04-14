from grid2op.Action import TopologyAction
from grid2op.Reward import L2RPNReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import ReadPypowNetData
from grid2op.Backend import PandaPowerBackend

config = {
    "backend": PandaPowerBackend,
    "action_class": TopologyAction,
    "observation_class": None,
    "reward_class": L2RPNReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": ReadPypowNetData,
    "volagecontroler_class": None,
    "thermal_limits": None,
    "names_chronics_to_grid": {
        "loads": {
            "2_C-10.61": "load_1_0",
            "3_C151.15": "load_2_1",
            "14_C63.6": "load_13_10",
            "4_C-9.47": "load_3_2",
            "5_C201.84": "load_4_3",
            "6_C-6.27": "load_5_4",
            "9_C130.49": "load_8_5",
            "10_C228.66": "load_9_6",
            "11_C-138.89": "load_10_7",
            "12_C-27.88": "load_11_8",
            "13_C-13.33": "load_12_9"
        },
        "lines": {
            "1_2_1": "0_1_0",
            "1_5_2": "0_4_1",
            "9_10_16": "8_9_16",
            "9_14_17": "8_13_15",
            "10_11_18": "9_10_17",
            "12_13_19": "11_12_18",
            "13_14_20": "12_13_19",
            "2_3_3": "1_2_2",
            "2_4_4": "1_3_3",
            "2_5_5": "1_4_4",
            "3_4_6": "2_3_5",
            "4_5_7": "3_4_6",
            "6_11_11": "5_10_12",
            "6_12_12": "5_11_11",
            "6_13_13": "5_12_10",
            "4_7_8": "3_6_7",
            "4_9_9": "3_8_8",
            "5_6_10": "4_5_9",
            "7_8_14": "6_7_13",
            "7_9_15": "6_8_14"
        },
        "prods": {
            "1_G137.1": "gen_0_4",
            "3_G36.31": "gen_1_0",
            "6_G63.29": "gen_2_1",
            "2_G-56.47": "gen_5_2",
            "8_G40.43": "gen_7_3"
        }
    }
}
