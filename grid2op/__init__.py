"""
Grid2Op
Document will be made later on.

"""
import os
import pkg_resources

__version__ = "0.0.1"

__all__ = ['Action', "BackendPandaPower", "Agent", "Backend", "ChronicsHandler", "Environment", "Exceptions",
           "Observation", "Parameters", "GameRules", "Reward", "Runner", "main"]


CASE_14_FILE = os.path.abspath(os.path.join(pkg_resources.resource_filename(__name__, "data"), "test_PandaPower", "test_case14.json"))
CHRONICS_FODLER = os.path.abspath(os.path.join(pkg_resources.resource_filename(__name__, "data")))
CHRONICS_MLUTIEPISODE = os.path.join(CHRONICS_FODLER, "test_multi_chronics")

NAMES_CHRONICS_TO_BACKEND = {"loads": {"2_C-10.61": 'load_1_0', "3_C151.15": 'load_2_1',
                                       "14_C63.6": 'load_13_2', "4_C-9.47": 'load_3_3',
                                       "5_C201.84": 'load_4_4',
                                       "6_C-6.27": 'load_5_5', "9_C130.49": 'load_8_6',
                                       "10_C228.66": 'load_9_7',
                                       "11_C-138.89": 'load_10_8', "12_C-27.88": 'load_11_9',
                                       "13_C-13.33": 'load_12_10'},
                             "lines": {'1_2_1': '0_1_0', '1_5_2': '0_4_1', '9_10_16': '8_9_2',
                                       '9_14_17': '8_13_3',
                                       '10_11_18': '9_10_4', '12_13_19': '11_12_5', '13_14_20': '12_13_6',
                                       '2_3_3': '1_2_7', '2_4_4': '1_3_8', '2_5_5': '1_4_9',
                                       '3_4_6': '2_3_10',
                                       '4_5_7': '3_4_11', '6_11_11': '5_10_12', '6_12_12': '5_11_13',
                                       '6_13_13': '5_12_14', '4_7_8': '3_6_15', '4_9_9': '3_8_16',
                                       '5_6_10': '4_5_17',
                                       '7_8_14': '6_7_18', '7_9_15': '6_8_19'},
                             "prods": {"1_G137.1": 'gen_0_4', "3_G36.31": "gen_2_1", "6_G63.29": "gen_5_2",
                                       "2_G-56.47": "gen_1_0", "8_G40.43": "gen_7_3"},
                             }