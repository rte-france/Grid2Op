# making sure test can be ran from:
# root package directory
# Grid2Op subdirectory
# Grid2Op/tests subdirectory
import sys
import os

sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./grid2op'))
sys.path.insert(0, os.path.abspath('../grid2op/'))

PATH_DATA_TEST = os.path.abspath("../data/")
PATH_CHRONICS = os.path.abspath("../data")
if not os.path.exists(os.path.join(PATH_DATA_TEST, "chronics_with_forecast")):
    PATH_DATA_TEST = os.path.abspath("./data/")
    PATH_CHRONICS = os.path.abspath("./data/")
    if not os.path.exists(os.path.join(PATH_DATA_TEST, "chronics_with_forecast")):
        PATH_DATA_TEST = os.path.abspath("./data/")
        PATH_CHRONICS = os.path.abspath("./data")
        if not os.path.exists(os.path.join(PATH_DATA_TEST, "chronics_with_forecast")):  # I am lost
            raise RuntimeError("Impossible to find the test data folder")
PATH_DATA_TEST_PP = os.path.abspath(os.path.join(PATH_DATA_TEST, "test_PandaPower"))