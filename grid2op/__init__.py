"""
Grid2Op
Document will be made later on.

"""
import os
import pkg_resources

__version__ = '0.5.8'

__all__ = ['Action', "BackendPandaPower", "Agent", "Backend", "ChronicsHandler", "Environment", "Exceptions",
           "Observation", "Parameters", "GameRules", "Reward", "Runner", "main", "Utils", "PlotPlotly", "EpisodeData",
           "DownloadDataset", "PlotMatplotlib", "VoltageControler"]

from .MakeEnv import make

# reference case 14
from .MakeEnv import CASE_14_FILE, CHRONICS_FODLER, CHRONICS_MLUTIEPISODE, NAMES_CHRONICS_TO_BACKEND
# small example
from .MakeEnv import EXAMPLE_CHRONICSPATH, EXAMPLE_CASEFILE
# case 14 as in L2RPN 2019 edition
from .MakeEnv import L2RPN2019_CASEFILE, L2RPN2019_DICT_NAMES, ReadPypowNetData


# TODO integrate Balthazar's code in the data generation process, to have a better environment.
# At least for the case 14
