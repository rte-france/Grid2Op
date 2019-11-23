"""
Grid2Op
Document will be made later on.

"""
import os
import pkg_resources

__version__ = "0.3.0"

__all__ = ['Action', "BackendPandaPower", "Agent", "Backend", "ChronicsHandler", "Environment", "Exceptions",
           "Observation", "Parameters", "GameRules", "Reward", "Runner", "main"]

from .MakeEnv import make
from .MakeEnv import CASE_14_FILE, CHRONICS_FODLER, CHRONICS_MLUTIEPISODE, NAMES_CHRONICS_TO_BACKEND


# TODO integrate Balthazar's code in the data generation process, to have a better environment.
# At least for the case 14
