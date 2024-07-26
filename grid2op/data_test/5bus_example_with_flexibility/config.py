from grid2op.Action import CompleteAction
from grid2op.Reward import L2RPNReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend

config = {
    "backend": PandaPowerBackend,
    "action_class": CompleteAction,
    "observation_class": None,
    "reward_class": L2RPNReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "thermal_limits": None,
    "names_chronics_to_grid": None,
}
