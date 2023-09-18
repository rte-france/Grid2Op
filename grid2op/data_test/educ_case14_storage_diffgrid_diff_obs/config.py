from grid2op.Action import PowerlineChangeDispatchAndStorageAction
from grid2op.Reward import L2RPNReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend
from grid2op.l2rpn_utils import ActionIDF2023, ObservationIDF2023

config = {
    "backend": PandaPowerBackend,
    "action_class": PowerlineChangeDispatchAndStorageAction,
    "observation_class": ObservationIDF2023,
    "action_class": ActionIDF2023,
    "reward_class": L2RPNReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "thermal_limits": [
        541.0,
        450.0,
        375.0,
        636.0,
        175.0,
        285.0,
        335.0,
        657.0,
        496.0,
        827.0,
        442.0,
        641.0,
        840.0,
        156.0,
        664.0,
        235.0,
        119.0,
        179.0,
        1986.0,
        1572.0,
    ],
    "names_chronics_to_grid": None,
}
