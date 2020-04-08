from grid2op.Action import TopologyAndDispatchAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend

config = {
    "backend": PandaPowerBackend,
    "action_class": TopologyAndDispatchAction,
    "observation_class": None,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "thermal_limits": [
        3.84900179e+02,
        3.84900179e+02,
        2.28997102e+05,
        2.28997102e+05,
        2.28997102e+05,
        1.52664735e+04,
        2.28997102e+05,
        3.84900179e+02,
        3.84900179e+02,
        1.83285800e+02,
        3.84900179e+02,
        3.84900179e+02,
        2.28997102e+05,
        2.28997102e+05,
        6.93930612e+04,
        3.84900179e+02,
        3.84900179e+02,
        2.40562612e+02,
        3.40224266e+03,
        3.40224266e+03
    ],
    "names_chronics_to_grid": None
}
