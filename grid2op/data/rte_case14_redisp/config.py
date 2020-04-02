from grid2op.Action import TopoAndRedispAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend

config = {
    "backend": PandaPowerBackend,
    "action_class": TopoAndRedispAction,
    "observation_class": None,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "graph_layout": [
	[-280, -81],
	[-100, -270],
	[366, -270],
	[366, -54],
	[-64, -54],
	[-64, 54],
	[450, 0],
	[550, 0],
	[326, 54],
	[222, 108],
	[79, 162],
	[-170, 270],
	[-64, 270],
	[222, 216]
    ],
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
