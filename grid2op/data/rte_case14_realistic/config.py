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
        384.900179,
        384.900179,
        380.0,
        380.0,
        157.0,
        380.0,
        380.0,
        1077.7205012,
        461.8802148,
        769.80036,
        269.4301253,
        384.900179,
        760.0,
        380.0,
        760.0,
        384.900179,
        230.9401074,
        170.79945452,
        3402.24266,
        3402.24266
    ],
    "names_chronics_to_grid": None
}
