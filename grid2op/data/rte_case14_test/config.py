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
	352.8251645,
	352.8251645,
	183197.68156979,
	183197.68156979,
	183197.68156979,
	12213.17877132,
	183197.68156979,
	352.8251645,
	352.8251645,
	352.8251645,
	352.8251645,
	352.8251645,
	183197.68156979,
	183197.68156979,
	183197.68156979,
	352.8251645,
	352.8251645,
	352.8251645,
	2721.79412618,
	2721.79412618
    ],
    "names_chronics_to_grid": None
}
